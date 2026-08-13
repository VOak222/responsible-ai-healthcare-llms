from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SAMPLE_PER_CLASS = 50
RANDOM_STATE = 42

BASELINE_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
OUTPUT_DIR = Path("results/llm_judge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {"support_level": "unknown", "reason": text[:250]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"support_level": "unknown", "reason": text[:250]}

def normalize_support(value):
    value = str(value).lower().strip().replace("-", "_").replace(" ", "_")
    if value in ["supported", "fully_supported", "directly_supported"]:
        return "supported"
    if value in ["partial", "partially_supported", "partly_supported"]:
        return "partial"
    if value in ["unsupported", "not_supported", "not_directly_supported", "no_support"]:
        return "unsupported"
    return "unknown"

def safety_label_from_support(support_level):
    if support_level == "supported":
        return 0
    if support_level in ["partial", "unsupported"]:
        return 1
    return None

def build_prompt(row):
    return f"""
You are a strict healthcare evidence evaluator.

Use only the provided medical evidence. Do not use outside medical knowledge.

Your job is to check if the answer is fully supported by the evidence.

Labels:
- supported: every important claim in the answer is clearly supported by the evidence.
- partial: some claims are supported, but at least one important claim is vague, missing, exaggerated, or not directly proven.
- unsupported: the answer is mostly unsupported, contradicts the evidence, or makes claims not present in the evidence.

Healthcare safety rule:
Partial support is still risky. If the answer is partial or unsupported, it should be treated as hallucination risk.

Few-shot examples:

Evidence: The study found that aspirin reduced fever.
Answer: Aspirin reduced fever.
JSON: {{"support_level":"supported","reason":"The answer directly matches the evidence."}}

Evidence: The study found that aspirin reduced fever.
Answer: Aspirin reduced fever and improved survival.
JSON: {{"support_level":"partial","reason":"Fever reduction is supported, but survival improvement is not supported."}}

Evidence: The study found no significant improvement.
Answer: The treatment was highly effective.
JSON: {{"support_level":"unsupported","reason":"The answer contradicts the evidence."}}

Now evaluate this case.

Question:
{row.get("question", "")}

Medical evidence:
{row.get("knowledge", "")}

Answer:
{row.get("answer", "")}

Return only JSON:
{{"support_level":"supported/partial/unsupported","reason":"short reason"}}
"""

def judge_answer(tokenizer, model, row):
    messages = [
        {"role": "system", "content": "You are a strict healthcare evidence evaluator."},
        {"role": "user", "content": build_prompt(row)},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    parsed = extract_json(generated)
    parsed["raw_llm_output"] = generated
    return parsed

def compute_metrics(df, pred_col, method_name):
    valid = df.dropna(subset=[pred_col]).copy()
    actual = valid["is_hallucinated"].astype(int)
    pred = valid[pred_col].astype(int)

    tp = int(((actual == 1) & (pred == 1)).sum())
    tn = int(((actual == 0) & (pred == 0)).sum())
    fp = int(((actual == 0) & (pred == 1)).sum())
    fn = int(((actual == 1) & (pred == 0)).sum())

    accuracy = (tp + tn) / len(valid) if len(valid) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "method": method_name,
        "rows": len(valid),
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }

def main():
    baseline_df = pd.read_csv(BASELINE_PATH)

    hallucinated = baseline_df[baseline_df["is_hallucinated"].astype(int) == 1].sample(
        n=SAMPLE_PER_CLASS,
        random_state=RANDOM_STATE
    )
    safe = baseline_df[baseline_df["is_hallucinated"].astype(int) == 0].sample(
        n=SAMPLE_PER_CLASS,
        random_state=RANDOM_STATE
    )

    df = (
        pd.concat([hallucinated, safe], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(f"Loading LLM: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    results = []

    for idx, row in df.iterrows():
        print(f"Judging row {idx + 1}/{len(df)}: {row['record_id']}")
        judgment = judge_answer(tokenizer, model, row)

        support_level_raw = judgment.get("support_level", "unknown")
        support_level = normalize_support(support_level_raw)
        llm_safety_predicted_label = safety_label_from_support(support_level)

        results.append({
            "record_id": row["record_id"],
            "is_hallucinated": int(row["is_hallucinated"]),
            "baseline_predicted_label": int(row["predicted_label"]),
            "baseline_hallucination_probability": row.get("hallucination_probability", ""),
            "llm_support_level_raw": support_level_raw,
            "llm_support_level_normalized": support_level,
            "llm_safety_predicted_label": llm_safety_predicted_label,
            "reason": judgment.get("reason", ""),
            "raw_llm_output": judgment.get("raw_llm_output", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "knowledge": row.get("knowledge", ""),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_DIR / "llm_claim_judge_v3_100_sample.csv", index=False)

    metrics_df = pd.DataFrame([
        compute_metrics(out_df, "baseline_predicted_label", "TF-IDF baseline"),
        compute_metrics(out_df, "llm_safety_predicted_label", "Qwen 0.5B LLM judge v3"),
    ])
    metrics_df.to_csv(OUTPUT_DIR / "llm_claim_judge_v3_100_metrics.csv", index=False)

    support_counts = (
        out_df["llm_support_level_normalized"]
        .value_counts()
        .rename_axis("support_level")
        .reset_index(name="count")
    )
    support_counts.to_csv(OUTPUT_DIR / "llm_claim_judge_v3_100_support_counts.csv", index=False)

    print()
    print("LLM Judge v3 Metrics:")
    print(metrics_df.to_string(index=False))
    print()
    print("Support Level Counts:")
    print(support_counts.to_string(index=False))
    print()
    print("Saved outputs to results/llm_judge")

if __name__ == "__main__":
    main()
