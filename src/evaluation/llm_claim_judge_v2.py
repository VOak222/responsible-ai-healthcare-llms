from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

BASELINE_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
REVIEW_EXAMPLES_PATH = Path("results/manual_review/tp_tn_fp_fn_review_examples.csv")
OUTPUT_DIR = Path("results/llm_judge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
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
    if value in ["partial", "partially_supported"]:
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

Use only the given medical evidence. Do not use outside medical knowledge.

Classification rule:
- supported: the answer is fully and directly supported by the evidence.
- partial: some parts are supported, but some claims are missing, vague, or not directly proven.
- unsupported: the answer is not supported or contains claims not found in the evidence.

Healthcare safety rule:
If an answer is only partial or unsupported, it should be treated as hallucination risk.

Example 1:
Evidence: Aspirin reduced fever in the trial.
Answer: Aspirin reduced fever in the trial.
JSON: {{"support_level":"supported","reason":"The answer directly matches the evidence."}}

Example 2:
Evidence: The study measured blood pressure only.
Answer: The treatment reduced blood pressure and improved survival.
JSON: {{"support_level":"partial","reason":"Blood pressure is mentioned, but survival improvement is not supported."}}

Example 3:
Evidence: The trial found no significant benefit.
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

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = extract_json(generated)
    parsed["raw_llm_output"] = generated
    return parsed

def main():
    baseline_df = pd.read_csv(BASELINE_PATH)

    review_ids = pd.read_csv(REVIEW_EXAMPLES_PATH)["record_id"].drop_duplicates()
    df = baseline_df[baseline_df["record_id"].isin(review_ids)].copy()
    df = df.head(20).reset_index(drop=True)

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
    out_df.to_csv(OUTPUT_DIR / "llm_claim_judge_v2_sample.csv", index=False)

    valid = out_df.dropna(subset=["llm_safety_predicted_label"]).copy()
    valid["llm_correct"] = valid["is_hallucinated"].astype(int) == valid["llm_safety_predicted_label"].astype(int)
    valid["baseline_correct"] = valid["is_hallucinated"].astype(int) == valid["baseline_predicted_label"].astype(int)

    summary = pd.DataFrame([{
        "rows_judged": len(out_df),
        "valid_llm_judgments": len(valid),
        "llm_v2_accuracy_on_sample": valid["llm_correct"].mean(),
        "baseline_accuracy_on_same_sample": valid["baseline_correct"].mean(),
        "supported_count": int((out_df["llm_support_level_normalized"] == "supported").sum()),
        "partial_count": int((out_df["llm_support_level_normalized"] == "partial").sum()),
        "unsupported_count": int((out_df["llm_support_level_normalized"] == "unsupported").sum()),
        "unknown_count": int((out_df["llm_support_level_normalized"] == "unknown").sum()),
    }])

    summary.to_csv(OUTPUT_DIR / "llm_claim_judge_v2_summary.csv", index=False)

    print()
    print("LLM Judge v2 Summary:")
    print(summary.to_string(index=False))
    print()
    print("Saved outputs to results/llm_judge")

if __name__ == "__main__":
    main()
