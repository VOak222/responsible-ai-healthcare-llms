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
        return {
            "llm_label": "unknown",
            "support_level": "unknown",
            "reason": text[:250],
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "llm_label": "unknown",
            "support_level": "unknown",
            "reason": text[:250],
        }

def build_prompt(row):
    return f"""
You are evaluating a healthcare answer for hallucination risk.

Use only the provided medical evidence. Do not use outside knowledge.

Question:
{row.get("question", "")}

Medical evidence:
{row.get("knowledge", "")}

Answer:
{row.get("answer", "")}

Task:
Decide whether the answer is fully supported by the medical evidence. In this healthcare project, mark the answer as hallucinated if it is unsupported, partially supported, vague, missing important evidence, or contains any claim not directly supported by the evidence. Only mark not_hallucinated when the answer is fully supported by the evidence.

Return only JSON in this format:
{{
  "llm_label": "hallucinated" or "not_hallucinated",
  "support_level": "supported" or "partial" or "unsupported",
  "reason": "short reason"
}}
"""

def judge_answer(tokenizer, model, row):
    prompt = build_prompt(row)

    messages = [
        {"role": "system", "content": "You are a careful healthcare hallucination evaluator."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    parsed = extract_json(generated)
    parsed["raw_llm_output"] = generated
    return parsed

def main():
    baseline_df = pd.read_csv(BASELINE_PATH)

    if REVIEW_EXAMPLES_PATH.exists():
        review_ids = pd.read_csv(REVIEW_EXAMPLES_PATH)["record_id"].drop_duplicates()
        df = baseline_df[baseline_df["record_id"].isin(review_ids)].copy()
    else:
        df = baseline_df.sample(n=min(20, len(baseline_df)), random_state=42).copy()

    df = df.head(20).reset_index(drop=True)

    print(f"Loading LLM: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    results = []

    for idx, row in df.iterrows():
        print(f"Judging row {idx + 1}/{len(df)}: {row['record_id']}")
        judgment = judge_answer(tokenizer, model, row)

        actual = int(row["is_hallucinated"])
        llm_label = judgment.get("llm_label", "unknown")
        llm_predicted_label = 1 if llm_label == "hallucinated" else 0 if llm_label == "not_hallucinated" else None

        results.append({
            "record_id": row["record_id"],
            "is_hallucinated": actual,
            "baseline_predicted_label": row.get("predicted_label", ""),
            "baseline_hallucination_probability": row.get("hallucination_probability", ""),
            "llm_label": llm_label,
            "llm_predicted_label": llm_predicted_label,
            "support_level": judgment.get("support_level", "unknown"),
            "reason": judgment.get("reason", ""),
            "raw_llm_output": judgment.get("raw_llm_output", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "knowledge": row.get("knowledge", ""),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_DIR / "llm_claim_judge_sample.csv", index=False)

    valid = out_df.dropna(subset=["llm_predicted_label"]).copy()
    if len(valid) > 0:
        valid["llm_correct"] = valid["is_hallucinated"].astype(int) == valid["llm_predicted_label"].astype(int)
        valid["baseline_correct"] = valid["is_hallucinated"].astype(int) == valid["baseline_predicted_label"].astype(int)

        summary = pd.DataFrame([{
            "rows_judged": len(out_df),
            "valid_llm_judgments": len(valid),
            "llm_accuracy_on_sample": valid["llm_correct"].mean(),
            "baseline_accuracy_on_same_sample": valid["baseline_correct"].mean(),
        }])
    else:
        summary = pd.DataFrame([{
            "rows_judged": len(out_df),
            "valid_llm_judgments": 0,
            "llm_accuracy_on_sample": None,
            "baseline_accuracy_on_same_sample": None,
        }])

    summary.to_csv(OUTPUT_DIR / "llm_claim_judge_summary.csv", index=False)

    print()
    print("LLM Judge Summary:")
    print(summary.to_string(index=False))
    print()
    print("Saved outputs to results/llm_judge")

if __name__ == "__main__":
    main()
