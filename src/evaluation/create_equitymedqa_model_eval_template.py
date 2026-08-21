from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/equitymedqa_fairness")
CATEGORY_PATH = OUT_DIR / "equitymedqa_fairness_review_sample.csv"
PAIRED_PATH = OUT_DIR / "equitymedqa_paired_prompt_review_sample.csv"
OUT_PATH = OUT_DIR / "equitymedqa_model_evaluation_template.csv"

category_df = pd.read_csv(CATEGORY_PATH)
paired_df = pd.read_csv(PAIRED_PATH)

rows = []

# Single prompt rows: evaluate if one model answer is fair, safe, and clinically careful.
for _, row in category_df.iterrows():
    rows.append({
        "evaluation_type": "single_prompt_fairness",
        "dataset_name": row.get("dataset_name", ""),
        "record_id": row.get("record_id", ""),
        "fairness_category": row.get("fairness_category", ""),
        "prompt_a": row.get("prompt_a", ""),
        "prompt_b": "",
        "model_name": "",
        "model_response_a": "",
        "model_response_b": "",
        "unsafe_clinical_advice_flag": "",
        "biased_or_stereotyped_language_flag": "",
        "demographic_difference_flag": "",
        "safe_refusal_or_caution_needed": "",
        "overall_fairness_risk": "",
        "notes": ""
    })

# Paired prompt rows: evaluate if two similar prompts get unfairly different answers.
for _, row in paired_df.iterrows():
    rows.append({
        "evaluation_type": "paired_prompt_comparison",
        "dataset_name": row.get("dataset_name", ""),
        "record_id": row.get("record_id", ""),
        "fairness_category": "",
        "prompt_a": row.get("prompt_a", ""),
        "prompt_b": row.get("prompt_b", ""),
        "model_name": "",
        "model_response_a": "",
        "model_response_b": "",
        "unsafe_clinical_advice_flag": "",
        "biased_or_stereotyped_language_flag": "",
        "demographic_difference_flag": "",
        "safe_refusal_or_caution_needed": "",
        "overall_fairness_risk": "",
        "notes": ""
    })

template = pd.DataFrame(rows)
template.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(template))
print()
print(template["evaluation_type"].value_counts().to_string())
