from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/model_expansion")
REPORT_DIR = Path("reports/future_work")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "qwen_gemma_phi_100row_model_comparison.csv"
REPORT_PATH = REPORT_DIR / "qwen_gemma_phi_100row_model_comparison_report.md"

summary = pd.DataFrame([
    {
        "model": "Qwen 1.5B",
        "rows": 100,
        "accuracy": 0.86,
        "precision": 0.790,
        "recall": 0.98,
        "f1_score": 0.875,
        "false_positives": 13,
        "false_negatives": 1,
        "decision": "Best main model so far",
    },
    {
        "model": "Qwen2.5-3B",
        "rows": 100,
        "accuracy": 0.66,
        "precision": 0.598,
        "recall": 0.98,
        "f1_score": 0.743,
        "false_positives": 33,
        "false_negatives": 1,
        "decision": "Conservative secondary reviewer candidate",
    },
    {
        "model": "Gemma 2 2B",
        "rows": 100,
        "accuracy": 0.65,
        "precision": 0.588,
        "recall": 1.00,
        "f1_score": 0.741,
        "false_positives": 35,
        "false_negatives": 0,
        "decision": "Safest recall, but too many false positives",
    },
    {
        "model": "Phi-3.5-mini",
        "rows": 100,
        "accuracy": 0.67,
        "precision": 0.608,
        "recall": 0.96,
        "f1_score": 0.744,
        "false_positives": 31,
        "false_negatives": 2,
        "decision": "Not preferred; slower and missed more hallucinations",
    },
])

summary.to_csv(SUMMARY_PATH, index=False)

report = """# Week 6 Model Expansion: Qwen, Gemma, and Phi Comparison

## Purpose

This comparison checks whether newer or slightly larger local open-source models improve healthcare hallucination detection compared with the current best model, Qwen 1.5B.

All models in this table are compared on the same 100-row Med-HALT balanced sample.

## Results

| Model | Rows | Accuracy | Precision | Recall | F1 | False Positives | False Negatives | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen 1.5B | 100 | 0.86 | 0.790 | 0.98 | 0.875 | 13 | 1 | Best main model so far |
| Qwen2.5-3B | 100 | 0.66 | 0.598 | 0.98 | 0.743 | 33 | 1 | Conservative secondary reviewer candidate |
| Gemma 2 2B | 100 | 0.65 | 0.588 | 1.00 | 0.741 | 35 | 0 | Safest recall, but too many false positives |
| Phi-3.5-mini | 100 | 0.67 | 0.608 | 0.96 | 0.744 | 31 | 2 | Not preferred; slower and missed more hallucinations |

## Interpretation

Qwen 1.5B remains the best main model because it has the strongest balance of accuracy, precision, recall, and F1 score.

Gemma 2 2B had the safest recall result because it produced zero false negatives. However, it over-flagged many supported answers as hallucinated, so it is better as a secondary safety reviewer than as the main model.

Qwen2.5-3B also behaved conservatively and can be used as a secondary reviewer candidate, but it did not improve over Qwen 1.5B.

Phi-3.5-mini did not beat Qwen 1.5B. It missed two hallucinated answers, produced many false positives, and was slower during local evaluation.

## Current Project Decision

Qwen 1.5B remains the main local hallucination detection model.

Gemma 2 2B and Qwen2.5-3B remain useful secondary reviewer candidates.

Phi-3.5-mini is not preferred for the current main model because it did not improve the safety or balance of the system.

Llama 3.2 3B was tested only as a 25-row pilot and was not advanced to the 100-row comparison.

Qwen3-1.7B should be tested next.
"""

REPORT_PATH.write_text(report, encoding="utf-8")

print("Qwen, Gemma, and Phi 100-row comparison complete")
print()
print(summary.to_string(index=False))
print()
print("Saved:", SUMMARY_PATH)
print("Saved:", REPORT_PATH)
