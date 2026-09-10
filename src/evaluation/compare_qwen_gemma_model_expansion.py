from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/model_expansion")
REPORT_DIR = Path("reports/future_work")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "qwen_1_5b_vs_qwen_2_5_3b_vs_gemma_2_2b_summary.csv"
REPORT_PATH = REPORT_DIR / "qwen_1_5b_vs_qwen_2_5_3b_vs_gemma_2_2b_report.md"

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
        "decision": "Useful as conservative secondary reviewer",
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
        "decision": "Safest recall, but too many false positives for main model",
    },
])

summary.to_csv(SUMMARY_PATH, index=False)

report = """# Three-Model Med-HALT Comparison

## Purpose

This comparison checks whether newer small open-source models improve healthcare hallucination detection compared with the current best model, Qwen 1.5B.

All models are compared on the same 100-row Med-HALT balanced sample.

## Results

| Model | Rows | Accuracy | Precision | Recall | F1 | False Positives | False Negatives | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen 1.5B | 100 | 0.86 | 0.790 | 0.98 | 0.875 | 13 | 1 | Best main model so far |
| Qwen2.5-3B | 100 | 0.66 | 0.598 | 0.98 | 0.743 | 33 | 1 | Useful as conservative secondary reviewer |
| Gemma 2 2B | 100 | 0.65 | 0.588 | 1.00 | 0.741 | 35 | 0 | Safest recall, but too many false positives for main model |

## Interpretation

Qwen 1.5B remains the best main model because it has the strongest balance between accuracy, precision, recall, and F1 score.

Gemma 2 2B achieved the safest hallucination-catching behavior with zero false negatives. However, it over-flagged many supported answers as hallucinated, so it should not replace Qwen 1.5B as the main model.

Qwen2.5-3B behaved similarly to Gemma. It was conservative, but did not improve enough to replace Qwen 1.5B.

## Current Project Decision

Use Qwen 1.5B as the main local model.

Keep Qwen2.5-3B and Gemma 2 2B as future secondary-review candidates, especially for disagreement-based human review routing.

Llama 3.2 3B Instruct is still waiting for Hugging Face access and should be tested next.
"""

REPORT_PATH.write_text(report, encoding="utf-8")

print("Three-model comparison complete")
print()
print(summary.to_string(index=False))
print()
print("Saved:", SUMMARY_PATH)
print("Saved:", REPORT_PATH)
