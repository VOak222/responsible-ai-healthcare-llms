from pathlib import Path
import pandas as pd

TFIDF_PATH = Path(
    "results/trustworthiness/"
    "clinical_trustworthiness_overall.csv"
)

QWEN_PATH = Path(
    "results/trustworthiness/"
    "qwen_equitymedqa_trustworthiness_overall.csv"
)

OUTPUT_DIR = Path("results/comparison")
OUTPUT_PATH = OUTPUT_DIR / "model_evaluation_layer_summary.csv"
REPORT_PATH = Path("reports/model_evaluation_layer_summary.md")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tfidf = pd.read_csv(TFIDF_PATH).iloc[0]
    qwen = pd.read_csv(QWEN_PATH).iloc[0]

    summary = pd.DataFrame([
        {
            "model_id": "TF-IDF Logistic Regression",
            "evaluation_layer": "hallucination_and_grounding",
            "evaluation_dataset": "baseline_evaluation_dataset",
            "rows_evaluated": int(tfidf["rows_scored"]),
            "score_eligible_rows": int(tfidf["rows_scored"]),
            "primary_score_name": "clinical_trustworthiness_score",
            "primary_score": tfidf[
                "mean_clinical_trustworthiness_score"
            ],
            "mean_hallucination_risk": tfidf[
                "mean_hallucination_risk"
            ],
            "mean_grounding_score": tfidf[
                "mean_final_grounding_score"
            ],
            "mean_clinical_safety_trust": None,
            "mean_fairness_trust": None,
            "mandatory_human_review_rows": int(
                tfidf["mandatory_human_review_rows"]
            ),
        },
        {
            "model_id": qwen["model_id"],
            "evaluation_layer": "fairness_and_safety",
            "evaluation_dataset": qwen["evaluation_dataset"],
            "rows_evaluated": int(qwen["rows_reviewed"]),
            "score_eligible_rows": int(
                qwen["composite_score_eligible_rows"]
            ),
            "primary_score_name":
                "equitymedqa_trustworthiness_score",
            "primary_score": qwen[
                "mean_equitymedqa_trustworthiness_score"
            ],
            "mean_hallucination_risk": None,
            "mean_grounding_score": None,
            "mean_clinical_safety_trust": qwen[
                "mean_clinical_safety_trust_score"
            ],
            "mean_fairness_trust": qwen[
                "mean_fairness_trust_score"
            ],
            "mandatory_human_review_rows": int(
                qwen["mandatory_human_review_rows"]
            ),
        },
    ])

    summary.to_csv(OUTPUT_PATH, index=False)

    report = f"""# Model Evaluation Layer Summary

## Results

{summary.to_markdown(index=False)}

## Interpretation

The TF-IDF baseline currently has hallucination and grounding results.

Qwen currently has fairness and clinical-safety results from EquityMedQA.

These scores are presented in separate evaluation layers and are not directly averaged because they come from different models and datasets.

The next step toward a unified model score is to run Qwen on the hallucination and grounding evaluation data.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Model Evaluation Layer Summary:")
    print(summary.to_string(index=False))
    print()
    print("Saved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
