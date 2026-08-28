from pathlib import Path
import pandas as pd

PRED_PATH = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
TRUST_PATH = Path("results/trustworthiness/qwen_1_5b_medhalt_trustworthiness_scores.csv")

OUT_DIR = Path("results/qwen_medhalt")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_PATH = OUT_DIR / "qwen_1_5b_medhalt_error_analysis.csv"
REVIEW_PATH = OUT_DIR / "qwen_1_5b_false_accept_review_shortlist.csv"
ACTION_BY_LABEL_PATH = OUT_DIR / "qwen_1_5b_action_by_true_label.csv"
REPORT_PATH = REPORT_DIR / "qwen_1_5b_medhalt_error_review.md"


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def pick_col(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {label}. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def md_table(df):
    if df.empty:
        return "_No rows._"

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def classify_error(row):
    true_label = int(row["is_hallucinated"])
    pred = int(row["prediction"])

    if true_label == 1 and pred == 1:
        return "true_positive_detected_hallucination"
    if true_label == 0 and pred == 0:
        return "true_negative_accepted_supported"
    if true_label == 1 and pred == 0:
        return "false_negative_missed_hallucination"
    if true_label == 0 and pred == 1:
        return "false_positive_flagged_supported"
    return "unknown"


def main():
    require_file(PRED_PATH)
    require_file(TRUST_PATH)

    pred = pd.read_csv(PRED_PATH)
    trust = pd.read_csv(TRUST_PATH)

    prediction_col = pick_col(
        pred,
        ["prediction", "predicted_label", "model_prediction", "qwen_prediction"],
        "prediction column"
    )

    probability_col = pick_col(
        pred,
        ["hallucination_probability", "probability", "prediction_probability", "score"],
        "hallucination probability column"
    )

    if "record_id" not in pred.columns:
        raise KeyError(f"Prediction file is missing record_id. Columns: {list(pred.columns)}")

    if "record_id" not in trust.columns:
        raise KeyError(f"Trustworthiness file is missing record_id. Columns: {list(trust.columns)}")

    pred = pred.copy()
    pred["prediction"] = pred[prediction_col].astype(int)
    pred["hallucination_probability"] = pred[probability_col].astype(float)

    trust_cols = ["record_id"]
    for col in [
        "semantic_grounding_score",
        "unsupported_claim_count",
        "clinical_trustworthiness_score",
        "recommended_action"
    ]:
        if col in trust.columns:
            trust_cols.append(col)

    df = pred.merge(
        trust[trust_cols],
        on="record_id",
        how="left"
    )

    for col, default in [
        ("semantic_grounding_score", 0.0),
        ("unsupported_claim_count", 0),
        ("clinical_trustworthiness_score", 0.0),
        ("recommended_action", "missing_trust_score")
    ]:
        if col not in df.columns:
            df[col] = default

    df["error_type"] = df.apply(classify_error, axis=1)

    review = df[
        (
            (df["is_hallucinated"].astype(int) == 1)
            & (
                (df["prediction"].astype(int) == 0)
                | (df["recommended_action"] == "accept")
            )
        )
        |
        (
            (df["recommended_action"] == "accept")
            & (df["unsupported_claim_count"].fillna(0) > 0)
        )
    ].copy()

    review = review.sort_values(
        by=[
            "recommended_action",
            "clinical_trustworthiness_score",
            "hallucination_probability"
        ],
        ascending=[True, False, True]
    )

    keep_cols = [
        "record_id",
        "dataset_name",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
        "prediction",
        "hallucination_probability",
        "semantic_grounding_score",
        "unsupported_claim_count",
        "clinical_trustworthiness_score",
        "recommended_action",
        "error_type"
    ]
    keep_cols = [col for col in keep_cols if col in review.columns]

    df.to_csv(ERROR_PATH, index=False)
    review[keep_cols].to_csv(REVIEW_PATH, index=False)

    error_summary = (
        df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="rows")
    )

    action_by_label = (
        df.groupby(["is_hallucinated", "recommended_action"])
        .size()
        .reset_index(name="rows")
        .sort_values(["is_hallucinated", "recommended_action"])
    )
    action_by_label.to_csv(ACTION_BY_LABEL_PATH, index=False)

    overall = {
        "rows": len(df),
        "accuracy": round((df["is_hallucinated"].astype(int) == df["prediction"].astype(int)).mean(), 3),
        "false_negatives_missed_hallucinations": int((df["error_type"] == "false_negative_missed_hallucination").sum()),
        "false_positives_flagged_supported": int((df["error_type"] == "false_positive_flagged_supported").sum()),
        "accepted_hallucinated_rows": int(((df["is_hallucinated"].astype(int) == 1) & (df["recommended_action"] == "accept")).sum()),
        "manual_review_shortlist_rows": len(review),
        "mean_trustworthiness_score": round(df["clinical_trustworthiness_score"].mean(), 3),
    }

    report = f"""# Qwen 1.5B Med-HALT Error Review

## What We Checked

This review looks at the Qwen 1.5B Med-HALT run after trustworthiness scoring.

The goal is not only to report accuracy. The goal is to find risky cases where a hallucinated medical answer may have been accepted or not escalated strongly enough.

## Overall Results

| Metric | Value |
|---|---:|
| Rows reviewed | {overall["rows"]} |
| Accuracy | {overall["accuracy"]} |
| False negatives / missed hallucinations | {overall["false_negatives_missed_hallucinations"]} |
| False positives / supported answers flagged | {overall["false_positives_flagged_supported"]} |
| Hallucinated rows accepted | {overall["accepted_hallucinated_rows"]} |
| Manual review shortlist rows | {overall["manual_review_shortlist_rows"]} |
| Mean trustworthiness score | {overall["mean_trustworthiness_score"]} |

## Error Type Summary

{md_table(error_summary)}

## Action By True Label

{md_table(action_by_label)}

## Interpretation

Qwen 1.5B is much stronger than Qwen 0.5B on the balanced Med-HALT sample, but the important healthcare question is whether hallucinated answers are still slipping through.

The manual shortlist focuses on exactly those cases. These are the rows we should inspect before calling Qwen 1.5B reliable.

## Next Step

Open the false-accept review shortlist and manually inspect the highest-risk rows. If accepted hallucinated rows are truly unsafe, we should tighten the trustworthiness threshold before moving to the next dataset.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Qwen 1.5B Error Review Complete")
    print()
    print("Detected prediction column:", prediction_col)
    print("Detected probability column:", probability_col)
    print()
    print("Overall:")
    for key, value in overall.items():
        print(f"{key}: {value}")

    print()
    print("Error Summary:")
    print(error_summary.to_string(index=False))

    print()
    print("Action By True Label:")
    print(action_by_label.to_string(index=False))

    print()
    print("Saved:", ERROR_PATH)
    print("Saved:", REVIEW_PATH)
    print("Saved:", ACTION_BY_LABEL_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
