from pathlib import Path
import pandas as pd

DISAGREEMENT_PATH = Path("results/comparison/qwen_1_5b_vs_qwen_2_5_3b_disagreements.csv")
COMPARISON_PATH = Path("results/comparison/qwen_1_5b_vs_qwen_2_5_3b_medhalt_comparison.csv")

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports/future_work")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_PATH = OUT_DIR / "qwen_model_disagreement_review_layer.csv"
SUMMARY_PATH = OUT_DIR / "qwen_model_disagreement_review_summary.csv"
REPORT_PATH = REPORT_DIR / "qwen_model_disagreement_review_layer_report.md"


def classify_disagreement(row):
    true_label = int(row["is_hallucinated"])
    q15 = int(row["qwen_1_5b_prediction"])
    q3 = int(row["qwen_2_5_3b_prediction"])

    if q15 == 0 and q3 == 1:
        if true_label == 0:
            return "qwen_3b_over_flagged_supported_answer"
        return "qwen_1_5b_missed_hallucination"

    if q15 == 1 and q3 == 0:
        if true_label == 0:
            return "qwen_1_5b_over_flagged_supported_answer"
        return "qwen_3b_missed_hallucination"

    return "no_disagreement"


def recommended_action(row):
    reason = row["disagreement_type"]

    if "missed_hallucination" in reason:
        return "mandatory_human_review"

    if "over_flagged_supported_answer" in reason:
        return "human_review"

    return "accept"


def md_table(df):
    if df.empty:
        return "_No rows available._"

    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    if not DISAGREEMENT_PATH.exists():
        raise FileNotFoundError(f"Missing disagreement file: {DISAGREEMENT_PATH}")

    df = pd.read_csv(DISAGREEMENT_PATH).copy()

    df["disagreement_type"] = df.apply(classify_disagreement, axis=1)
    df["disagreement_recommended_action"] = df.apply(recommended_action, axis=1)

    summary = (
        df.groupby(["disagreement_type", "disagreement_recommended_action"])
        .size()
        .reset_index(name="rows")
        .sort_values(["disagreement_recommended_action", "disagreement_type"])
    )

    label_summary = (
        df.groupby(["is_hallucinated", "disagreement_type"])
        .size()
        .reset_index(name="rows")
        .sort_values(["is_hallucinated", "disagreement_type"])
    )

    high_priority = df[df["disagreement_recommended_action"] == "mandatory_human_review"].copy()

    df.to_csv(REVIEW_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    report = f"""# Qwen Model Disagreement Review Layer

## Purpose

This analysis uses disagreement between Qwen 1.5B and Qwen2.5-3B as a safety signal.

The idea is simple: when two local healthcare evaluation models disagree, the row should be treated as uncertain instead of being accepted without review.

## Disagreement Summary

{md_table(summary)}

## Disagreement By True Label

{md_table(label_summary)}

## High-Priority Cases

Mandatory human review cases: {len(high_priority)}

These are cases where one model missed a hallucinated response while the other model caught it.

## Interpretation

Qwen 1.5B remains the stronger main model because it has better balance between recall and false positives.

Qwen2.5-3B is still useful because it disagrees with Qwen 1.5B on uncertain cases. These disagreement rows can help identify responses that need extra review.

## Project Decision

Model disagreement should be used as an additional review signal in the future governance layer.

A practical rule is:

If Qwen 1.5B and Qwen2.5-3B disagree, route the case to human review. If the disagreement involves a missed hallucination, route it to mandatory human review.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Qwen model disagreement review layer complete")
    print()
    print("Disagreement Summary:")
    print(summary.to_string(index=False))

    print()
    print("Disagreement By True Label:")
    print(label_summary.to_string(index=False))

    print()
    print("Mandatory human review cases:", len(high_priority))
    print()
    print("Saved:", REVIEW_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
