from pathlib import Path
import pandas as pd

INPUT_PATH = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_completed.csv")
TRUST_PATH = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_scores.csv")

OUT_DIR = Path("results/equitymedqa_fairness")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_SUMMARY_PATH = OUT_DIR / "qwen_equitymedqa_manual_review_label_summary.csv"
CATEGORY_SUMMARY_PATH = OUT_DIR / "qwen_equitymedqa_manual_review_by_category.csv"
OUTCOME_SUMMARY_PATH = OUT_DIR / "qwen_equitymedqa_manual_review_outcome_summary.csv"
REPORT_PATH = REPORT_DIR / "equitymedqa_manual_review_summary.md"


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


def value_summary(df, col):
    if col not in df.columns:
        return pd.DataFrame()

    return (
        df[col]
        .fillna("missing")
        .value_counts()
        .rename_axis("label")
        .reset_index(name="rows")
        .assign(field=col)
        [["field", "label", "rows"]]
    )


def main():
    df = pd.read_csv(INPUT_PATH)

    trust = pd.read_csv(TRUST_PATH) if TRUST_PATH.exists() else pd.DataFrame()

    summary_fields = [
        "manual_clinical_safety",
        "manual_fairness",
        "manual_stereotype",
        "manual_paired_consistency",
        "manual_care_guidance",
        "manual_response_completeness",
        "manual_overall_risk",
        "manual_review_outcome",
        "clinical_validation_required",
    ]

    label_summary = pd.concat(
        [value_summary(df, col) for col in summary_fields],
        ignore_index=True
    )

    category_summary = (
        df.groupby(["fairness_category", "manual_overall_risk", "manual_review_outcome"])
        .size()
        .reset_index(name="rows")
        .sort_values(["fairness_category", "manual_overall_risk", "manual_review_outcome"])
    )

    outcome_summary = (
        df["manual_review_outcome"]
        .fillna("missing")
        .value_counts()
        .rename_axis("manual_review_outcome")
        .reset_index(name="rows")
    )

    label_summary.to_csv(LABEL_SUMMARY_PATH, index=False)
    category_summary.to_csv(CATEGORY_SUMMARY_PATH, index=False)
    outcome_summary.to_csv(OUTCOME_SUMMARY_PATH, index=False)

    trust_text = ""
    if not trust.empty and "recommended_action" in trust.columns:
        action_summary = (
            trust["recommended_action"]
            .value_counts()
            .rename_axis("recommended_action")
            .reset_index(name="rows")
        )

        trust_text = f"""
## Trustworthiness Routing

{md_table(action_summary)}

This shows how the reviewed EquityMedQA responses are routed by the trustworthiness layer. Responses with higher uncertainty or safety concern are sent to human review instead of being accepted directly.
"""

    report = f"""# EquityMedQA Manual Review Summary

## What Was Reviewed

I reviewed the 33 shortlisted EquityMedQA rows from the Qwen fairness evaluation run.

These rows were selected because the automated screening found possible fairness, safety, completeness, care guidance, stereotype, or paired-response consistency issues.

## Manual Review Outcome

{md_table(outcome_summary)}

This tells us how many shortlisted responses were accepted, marked for revision, or kept for further review after manual checking.

## Manual Review Label Summary

{md_table(label_summary)}

These labels break down the review from different angles: clinical safety, fairness, stereotype risk, paired consistency, care guidance, response completeness, and overall risk.

## Review By Fairness Category

{md_table(category_summary)}

This helps identify which fairness categories created the most review concerns.

{trust_text}

## Interpretation

The EquityMedQA layer adds a fairness and health-equity check to the project. It goes beyond hallucination detection by asking whether the model gives careful, complete, and consistent responses when prompts include demographic or access-related details.

The main takeaway is that many responses can sound reasonable but still need review because they may miss care guidance, provide incomplete support, or behave inconsistently across paired demographic prompts.

## Project Decision

EquityMedQA should remain part of the overall Responsible AI evaluation framework.

The current best use is as a manual-review support layer. It should not be treated as a final fairness judgment by itself, but it gives a structured way to find responses that need human checking before being trusted.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("EquityMedQA manual review summary complete")
    print()
    print("Outcome Summary:")
    print(outcome_summary.to_string(index=False))
    print()
    print("Label Summary:")
    print(label_summary.to_string(index=False))
    print()
    print("Review By Fairness Category:")
    print(category_summary.to_string(index=False))
    print()
    if trust_text:
        print("Trustworthiness Routing:")
        print(action_summary.to_string(index=False))
        print()
    print("Saved:", LABEL_SUMMARY_PATH)
    print("Saved:", CATEGORY_SUMMARY_PATH)
    print("Saved:", OUTCOME_SUMMARY_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
