from pathlib import Path
import pandas as pd

INPUT_PATH = Path("results/qwen_medhalt/qwen_1_5b_medhalt_error_analysis.csv")

OUT_DIR = Path("results/trustworthiness")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_PATH = OUT_DIR / "qwen_1_5b_logic_aware_trustworthiness_scores.csv"
SUMMARY_PATH = OUT_DIR / "qwen_1_5b_logic_aware_routing_summary.csv"
REVIEW_PATH = OUT_DIR / "qwen_1_5b_logic_aware_review_cases.csv"
REPORT_PATH = REPORT_DIR / "qwen_1_5b_logic_aware_routing_report.md"

LOGIC_RISK_TERMS = [
    "not correct",
    "not true",
    "incorrect",
    "false",
    "except",
    "least likely",
    "not associated",
    "not characteristic",
    "not indicated",
    "contraindicated",
]


def has_logic_risk(question):
    question = str(question).lower()
    return any(term in question for term in LOGIC_RISK_TERMS)


def logic_aware_action(row):
    base_action = str(row.get("tuned_recommended_action", row.get("recommended_action", "")))

    question_logic_risk = bool(row["question_logic_risk"])
    hallucination_prob = float(row.get("hallucination_probability", 0))
    trust_score = float(row.get("clinical_trustworthiness_score", 0))
    grounding_score = float(row.get("semantic_grounding_score", 0))
    unsupported = float(row.get("unsupported_claim_count", 0))

    # Important: this rule does not use the true label.
    # It catches tricky medical exam-style prompts where a true statement can still be the wrong answer.
    if question_logic_risk and base_action == "accept":
        return "mandatory_human_review"

    if question_logic_risk and hallucination_prob < 0.20 and grounding_score >= 0.50:
        return "mandatory_human_review"

    if hallucination_prob >= 0.70:
        return "mandatory_human_review"

    if trust_score < 0.55:
        return "mandatory_human_review"

    if unsupported > 0 and grounding_score < 0.55:
        return "mandatory_human_review"

    if trust_score < 0.70:
        return "human_review"

    if trust_score < 0.82:
        return "revise"

    return base_action if base_action else "human_review"


def summarize(df, action_col):
    accepted = df[df[action_col] == "accept"]
    return {
        "action_column": action_col,
        "accepted_rows": len(accepted),
        "accepted_supported_rows": int((accepted["is_hallucinated"].astype(int) == 0).sum()),
        "accepted_hallucinated_rows": int((accepted["is_hallucinated"].astype(int) == 1).sum()),
        "mandatory_human_review_rows": int((df[action_col] == "mandatory_human_review").sum()),
        "human_review_rows": int((df[action_col] == "human_review").sum()),
        "revise_rows": int((df[action_col] == "revise").sum()),
        "logic_risk_rows": int(df["question_logic_risk"].sum()),
    }


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


def main():
    df = pd.read_csv(INPUT_PATH)

    if "tuned_recommended_action" not in df.columns:
        df["tuned_recommended_action"] = df["recommended_action"]

    df["question_logic_risk"] = df["question"].apply(has_logic_risk)
    df["logic_aware_recommended_action"] = df.apply(logic_aware_action, axis=1)

    summary = pd.DataFrame([
        summarize(df, "recommended_action"),
        summarize(df, "tuned_recommended_action"),
        summarize(df, "logic_aware_recommended_action"),
    ])

    action_by_label = (
        df.groupby(["is_hallucinated", "logic_aware_recommended_action"])
        .size()
        .reset_index(name="rows")
        .sort_values(["is_hallucinated", "logic_aware_recommended_action"])
    )

    review_cases = df[
        df["question_logic_risk"]
        | (
            (df["is_hallucinated"].astype(int) == 1)
            & (df["recommended_action"] == "accept")
        )
    ].copy()

    keep_cols = [
        "record_id",
        "dataset_name",
        "question",
        "answer",
        "is_hallucinated",
        "prediction",
        "hallucination_probability",
        "semantic_grounding_score",
        "unsupported_claim_count",
        "clinical_trustworthiness_score",
        "recommended_action",
        "tuned_recommended_action",
        "question_logic_risk",
        "logic_aware_recommended_action",
        "error_type",
    ]
    keep_cols = [col for col in keep_cols if col in review_cases.columns]

    df.to_csv(SCORES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    review_cases[keep_cols].to_csv(REVIEW_PATH, index=False)

    report = f"""# Qwen 1.5B Logic-Aware Routing Report

## Why This Was Needed

The first threshold tuning still accepted one hallucinated row.

That row was a tricky medical question because it asked which statement was NOT correct. The model selected an answer that looked medically supported, but it was wrong for the actual question logic.

This means the issue was not only hallucination probability or grounding score. It was a question-logic issue.

## Before vs After

{md_table(summary)}

## Logic-Aware Action Breakdown

{md_table(action_by_label)}

## Interpretation

The logic-aware routing rule catches exam-style medical prompts where words like NOT correct, incorrect, false, except, or least likely can change the meaning of the task.

This rule does not use the true label. It only uses the question text and model routing signals, so it is closer to what the final evaluation tool could do in practice.

## Project Decision

For healthcare evaluation, tricky negative questions should not be automatically accepted even when the answer seems grounded. These cases should go to human review because the answer may be factually true but still wrong for the question.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Logic-aware routing complete")
    print()
    print(summary.to_string(index=False))

    print()
    print("Logic-Aware Action Breakdown:")
    print(action_by_label.to_string(index=False))

    print()
    print("Saved:", SCORES_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", REVIEW_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
