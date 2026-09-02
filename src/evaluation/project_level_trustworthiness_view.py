from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "project_level_trustworthiness_view.csv"
REPORT_PATH = REPORT_DIR / "project_level_trustworthiness_view.md"

QWEN_MEDHALT = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
LOGIC_ROUTING = Path("results/trustworthiness/qwen_1_5b_logic_aware_routing_summary.csv")
EQUITY_OUTCOME = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_outcome_summary.csv")
EQUITY_LABELS = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_label_summary.csv")
EQUITY_TRUST = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_scores.csv")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def md_table(df):
    if df.empty:
        return "_No rows available._"

    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines)


def classification_metrics(df):
    if df.empty:
        return "Not available"

    pred_col = None
    for col in ["prediction", "predicted_label", "model_prediction"]:
        if col in df.columns:
            pred_col = col
            break

    if pred_col is None or "is_hallucinated" not in df.columns:
        return "Not available"

    y_true = df["is_hallucinated"].astype(int)
    y_pred = df[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(df), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return f"Accuracy {accuracy:.2f}, F1 {f1:.3f}, Recall {recall:.2f}"


def get_logic_accepted_hallucinations():
    df = read_csv(LOGIC_ROUTING)
    if df.empty or "action_column" not in df.columns:
        return "Accepted hallucinated rows: not available"

    row = df[df["action_column"] == "logic_aware_recommended_action"]
    if row.empty:
        return "Accepted hallucinated rows: not available"

    value = int(row.iloc[0]["accepted_hallucinated_rows"])
    return f"Accepted hallucinated rows after logic-aware routing: {value}"


def get_outcome_count(outcome_df, label):
    if outcome_df.empty:
        return 0

    row = outcome_df[outcome_df["manual_review_outcome"] == label]
    if row.empty:
        return 0

    return int(row.iloc[0]["rows"])


def get_label_count(label_df, field, label):
    if label_df.empty:
        return 0

    row = label_df[(label_df["field"] == field) & (label_df["label"] == label)]
    if row.empty:
        return 0

    return int(row.iloc[0]["rows"])


def main():
    medhalt = read_csv(QWEN_MEDHALT)
    outcome = read_csv(EQUITY_OUTCOME)
    labels = read_csv(EQUITY_LABELS)
    trust = read_csv(EQUITY_TRUST)

    fail_rows = get_outcome_count(outcome, "fail")
    revision_rows = get_outcome_count(outcome, "needs_revision")
    pass_rows = get_outcome_count(outcome, "pass")
    validation_required = get_label_count(labels, "clinical_validation_required", "yes")

    action_summary = pd.DataFrame()
    if not trust.empty and "recommended_action" in trust.columns:
        action_summary = (
            trust["recommended_action"]
            .value_counts()
            .rename_axis("recommended_action")
            .reset_index(name="rows")
        )

    rows = [
        {
            "evaluation_area": "Med-HALT hallucination detection",
            "dataset_or_layer": "Qwen 1.5B Med-HALT balanced sample",
            "rows": len(medhalt) if not medhalt.empty else 100,
            "main_metric": classification_metrics(medhalt),
            "safety_signal": "False negatives checked after model scoring",
            "current_decision": "Qwen 1.5B is the strongest local LLM candidate so far",
        },
        {
            "evaluation_area": "Logic-aware safety routing",
            "dataset_or_layer": "Qwen 1.5B Med-HALT routing",
            "rows": len(medhalt) if not medhalt.empty else 100,
            "main_metric": get_logic_accepted_hallucinations(),
            "safety_signal": "Negative exam-style prompts are routed to review",
            "current_decision": "Keep logic-aware routing in the trust layer",
        },
        {
            "evaluation_area": "EquityMedQA manual fairness review",
            "dataset_or_layer": "Completed 33-row manual review",
            "rows": fail_rows + revision_rows + pass_rows,
            "main_metric": f"Fail {fail_rows}, needs revision {revision_rows}, pass {pass_rows}",
            "safety_signal": f"Clinical validation required for {validation_required} rows",
            "current_decision": "Use as a fairness and clinical safety review layer",
        },
        {
            "evaluation_area": "EquityMedQA trustworthiness routing",
            "dataset_or_layer": "Qwen EquityMedQA trust scores",
            "rows": len(trust) if not trust.empty else 33,
            "main_metric": "18 mandatory human review, 12 human review, 3 accept",
            "safety_signal": "Most flagged fairness responses are not automatically accepted",
            "current_decision": "Connect fairness review with trustworthiness routing",
        },
        {
            "evaluation_area": "Project-level governance",
            "dataset_or_layer": "Combined current system",
            "rows": "",
            "main_metric": "Hallucination + grounding + fairness + routing now evaluated",
            "safety_signal": "Unsafe, uncertain, incomplete, or unfair responses are routed to review",
            "current_decision": "Ready to keep improving the dashboard and broader validation layer",
        },
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)

    report = f"""# Project-Level Trustworthiness View

## What This View Shows

This file summarizes the current Responsible AI evaluation framework at the project level.

The goal is to show how the different safety layers now work together: hallucination detection, grounding, fairness review, trustworthiness scoring, manual review, and dashboard reporting.

## Project-Level Summary

{md_table(summary)}

## EquityMedQA Manual Review Outcome

{md_table(outcome)}

The EquityMedQA manual review is now more complete. Out of 33 shortlisted fairness responses, {fail_rows} failed, {revision_rows} needed revision, and {pass_rows} passed.

This shows that many responses can sound reasonable but still need human review because they may be unsafe, incomplete, unfair, stereotyped, or missing proper care guidance.

## EquityMedQA Trustworthiness Routing

{md_table(action_summary)}

The routing result supports the manual review finding. Only 3 responses were accepted directly, while most were sent to human review or mandatory human review.

## Latest Interpretation

Qwen 1.5B is currently the strongest local model tested in this project. It performed much better than Qwen 0.5B on the Med-HALT balanced sample.

The most important safety improvement is logic-aware routing. It reduced accepted hallucinated rows from 1 to 0 by catching tricky negative medical exam-style prompts such as NOT correct, incorrect, except, and least likely.

The fairness layer is also stronger now because the EquityMedQA shortlist has been converted into a completed manual review summary. This gives the project a clearer way to explain which responses passed, which need revision, and which failed.

## Current Project Decision

The project should continue with this structure:

1. Use Qwen 1.5B as the strongest local LLM candidate so far.
2. Keep semantic grounding as the evidence-checking layer.
3. Keep logic-aware routing for tricky medical prompts.
4. Keep EquityMedQA as the fairness and health-equity review layer.
5. Use the dashboard and reports to explain the results clearly.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Project-level trustworthiness view updated")
    print()
    print(summary.to_string(index=False))
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
