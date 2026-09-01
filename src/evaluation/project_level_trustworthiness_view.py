from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "project_level_trustworthiness_view.csv"
REPORT_PATH = REPORT_DIR / "project_level_trustworthiness_view.md"

MEDHALT_PRED = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
MEDHALT_LOGIC = Path("results/trustworthiness/qwen_1_5b_logic_aware_routing_summary.csv")

EQUITY_FULL_RESPONSES = Path("results/equitymedqa_fairness/qwen_equitymedqa_fairness_full_responses.csv")
EQUITY_FULL_REVIEW = Path("results/equitymedqa_fairness/qwen_equitymedqa_fairness_full_review.csv")
EQUITY_FULL_REVIEW_SUMMARY = Path("results/equitymedqa_fairness/qwen_equitymedqa_fairness_full_review_summary.csv")
EQUITY_TRUST_OVERALL = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_overall.csv")
EQUITY_TRUST_ACTIONS = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_action_summary.csv")
EQUITY_SHORTLIST = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_shortlist.csv")


def read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def classification_metrics(df):
    pred_col = pick_col(df, ["prediction", "predicted_label", "model_prediction"])
    label_col = pick_col(df, ["is_hallucinated", "true_label", "label"])

    if df.empty or pred_col is None or label_col is None:
        return {}

    y_true = df[label_col].astype(int)
    y_pred = df[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "rows": len(df),
        "accuracy": round((tp + tn) / max(len(df), 1), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "false_negatives": fn,
        "false_positives": fp,
    }


def summarize_action_file(df):
    if df.empty:
        return "No action summary found"

    action_col = pick_col(df, ["recommended_action", "action", "trustworthiness_action"])
    count_col = pick_col(df, ["rows", "count", "n"])

    if action_col and count_col:
        parts = []
        for _, row in df.iterrows():
            parts.append(f"{row[action_col]}: {row[count_col]}")
        return "; ".join(parts)

    return f"{len(df)} summary rows available"


def get_first_numeric(df, preferred_terms):
    if df.empty:
        return ""

    for term in preferred_terms:
        for col in df.columns:
            if term.lower() in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return round(float(df[col].iloc[0]), 3)

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        return round(float(df[numeric_cols[0]].iloc[0]), 3)

    return ""


def md_table(df):
    if df.empty:
        return "_No rows._"

    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    medhalt_pred = read_csv(MEDHALT_PRED)
    medhalt_logic = read_csv(MEDHALT_LOGIC)

    equity_responses = read_csv(EQUITY_FULL_RESPONSES)
    equity_review = read_csv(EQUITY_FULL_REVIEW)
    equity_review_summary = read_csv(EQUITY_FULL_REVIEW_SUMMARY)
    equity_trust_overall = read_csv(EQUITY_TRUST_OVERALL)
    equity_trust_actions = read_csv(EQUITY_TRUST_ACTIONS)
    equity_shortlist = read_csv(EQUITY_SHORTLIST)

    medhalt_metrics = classification_metrics(medhalt_pred)

    accepted_hallucinated = ""
    if not medhalt_logic.empty and "action_column" in medhalt_logic.columns:
        row = medhalt_logic[medhalt_logic["action_column"] == "logic_aware_recommended_action"]
        if not row.empty and "accepted_hallucinated_rows" in row.columns:
            accepted_hallucinated = int(row["accepted_hallucinated_rows"].iloc[0])

    rows = [
        {
            "evaluation_area": "Med-HALT hallucination detection",
            "dataset_or_layer": "Med-HALT balanced sample",
            "rows": medhalt_metrics.get("rows", ""),
            "main_metric": f"Accuracy {medhalt_metrics.get('accuracy', '')}, F1 {medhalt_metrics.get('f1_score', '')}, Recall {medhalt_metrics.get('recall', '')}",
            "safety_signal": f"False negatives: {medhalt_metrics.get('false_negatives', '')}",
            "current_decision": "Qwen 1.5B is the stronger local LLM candidate"
        },
        {
            "evaluation_area": "Logic-aware safety routing",
            "dataset_or_layer": "Qwen 1.5B Med-HALT routing",
            "rows": medhalt_metrics.get("rows", ""),
            "main_metric": f"Accepted hallucinated rows: {accepted_hallucinated}",
            "safety_signal": "Negative exam-style prompts are sent to review",
            "current_decision": "Keep logic-aware routing in the trust layer"
        },
        {
            "evaluation_area": "EquityMedQA fairness responses",
            "dataset_or_layer": "Full EquityMedQA Qwen run",
            "rows": len(equity_responses),
            "main_metric": f"Review rows: {len(equity_review)}",
            "safety_signal": f"Manual review shortlist rows: {len(equity_shortlist)}",
            "current_decision": "Use as fairness and bias review layer"
        },
        {
            "evaluation_area": "EquityMedQA trustworthiness routing",
            "dataset_or_layer": "Qwen EquityMedQA trust scores",
            "rows": get_first_numeric(equity_trust_overall, ["rows", "scored"]),
            "main_metric": f"Mean trustworthiness score: {get_first_numeric(equity_trust_overall, ['mean', 'trustworthiness'])}",
            "safety_signal": summarize_action_file(equity_trust_actions),
            "current_decision": "Connect fairness review with trustworthiness routing"
        },
        {
            "evaluation_area": "Project-level governance",
            "dataset_or_layer": "Combined current system",
            "rows": "",
            "main_metric": "Hallucination + grounding + fairness + routing now evaluated",
            "safety_signal": "Unsafe or uncertain cases are routed to human review",
            "current_decision": "Ready to prepare dashboard/reporting layer next"
        }
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)

    review_summary_text = "No EquityMedQA review summary file found."
    if not equity_review_summary.empty:
        review_summary_text = md_table(equity_review_summary.head(10))

    report = f"""# Project-Level Trustworthiness View

## What This Shows

This report combines the main safety evaluation layers built so far.

The project now evaluates healthcare LLM outputs across hallucination detection, evidence grounding, fairness review, and human-review routing.

## Project-Level Summary

{md_table(summary)}

## EquityMedQA Review Summary

{review_summary_text}

## Interpretation

The strongest current model result is Qwen 1.5B on the Med-HALT balanced sample. It performed much better than Qwen 0.5B, but the important safety improvement was the logic-aware routing layer.

The logic-aware rule reduced accepted hallucinated rows to 0 by catching negative medical question patterns such as NOT correct, incorrect, except, and least likely.

EquityMedQA adds the fairness and bias-assessment side of the project. This means the framework is no longer only checking whether an answer is hallucinated. It is also checking whether model behavior changes across sensitive or demographic prompt variations.

## Current Project Position

At this point, the project has a usable responsible-AI evaluation structure:

1. Hallucination detection using baseline and local LLM evaluation.
2. Evidence grounding using semantic similarity and unsupported claim checks.
3. Clinical trustworthiness scoring.
4. Logic-aware safety routing for tricky medical prompts.
5. EquityMedQA fairness and bias review.

## Recommended Next Step

The next step is to build a simple dashboard/reporting view so these results can be shown clearly in a demo.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Project-level trustworthiness view complete")
    print()
    print(summary.to_string(index=False))
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
