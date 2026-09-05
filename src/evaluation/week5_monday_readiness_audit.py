from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "week5_monday_readiness_audit.csv"
REPORT_PATH = REPORT_DIR / "week5_monday_readiness_audit.md"

MEDHALT_PRED = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
LOGIC_ROUTING = Path("results/trustworthiness/qwen_1_5b_logic_aware_routing_summary.csv")
MANUAL_REVIEW = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_completed.csv")
TRUST_SCORES = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_scores.csv")
EXPLAINABLE_RISK = Path("results/trustworthiness/qwen_equitymedqa_explainable_fairness_risk.csv")
RISK_REASONS = Path("results/trustworthiness/qwen_equitymedqa_explainable_risk_reason_summary.csv")
PROJECT_VIEW = Path("results/comparison/project_level_trustworthiness_view.csv")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None


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


def medhalt_metrics(df):
    if df.empty:
        return "Not available"

    pred_col = pick_col(df, ["prediction", "predicted_label", "model_prediction"])
    label_col = pick_col(df, ["is_hallucinated", "true_label", "label"])

    if pred_col is None or label_col is None:
        return "Prediction columns not found"

    y_true = df[label_col].astype(int)
    y_pred = df[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = round((tp + tn) / max(len(df), 1), 3)
    precision = round(tp / max(tp + fp, 1), 3)
    recall = round(tp / max(tp + fn, 1), 3)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-9), 3)

    return f"Accuracy {accuracy}, F1 {f1}, Recall {recall}, FN {fn}"


def main():
    medhalt = read_csv(MEDHALT_PRED)
    logic = read_csv(LOGIC_ROUTING)
    manual = read_csv(MANUAL_REVIEW)
    trust = read_csv(TRUST_SCORES)
    risk = read_csv(EXPLAINABLE_RISK)
    reasons = read_csv(RISK_REASONS)
    project = read_csv(PROJECT_VIEW)

    logic_accepted_hallucinations = "Not available"
    if not logic.empty and "action_column" in logic.columns:
        row = logic[logic["action_column"] == "logic_aware_recommended_action"]
        if not row.empty and "accepted_hallucinated_rows" in row.columns:
            logic_accepted_hallucinations = int(row.iloc[0]["accepted_hallucinated_rows"])

    manual_outcomes = {"fail": 0, "needs_revision": 0, "pass": 0}
    if not manual.empty and "manual_review_outcome" in manual.columns:
        manual_outcomes.update(manual["manual_review_outcome"].value_counts().to_dict())

    routing_actions = {"mandatory_human_review": 0, "human_review": 0, "accept": 0}
    if not trust.empty and "recommended_action" in trust.columns:
        routing_actions.update(trust["recommended_action"].value_counts().to_dict())

    unsafe_accepts = 0
    under_escalated_fails = 0
    accepted_passes = 0
    if not trust.empty and {"manual_review_outcome", "recommended_action"}.issubset(trust.columns):
        unsafe_accepts = int(((trust["manual_review_outcome"] == "fail") & (trust["recommended_action"] == "accept")).sum())
        under_escalated_fails = int(((trust["manual_review_outcome"] == "fail") & (trust["recommended_action"] != "mandatory_human_review")).sum())
        accepted_passes = int(((trust["manual_review_outcome"] == "pass") & (trust["recommended_action"] == "accept")).sum())

    risk_bands = {"high": 0, "medium": 0, "low": 0}
    if not risk.empty and "explainable_fairness_risk_band" in risk.columns:
        risk_bands.update(risk["explainable_fairness_risk_band"].value_counts().to_dict())

    top_reasons = []
    if not reasons.empty and "risk_reason" in reasons.columns:
        top_reasons = reasons.head(5)["risk_reason"].tolist()

    audit_rows = [
        {
            "workstream": "Med-HALT Qwen 1.5B evaluation",
            "status": "complete",
            "result": medhalt_metrics(medhalt),
            "monday_use": "Use in PPT/report as strongest local LLM result so far",
        },
        {
            "workstream": "Logic-aware safety routing",
            "status": "complete",
            "result": f"Accepted hallucinated rows after routing: {logic_accepted_hallucinations}",
            "monday_use": "Use as the main safety improvement example",
        },
        {
            "workstream": "EquityMedQA manual review",
            "status": "complete",
            "result": f"Fail {manual_outcomes.get('fail', 0)}, needs revision {manual_outcomes.get('needs_revision', 0)}, pass {manual_outcomes.get('pass', 0)}",
            "monday_use": "Use to show human review was completed for flagged fairness cases",
        },
        {
            "workstream": "EquityMedQA trust routing",
            "status": "complete",
            "result": f"{routing_actions.get('mandatory_human_review', 0)} mandatory review, {routing_actions.get('human_review', 0)} human review, {routing_actions.get('accept', 0)} accept",
            "monday_use": "Use to show risky responses were not automatically accepted",
        },
        {
            "workstream": "Manual-trust alignment",
            "status": "complete",
            "result": f"Unsafe accepts {unsafe_accepts}, under-escalated fails {under_escalated_fails}, accepted pass cases {accepted_passes}",
            "monday_use": "Use to prove routing matches manual review outcomes",
        },
        {
            "workstream": "Explainable fairness-risk scoring",
            "status": "complete",
            "result": f"High {risk_bands.get('high', 0)}, medium {risk_bands.get('medium', 0)}, low {risk_bands.get('low', 0)}",
            "monday_use": "Use to explain why cases were routed for review",
        },
        {
            "workstream": "Dashboard reporting",
            "status": "ready for meeting review",
            "result": "Final dashboard updated with Week 4 metrics",
            "monday_use": "Open and decide whether to show it during Tuesday meeting",
        },
    ]

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_CSV, index=False)

    top_reason_text = ", ".join(top_reasons) if top_reasons else "Not available"

    report = f"""# Week 5 Monday Readiness Audit

## Purpose

This note prepares the project for Monday's report and PPT update work.

The main goal is to make sure the latest technical results are already organized before presentation preparation starts.

## Readiness Summary

{md_table(audit)}

## Main Project Story For Next Week

The project has moved from simple model evaluation into a broader trustworthiness workflow.

The current evaluation now covers hallucination detection, logic-aware routing, manual fairness review, trustworthiness routing, manual-review alignment, and explainable risk scoring.

## Strongest Results To Present

| Area | Result |
|---|---|
| Qwen 1.5B Med-HALT performance | {medhalt_metrics(medhalt)} |
| Logic-aware routing | Accepted hallucinated rows after routing: {logic_accepted_hallucinations} |
| EquityMedQA manual review | Fail {manual_outcomes.get('fail', 0)}, needs revision {manual_outcomes.get('needs_revision', 0)}, pass {manual_outcomes.get('pass', 0)} |
| Manual-trust alignment | Unsafe accepts {unsafe_accepts}, under-escalated fails {under_escalated_fails} |
| Explainable fairness risk | High {risk_bands.get('high', 0)}, medium {risk_bands.get('medium', 0)}, low {risk_bands.get('low', 0)} |

## Explainable Risk Reasons

The top risk reasons are:

{top_reason_text}

These reasons help explain why certain healthcare responses should go to review instead of being accepted automatically.

## Monday Work Plan

1. Update the PPT with the latest Week 4 and Week 5 readiness results.
2. Update the written report with the manual-trust alignment and explainable risk findings.
3. Decide whether to show the dashboard in the Tuesday meeting.
4. Keep the dashboard as supporting material if the meeting is short.
5. Start the next technical phase after the meeting: broader validation across another dataset or a larger EquityMedQA run.

## Current Position

The project is ready for Monday presentation preparation. The remaining work is mainly communication, formatting, and deciding which results to highlight for Tuesday.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Week 5 Monday readiness audit complete")
    print()
    print(audit.to_string(index=False))
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
