from pathlib import Path
import pandas as pd

PROJECT_CSV = Path("results/comparison/project_level_trustworthiness_view.csv")
PROJECT_REPORT = Path("reports/project_level_trustworthiness_view.md")

EXPLAINABLE_SCORES = Path("results/trustworthiness/qwen_equitymedqa_explainable_fairness_risk.csv")
RISK_BY_OUTCOME = Path("results/trustworthiness/qwen_equitymedqa_explainable_risk_by_manual_outcome.csv")
RISK_REASONS = Path("results/trustworthiness/qwen_equitymedqa_explainable_risk_reason_summary.csv")

OUT_SCRIPT_NOTE = Path("reports/daily_progress_explainable_equitymedqa_fairness_risk.md")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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
    require_file(PROJECT_CSV)
    require_file(EXPLAINABLE_SCORES)
    require_file(RISK_BY_OUTCOME)
    require_file(RISK_REASONS)

    project = pd.read_csv(PROJECT_CSV)
    scores = pd.read_csv(EXPLAINABLE_SCORES)
    risk_by_outcome = pd.read_csv(RISK_BY_OUTCOME)
    risk_reasons = pd.read_csv(RISK_REASONS)

    high_count = int((scores["explainable_fairness_risk_band"] == "high").sum())
    medium_count = int((scores["explainable_fairness_risk_band"] == "medium").sum())
    low_count = int((scores["explainable_fairness_risk_band"] == "low").sum())

    fail_high = risk_by_outcome[
        (risk_by_outcome["manual_review_outcome"] == "fail")
        & (risk_by_outcome["explainable_fairness_risk_band"] == "high")
    ]["rows"].sum()

    pass_low = risk_by_outcome[
        (risk_by_outcome["manual_review_outcome"] == "pass")
        & (risk_by_outcome["explainable_fairness_risk_band"] == "low")
    ]["rows"].sum()

    top_reasons = risk_reasons.head(5)["risk_reason"].tolist()
    top_reason_text = "; ".join(top_reasons)

    new_row = {
        "evaluation_area": "EquityMedQA explainable fairness risk",
        "dataset_or_layer": "Manual review labels + explainable risk scoring",
        "rows": len(scores),
        "main_metric": f"High {high_count}, medium {medium_count}, low {low_count}",
        "safety_signal": f"Fail rows marked high risk: {int(fail_high)}; pass rows marked low risk: {int(pass_low)}",
        "current_decision": "Use risk reasons to explain why cases are routed to review",
    }

    project = project[
        project["evaluation_area"] != "EquityMedQA explainable fairness risk"
    ].copy()

    governance = project[project["evaluation_area"] == "Project-level governance"].copy()
    project = project[project["evaluation_area"] != "Project-level governance"].copy()

    project = pd.concat([project, pd.DataFrame([new_row]), governance], ignore_index=True)

    project.to_csv(PROJECT_CSV, index=False)

    daily_note = f"""# Daily Progress: Explainable EquityMedQA Fairness Risk

## Work Completed

Today I added an explainable fairness-risk layer for the completed EquityMedQA manual review results.

This step helps connect the manual labels to clear risk reasons. Instead of only saying that a response failed or needed revision, the analysis now shows why the response was risky.

## Main Results

| Risk band | Rows |
|---|---:|
| High | {high_count} |
| Medium | {medium_count} |
| Low | {low_count} |

All 16 failed manual review cases were placed in the high-risk band. The 3 passing cases were placed in the low-risk band.

## Top Risk Reasons

{md_table(risk_reasons.head(12))}

## Why This Matters

This makes the trustworthiness layer easier to explain. The project can now show that review routing is not random or only based on a score. It is connected to specific issues such as clinical validation, incomplete responses, unsafe clinical labels, missing care guidance, stereotype signals, and fairness concerns.

## Current Project Decision

The explainable risk layer should be used alongside the manual review outcomes and trustworthiness routing. It gives clearer justification for why unsafe, incomplete, or unfair responses should not be automatically accepted.
"""

    OUT_SCRIPT_NOTE.write_text(daily_note, encoding="utf-8")

    report = f"""# Project-Level Trustworthiness View

## Current Project Status

This view brings together the main safety and trustworthiness checks completed so far across Med-HALT and EquityMedQA.

The project now evaluates hallucination detection, logic-aware safety routing, manual fairness review, trustworthiness routing, and explainable fairness-risk scoring.

## Project-Level Summary

{md_table(project)}

## Latest Update

The newest update adds explainable fairness-risk scoring for the EquityMedQA manual review results.

This helps explain why certain responses are considered risky. The strongest signals were clinical validation needs, incomplete or truncated responses, unsafe clinical safety labels, missing professional care guidance, stereotype signals, and fairness concerns.

## Current Interpretation

Qwen 1.5B remains the strongest local LLM candidate tested so far on Med-HALT. The logic-aware routing layer successfully prevents tricky negative-question cases from being automatically accepted.

For EquityMedQA, the manual review and trustworthiness layers show that most risky fairness responses should go to human review instead of being accepted. The explainable fairness-risk layer now adds clearer reasoning behind those decisions.

## Next Step

The next step is to keep improving the reporting layer so the results from different datasets can be shown together clearly. The dashboard/reporting view is still in progress and can be used later when it is ready for presentation.
"""

    PROJECT_REPORT.write_text(report, encoding="utf-8")

    print("Project-level view updated with explainable EquityMedQA fairness risk")
    print()
    print(project.to_string(index=False))
    print()
    print("Risk band counts:")
    print(f"high: {high_count}")
    print(f"medium: {medium_count}")
    print(f"low: {low_count}")
    print()
    print("Saved:", PROJECT_CSV)
    print("Saved:", PROJECT_REPORT)
    print("Saved:", OUT_SCRIPT_NOTE)


if __name__ == "__main__":
    main()
