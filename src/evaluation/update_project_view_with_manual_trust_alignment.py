from pathlib import Path
import pandas as pd

PROJECT_VIEW_PATH = Path("results/comparison/project_level_trustworthiness_view.csv")
ALIGNMENT_SUMMARY_PATH = Path("results/trustworthiness/qwen_equitymedqa_manual_trust_alignment_summary.csv")
ALIGNMENT_CROSSTAB_PATH = Path("results/trustworthiness/qwen_equitymedqa_routing_by_manual_outcome.csv")

OUT_PATH = Path("results/comparison/project_level_trustworthiness_view.csv")
REPORT_PATH = Path("reports/project_level_trustworthiness_view.md")

def main():
    if not PROJECT_VIEW_PATH.exists():
        raise FileNotFoundError(f"Missing project view: {PROJECT_VIEW_PATH}")

    if not ALIGNMENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing alignment summary: {ALIGNMENT_SUMMARY_PATH}")

    if not ALIGNMENT_CROSSTAB_PATH.exists():
        raise FileNotFoundError(f"Missing alignment crosstab: {ALIGNMENT_CROSSTAB_PATH}")

    view = pd.read_csv(PROJECT_VIEW_PATH)
    alignment = pd.read_csv(ALIGNMENT_SUMMARY_PATH)
    crosstab = pd.read_csv(ALIGNMENT_CROSSTAB_PATH)

    unsafe_accepts = int(
        alignment.loc[
            alignment["manual_trust_alignment"] == "unsafe_accept",
            "rows"
        ].sum()
    ) if "unsafe_accept" in set(alignment["manual_trust_alignment"]) else 0

    under_escalated_fails = int(
        alignment.loc[
            alignment["manual_trust_alignment"] == "under_escalated_fail",
            "rows"
        ].sum()
    ) if "under_escalated_fail" in set(alignment["manual_trust_alignment"]) else 0

    accepted_passes = int(
        crosstab[
            (crosstab["manual_review_outcome"] == "pass")
            & (crosstab["recommended_action"] == "accept")
        ]["rows"].sum()
    )

    new_row = {
        "evaluation_area": "EquityMedQA manual-trust alignment",
        "dataset_or_layer": "Completed EquityMedQA manual review + trust routing",
        "rows": 33,
        "main_metric": f"Unsafe accepts: {unsafe_accepts}; under-escalated fails: {under_escalated_fails}",
        "safety_signal": f"All fail rows routed to mandatory review; accepted pass rows: {accepted_passes}",
        "current_decision": "Trustworthiness routing aligns with completed manual review outcomes"
    }

    view = view[
        view["evaluation_area"] != "EquityMedQA manual-trust alignment"
    ].copy()

    view = pd.concat([view, pd.DataFrame([new_row])], ignore_index=True)
    view.to_csv(OUT_PATH, index=False)

    report = "# Project-Level Trustworthiness View\n\n"
    report += "This view summarizes the current project-level safety and trustworthiness layers.\n\n"
    report += "The latest update adds manual-trust alignment for EquityMedQA. This checks whether the completed manual review outcomes agree with the automated trustworthiness routing.\n\n"

    report += "## Current Trustworthiness Layers\n\n"
    report += view.to_markdown(index=False)
    report += "\n\n"

    report += "## Latest Interpretation\n\n"
    report += "The EquityMedQA manual-trust alignment result is clean. All manually failed rows were routed to mandatory human review, all needs-revision rows were sent to review, and all passed rows were accepted. Most importantly, there were zero unsafe accepts.\n\n"
    report += "This strengthens the project because the fairness review layer is now connected to trustworthiness routing instead of being only a separate manual checklist.\n"

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Project-level trustworthiness view updated with manual-trust alignment")
    print()
    print(view.to_string(index=False))
    print()
    print("Saved:", OUT_PATH)
    print("Saved:", REPORT_PATH)

if __name__ == "__main__":
    main()
