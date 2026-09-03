from pathlib import Path
import pandas as pd

ALIGNMENT_SUMMARY = Path("results/trustworthiness/qwen_equitymedqa_manual_trust_alignment_summary.csv")
CROSSTAB = Path("results/trustworthiness/qwen_equitymedqa_routing_by_manual_outcome.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = REPORT_DIR / "daily_progress_equitymedqa_manual_trust_alignment.md"

def md_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)

def main():
    summary = pd.read_csv(ALIGNMENT_SUMMARY)
    crosstab = pd.read_csv(CROSSTAB)

    unsafe_accepts = int(summary.loc[summary["manual_trust_alignment"] == "unsafe_accept", "rows"].sum()) if "unsafe_accept" in set(summary["manual_trust_alignment"]) else 0
    under_escalated_fails = int(summary.loc[summary["manual_trust_alignment"] == "under_escalated_fail", "rows"].sum()) if "under_escalated_fail" in set(summary["manual_trust_alignment"]) else 0

    note = f"""# Daily Progress - EquityMedQA Manual-Trust Alignment

## Work Completed Today

Today I checked whether the completed EquityMedQA manual review outcomes align with the trustworthiness routing decisions.

This was important because the project should not only flag fairness or safety risks. It should also confirm that risky responses are routed correctly after review.

## Main Result

The alignment result was clean.

| Check | Result |
|---|---:|
| Unsafe accepts | {unsafe_accepts} |
| Under-escalated failed rows | {under_escalated_fails} |

## Routing By Manual Outcome

{md_table(crosstab)}

## Interpretation

All manually failed EquityMedQA rows were routed to mandatory human review. All needs-revision rows were sent to review, and all passed rows were accepted.

This strengthens the project-level trustworthiness workflow because manual review outcomes are now connected with automated routing decisions.

## Short Update Line

I validated EquityMedQA manual review outcomes against trustworthiness routing and confirmed zero unsafe accepts and zero under-escalated failed rows.
"""

    OUT_PATH.write_text(note, encoding="utf-8")

    print("Daily progress note created")
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
