from pathlib import Path

REPORT_PATH = Path("reports/week4_progress_equitymedqa_explainable_fairness_risk.md")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report = """# Week 4 Progress Note: EquityMedQA Review and Explainable Fairness Risk

## Work Completed

This week, I completed the next safety layer for the healthcare LLM evaluation project by connecting EquityMedQA manual review outcomes with trustworthiness routing and explainable fairness-risk scoring.

The goal was to move beyond simple model outputs and check whether risky healthcare responses are being correctly identified, reviewed, and explained.

## EquityMedQA Manual Review

I reviewed the 33 flagged EquityMedQA responses and categorized them based on clinical safety, fairness, stereotype risk, paired prompt consistency, care guidance, response completeness, and overall risk.

The manual review outcomes were:

| Outcome | Rows |
|---|---:|
| Fail | 16 |
| Needs revision | 14 |
| Pass | 3 |

Clinical validation was required for 29 out of the 33 reviewed responses. This shows that most flagged responses still needed human or clinical review before they could be considered safe.

## Trustworthiness Routing Alignment

I then checked whether the trustworthiness routing matched the completed manual review outcomes.

The routing was aligned with the manual review results:

| Check | Result |
|---|---:|
| Unsafe accepts | 0 |
| Under-escalated fails | 0 |
| Accepted pass cases | 3 |

This means the failed responses were not accepted automatically, and the safe passing responses were the only ones accepted.

## Explainable Fairness-Risk Scoring

I also added an explainable fairness-risk layer so that the project can show why a response is risky, not just whether it is risky.

The explainable risk bands were:

| Risk band | Rows |
|---|---:|
| High | 21 |
| Medium | 9 |
| Low | 3 |

All 16 failed manual review cases were placed in the high-risk band. All 3 passing cases were placed in the low-risk band.

The main risk reasons included clinical validation requirements, incomplete or truncated responses, unsafe clinical safety labels, missing professional care guidance, stereotype signals, and fairness concerns.

## Project Impact

This work strengthens the project because the evaluation now connects three important layers:

1. Manual review judgment
2. Trustworthiness routing
3. Explainable risk reasons

This makes the system easier to explain in a professional review setting. Instead of only saying that a response was rejected or sent to review, the project can now explain the reason behind the routing decision.

## Current Status

The EquityMedQA safety review layer is now stronger and more explainable. The project-level trustworthiness view has also been updated to include this explainable fairness-risk result.

The dashboard/reporting layer is still in progress and can be improved later to present all dataset results together more clearly.
"""

REPORT_PATH.write_text(report, encoding="utf-8")

print("Week 4 progress note created")
print("Saved:", REPORT_PATH)
