# EquityMedQA Explainable Fairness-Risk Report

## Purpose

This analysis makes the EquityMedQA fairness-risk score easier to explain.

Instead of only saying that a response was accepted, reviewed, or sent to mandatory human review, this version shows the risk reasons behind the decision.

## Explainable Risk Band Counts

| Risk band | Rows |
|---|---:|
| High | 21 |
| Medium | 9 |
| Low | 3 |

## Risk By Manual Review Outcome

| manual_review_outcome | explainable_fairness_risk_band | rows |
| --- | --- | --- |
| fail | high | 16 |
| needs_revision | high | 5 |
| needs_revision | medium | 9 |
| pass | low | 3 |

## Score Summary By Manual Outcome

| manual_review_outcome | rows | mean_explainable_fairness_risk_score | min_explainable_fairness_risk_score | max_explainable_fairness_risk_score |
| --- | --- | --- | --- | --- |
| fail | 16 | 0.955 | 0.78 | 1.0 |
| needs_revision | 14 | 0.629 | 0.38 | 0.9 |
| pass | 3 | 0.027 | 0.0 | 0.08 |

## Main Risk Reasons

| risk_reason | rows |
| --- | --- |
| clinical validation required | 29 |
| response incomplete or truncated | 26 |
| manual overall risk is high | 16 |
| manual review outcome failed | 16 |
| unsafe clinical safety label | 15 |
| missing professional care guidance | 15 |
| clinical safety concern | 14 |
| manual overall risk is medium | 14 |
| manual review outcome needs revision | 14 |
| stereotype signal present | 13 |
| fairness concern | 10 |
| unfair response label | 8 |
| paired prompt responses not comparable | 7 |
| paired prompt responses materially inconsistent | 4 |
| no major manual risk reason | 2 |

## Interpretation

The explainable fairness-risk score helps connect manual review labels with the trustworthiness workflow.

This is useful because a healthcare fairness review should not only say that a response is risky. It should also explain why the response is risky, such as unsafe clinical guidance, fairness concern, stereotype signal, missing care guidance, incomplete response, or required clinical validation.

## Project Decision

The fairness-risk score should be kept as an explainable support signal. It should help justify routing decisions, but it should not replace manual or clinical validation.
