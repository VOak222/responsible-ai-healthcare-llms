# EquityMedQA Manual Review Summary

## What Was Reviewed

I reviewed the 33 shortlisted EquityMedQA rows from the Qwen fairness evaluation run.

These rows were selected because the automated screening found possible fairness, safety, completeness, care guidance, stereotype, or paired-response consistency issues.

## Manual Review Outcome

| manual_review_outcome | rows |
| --- | --- |
| fail | 16 |
| needs_revision | 14 |
| pass | 3 |

This tells us how many shortlisted responses were accepted, marked for revision, or kept for further review after manual checking.

## Manual Review Label Summary

| field | label | rows |
| --- | --- | --- |
| manual_clinical_safety | unsafe | 15 |
| manual_clinical_safety | concern | 14 |
| manual_clinical_safety | safe | 4 |
| manual_fairness | fair | 15 |
| manual_fairness | concern | 10 |
| manual_fairness | unfair | 8 |
| manual_stereotype | absent | 20 |
| manual_stereotype | present | 13 |
| manual_paired_consistency | not_applicable | 21 |
| manual_paired_consistency | not_comparable | 7 |
| manual_paired_consistency | materially_inconsistent | 4 |
| manual_paired_consistency | consistent | 1 |
| manual_care_guidance | missing | 15 |
| manual_care_guidance | appropriate | 9 |
| manual_care_guidance | not_required | 9 |
| manual_response_completeness | truncated | 26 |
| manual_response_completeness | complete | 7 |
| manual_overall_risk | high | 16 |
| manual_overall_risk | medium | 14 |
| manual_overall_risk | low | 3 |
| manual_review_outcome | fail | 16 |
| manual_review_outcome | needs_revision | 14 |
| manual_review_outcome | pass | 3 |
| clinical_validation_required | yes | 29 |
| clinical_validation_required | no | 4 |

These labels break down the review from different angles: clinical safety, fairness, stereotype risk, paired consistency, care guidance, response completeness, and overall risk.

## Review By Fairness Category

| fairness_category | manual_overall_risk | manual_review_outcome | rows |
| --- | --- | --- | --- |
| age | high | fail | 1 |
| age | low | pass | 1 |
| age | medium | needs_revision | 1 |
| disability | low | pass | 1 |
| disability | medium | needs_revision | 1 |
| gender_sexuality | medium | needs_revision | 4 |
| general_equity_or_clinical_prompt | high | fail | 1 |
| general_equity_or_clinical_prompt | low | pass | 1 |
| general_equity_or_clinical_prompt | medium | needs_revision | 3 |
| geography | medium | needs_revision | 1 |
| income_access | medium | needs_revision | 1 |
| race_ethnicity | high | fail | 1 |
| religion_culture | high | fail | 2 |
| religion_culture | medium | needs_revision | 2 |

This helps identify which fairness categories created the most review concerns.


## Trustworthiness Routing

| recommended_action | rows |
| --- | --- |
| mandatory_human_review | 18 |
| human_review | 12 |
| accept | 3 |

This shows how the reviewed EquityMedQA responses are routed by the trustworthiness layer. Responses with higher uncertainty or safety concern are sent to human review instead of being accepted directly.


## Interpretation

The EquityMedQA layer adds a fairness and health-equity check to the project. It goes beyond hallucination detection by asking whether the model gives careful, complete, and consistent responses when prompts include demographic or access-related details.

The main takeaway is that many responses can sound reasonable but still need review because they may miss care guidance, provide incomplete support, or behave inconsistently across paired demographic prompts.

## Project Decision

EquityMedQA should remain part of the overall Responsible AI evaluation framework.

The current best use is as a manual-review support layer. It should not be treated as a final fairness judgment by itself, but it gives a structured way to find responses that need human checking before being trusted.
