# Qwen 1.5B Med-HALT Trustworthiness Summary

## Evaluation

This evaluation combines Qwen 1.5B hallucination probability with claim-level semantic evidence grounding.

- Qwen model: Qwen/Qwen2.5-1.5B-Instruct
- Grounding model: sentence-transformers/all-MiniLM-L6-v2
- Rows evaluated: 100
- Unsupported-claim similarity threshold: 0.45

## Overall Result

| model_id | rows_scored | embedding_model_id | mean_hallucination_risk | mean_semantic_grounding_score | mean_clinical_trustworthiness_score | trustworthiness_hallucination_roc_auc | unsupported_claim_rows | unsupported_high_impact_claim_rows | evidence_conflict_rows | mandatory_human_review_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 100 | sentence-transformers/all-MiniLM-L6-v2 | 0.6039 | 0.4273 | 0.4938 | 0.8648 | 56 | 0 | 12 | 68 |

## Results by Ground-Truth Label

| is_hallucinated | rows | mean_hallucination_risk | mean_grounding_score | mean_trustworthiness_score | unsupported_claim_rows | label_description |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 50 | 0.2550 | 0.4922 | 0.6695 | 26 | supported |
| 1 | 50 | 0.9527 | 0.3625 | 0.3182 | 30 | hallucinated |

## Recommended Actions

| recommended_action | row_count |
| --- | --- |
| mandatory_human_review | 68 |
| accept | 25 |
| human_review | 6 |
| revise | 1 |

## Trustworthiness Formula

The score is calculated from:

- 45% model hallucination risk;
- 35% semantic grounding risk;
- 15% unsupported-claim risk;
- 5% evidence-conflict risk.

Rows containing unsupported high-impact clinical claims, evidence conflicts, or scores below 0.45 are routed to mandatory human review.

## Interpretation

This is an automated screening score, not a clinician judgment. Its purpose is to identify answers that require review and to compare model behavior consistently. The ground-truth label is used only to evaluate the resulting score and is not used when calculating an individual row's trustworthiness.
