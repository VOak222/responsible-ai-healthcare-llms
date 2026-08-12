# Model Comparison Summary

## Overall Results

| rows_evaluated | baseline_accuracy | baseline_f1_score | baseline_false_positives | baseline_false_negatives | mean_claim_grounding_score | mean_trustworthiness_score | accept_rows | revise_rows | human_review_rows | mandatory_human_review_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4000.0 | 0.764 | 0.7632898696088265 | 466.0 | 478.0 | 0.6899430323071964 | 0.6795472150873322 | 161.0 | 2510.0 | 1231.0 | 98.0 |

## Evaluator Comparison

| component | role_in_pipeline | main_metric | main_value | secondary_metric | secondary_value | risk_signal | risk_value | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | Predicts whether the full answer is hallucinated | F1 score | 0.7632898696088265 | Accuracy | 0.764 | False negatives | 478 | Good first baseline, but missed hallucinated healthcare answers still require extra review logic. |
| Claim-level grounding | Breaks answers into claims and checks support from evidence | Mean grounding score | 0.6899430323071964 | Unsupported claims | 219.0 | Unsupported high-impact claims | 54 | Adds evidence-level checking so the system is not only relying on the classifier label. |
| Semantic claim verification | Compares each claim with the closest evidence sentence | Mean combined support score | 0.7250198785195171 | Mean semantic support score | 0.7400528126105114 | Combined unsupported claims | 95 | First semantic evaluator step. Current run may be strict because it uses TF-IDF fallback. |
| Clinical trustworthiness score | Combines hallucination, grounding, safety, and confidence | Mean trustworthiness score | 0.6795472150873322 | Human review rows | 1329.0 | Mandatory human review rows | 98 | Turns model outputs into an operational decision: accept, revise, human review, or mandatory human review. |

## TP/TN/FP/FN Counts

| case_type | row_count |
| --- | --- |
| true_negative | 1534 |
| true_positive | 1522 |
| false_negative | 478 |
| false_positive | 466 |

## Trustworthiness Action Summary

| recommended_action | row_count |
| --- | --- |
| revise | 2510 |
| human_review | 1231 |
| accept | 161 |
| mandatory_human_review | 98 |

## Key Takeaway

The baseline TF-IDF model performs reasonably well, but false negatives show that classifier-only evaluation is not enough for healthcare. Adding semantic claim verification improved grounding and reduced mandatory human review rows. The clinical trustworthiness score now combines hallucination risk, semantic grounding, unsupported claims, and contradiction risk into a practical review decision.
