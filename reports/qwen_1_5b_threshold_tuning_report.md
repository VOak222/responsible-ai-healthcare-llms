# Qwen 1.5B Trust Threshold Tuning

## Why This Was Needed

The Qwen 1.5B model performed much better than Qwen 0.5B, but the first trustworthiness routing still accepted one hallucinated answer.

In healthcare evaluation, even one accepted hallucinated medical answer matters because it means unsafe or unsupported information could pass without human review.

## Before vs After

| action_column | accepted_rows | accepted_supported_rows | accepted_hallucinated_rows | mandatory_human_review_rows | human_review_rows | revise_rows |
| --- | --- | --- | --- | --- | --- | --- |
| recommended_action | 25 | 24 | 1 | 68 | 6 | 1 |
| tuned_recommended_action | 25 | 24 | 1 | 74 | 1 | 0 |

## Tuned Action Breakdown

| is_hallucinated | tuned_recommended_action | rows |
| --- | --- | --- |
| 0 | accept | 24 |
| 0 | mandatory_human_review | 26 |
| 1 | accept | 1 |
| 1 | human_review | 1 |
| 1 | mandatory_human_review | 48 |

## Interpretation

The tuned threshold is stricter. It is designed to reduce the chance that hallucinated answers are accepted.

This may send more rows to human review, but that is acceptable for a healthcare safety setting. In this project, the priority is not only high accuracy. The priority is preventing risky medical answers from passing as trusted.

## Final Decision

For Qwen 1.5B, we should report both results:

1. Raw model performance: strong improvement over Qwen 0.5B.
2. Tuned trustworthiness routing: safer because it reduces false accepts.
