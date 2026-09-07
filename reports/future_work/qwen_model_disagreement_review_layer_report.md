# Qwen Model Disagreement Review Layer

## Purpose

This analysis uses disagreement between Qwen 1.5B and Qwen2.5-3B as a safety signal.

The idea is simple: when two local healthcare evaluation models disagree, the row should be treated as uncertain instead of being accepted without review.

## Disagreement Summary

| disagreement_type | disagreement_recommended_action | rows |
| --- | --- | --- |
| qwen_3b_over_flagged_supported_answer | human_review | 20 |
| qwen_1_5b_missed_hallucination | mandatory_human_review | 1 |
| qwen_3b_missed_hallucination | mandatory_human_review | 1 |

## Disagreement By True Label

| is_hallucinated | disagreement_type | rows |
| --- | --- | --- |
| 0 | qwen_3b_over_flagged_supported_answer | 20 |
| 1 | qwen_1_5b_missed_hallucination | 1 |
| 1 | qwen_3b_missed_hallucination | 1 |

## High-Priority Cases

Mandatory human review cases: 2

These are cases where one model missed a hallucinated response while the other model caught it.

## Interpretation

Qwen 1.5B remains the stronger main model because it has better balance between recall and false positives.

Qwen2.5-3B is still useful because it disagrees with Qwen 1.5B on uncertain cases. These disagreement rows can help identify responses that need extra review.

## Project Decision

Model disagreement should be used as an additional review signal in the future governance layer.

A practical rule is:

If Qwen 1.5B and Qwen2.5-3B disagree, route the case to human review. If the disagreement involves a missed hallucination, route it to mandatory human review.
