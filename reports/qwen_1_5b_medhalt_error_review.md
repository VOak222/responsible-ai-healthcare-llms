# Qwen 1.5B Med-HALT Error Review

## What We Checked

This review looks at the Qwen 1.5B Med-HALT run after trustworthiness scoring.

The goal is not only to report accuracy. The goal is to find risky cases where a hallucinated medical answer may have been accepted or not escalated strongly enough.

## Overall Results

| Metric | Value |
|---|---:|
| Rows reviewed | 100 |
| Accuracy | 0.86 |
| False negatives / missed hallucinations | 1 |
| False positives / supported answers flagged | 13 |
| Hallucinated rows accepted | 1 |
| Manual review shortlist rows | 1 |
| Mean trustworthiness score | 0.494 |

## Error Type Summary

| error_type | rows |
| --- | --- |
| true_positive_detected_hallucination | 49 |
| true_negative_accepted_supported | 37 |
| false_positive_flagged_supported | 13 |
| false_negative_missed_hallucination | 1 |

## Action By True Label

| is_hallucinated | recommended_action | rows |
| --- | --- | --- |
| 0 | accept | 24 |
| 0 | mandatory_human_review | 25 |
| 0 | revise | 1 |
| 1 | accept | 1 |
| 1 | human_review | 6 |
| 1 | mandatory_human_review | 43 |

## Interpretation

Qwen 1.5B is much stronger than Qwen 0.5B on the balanced Med-HALT sample, but the important healthcare question is whether hallucinated answers are still slipping through.

The manual shortlist focuses on exactly those cases. These are the rows we should inspect before calling Qwen 1.5B reliable.

## Next Step

Open the false-accept review shortlist and manually inspect the highest-risk rows. If accepted hallucinated rows are truly unsafe, we should tighten the trustworthiness threshold before moving to the next dataset.
