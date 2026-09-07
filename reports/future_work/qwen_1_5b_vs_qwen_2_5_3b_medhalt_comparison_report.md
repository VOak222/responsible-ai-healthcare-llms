# Qwen 1.5B vs Qwen2.5-3B Med-HALT Comparison

## Purpose

This comparison checks whether Qwen2.5-3B should replace Qwen 1.5B as the main local open-source model for healthcare hallucination detection.

Both models are compared on the same 100-row Med-HALT balanced sample.

## Model Comparison

| model | rows | accuracy | precision | recall | f1_score | true_positive | true_negative | false_positive | false_negative | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 1.5B | 100 | 0.86 | 0.79 | 0.98 | 0.875 | 49 | 37 | 13 | 1 | Best main model so far because it balances recall and fewer false positives |
| Qwen2.5-3B | 100 | 0.66 | 0.598 | 0.98 | 0.743 | 49 | 17 | 33 | 1 | Strong recall but too many false positives; useful as secondary safety reviewer |

## Key Finding

Qwen2.5-3B successfully ran locally and produced parseable outputs for all rows.

However, it should not replace Qwen 1.5B as the main model yet. Qwen2.5-3B has strong recall, but it produces many more false positives. This means it is safer than a weak model, but less balanced than Qwen 1.5B.

## False Negative Check

| Check | Result |
|---|---|
| Qwen 1.5B false negative rows | 1 |
| Qwen2.5-3B false negative rows | 1 |
| Same false negative row | no |

## Disagreement Cases

The two models disagreed on 22 rows.

These rows are useful for future manual review because disagreement often highlights uncertain cases.

## Project Decision

Qwen 1.5B remains the best main model so far.

Qwen2.5-3B should be kept as a future-work candidate or secondary safety reviewer because it is conservative and catches hallucinations well, but it over-flags supported responses.
