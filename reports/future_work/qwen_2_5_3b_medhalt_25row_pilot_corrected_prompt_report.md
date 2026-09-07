# Qwen2.5-3B Med-HALT 25-Row Pilot: Corrected Prompt

## Purpose

This corrected pilot evaluates Qwen2.5-3B-Instruct using the medical knowledge, question, and actual answer.

The earlier runtime test confirmed that the model could run locally, but this corrected version is the valid evaluation prompt.

## Result

| Metric | Value |
|---|---:|
| Rows tested | 25 |
| Parse success | 1.0 |
| Accuracy | 0.72 |
| Precision | 0.667 |
| Recall | 1.0 |
| F1 score | 0.8 |
| True positives | 14 |
| True negatives | 4 |
| False positives | 7 |
| False negatives | 0 |
| Total runtime seconds | 93.66 |
| Average runtime per row | 3.74 |

## Interpretation

This corrected pilot should be used to decide whether Qwen2.5-3B is worth scaling to a full 100-row Med-HALT comparison.
