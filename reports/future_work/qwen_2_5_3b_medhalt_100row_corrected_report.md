# Qwen2.5-3B Med-HALT 100-Row Corrected Evaluation

## Purpose

This evaluation tests Qwen2.5-3B-Instruct as a stronger local open-source model for healthcare hallucination detection.

The same 100-row Med-HALT balanced sample is used so it can be compared with the earlier Qwen 0.5B and Qwen 1.5B results.

## Results

| Metric | Value |
|---|---:|
| Rows tested | 100 |
| Parse success | 1.0 |
| Accuracy | 0.66 |
| Precision | 0.598 |
| Recall | 0.98 |
| F1 score | 0.743 |
| True positives | 49 |
| True negatives | 17 |
| False positives | 33 |
| False negatives | 1 |
| Error review rows | 34 |
| Total runtime seconds | 397.94 |
| Average runtime per row | 3.97 |

## Interpretation

This result should be treated as future-work evidence until it is reviewed and added to the main project comparison.

If the false negative count stays low, Qwen2.5-3B is useful as a conservative healthcare safety model. If false positives are high, it may still be valuable for review routing, but not necessarily for automatic acceptance.
