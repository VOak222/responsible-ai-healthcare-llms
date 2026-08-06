# Evidence Grounding Diagnostic

## Purpose

This diagnostic tested whether hallucinated answers have lower word overlap with the provided medical knowledge compared with grounded answers.

## Result

| Label | Count | Mean Overlap | Median Overlap |
|---|---:|---:|---:|
| Not Hallucinated | 10000 | 0.5000 | 0.50 |
| Hallucinated | 10000 | 0.5235 | 0.52 |

## Interpretation

The hallucinated answers did not show lower overlap with the provided knowledge. In fact, their average overlap was slightly higher.

This suggests that simple token overlap is not enough for healthcare hallucination detection. A hallucinated answer can reuse medical terms from the evidence while still changing the conclusion, direction, or meaning.

## Takeaway

Evidence grounding needs semantic comparison, contradiction detection, or claim-level verification instead of only word overlap.