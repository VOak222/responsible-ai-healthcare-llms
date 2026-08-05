# Manual Error Review Notes

## Purpose

After training the TF-IDF + Logistic Regression baseline, I reviewed sample false positives and false negatives to understand where the model is making mistakes.

## Error Types

| Error Type | Meaning |
|---|---|
| False Positive | The answer was correct, but the model predicted hallucinated |
| False Negative | The answer was hallucinated, but the model predicted not hallucinated |

## Observations

False positives often happened when correct medical answers contained dense biomedical terminology, mechanism-based explanations, or cautious clinical wording. The model may be treating complex technical language as suspicious.

False negatives happened when hallucinated answers sounded fluent, confident, and medically plausible. This suggests the baseline is relying more on surface-level text patterns instead of truly checking whether the answer is grounded in the provided knowledge.

## Key Takeaway

The baseline is useful as a first benchmark, but it is not enough for healthcare safety. Missing hallucinated answers is especially important because false negatives can allow incorrect medical information to pass as reliable.

## Next Step

The next step is to improve evaluation by checking model confidence thresholds and then compare this baseline with stronger approaches.