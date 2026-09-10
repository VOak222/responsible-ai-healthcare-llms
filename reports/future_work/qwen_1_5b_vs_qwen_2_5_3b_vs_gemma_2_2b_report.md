# Three-Model Med-HALT Comparison

## Purpose

This comparison checks whether newer small open-source models improve healthcare hallucination detection compared with the current best model, Qwen 1.5B.

All models are compared on the same 100-row Med-HALT balanced sample.

## Results

| Model | Rows | Accuracy | Precision | Recall | F1 | False Positives | False Negatives | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen 1.5B | 100 | 0.86 | 0.790 | 0.98 | 0.875 | 13 | 1 | Best main model so far |
| Qwen2.5-3B | 100 | 0.66 | 0.598 | 0.98 | 0.743 | 33 | 1 | Useful as conservative secondary reviewer |
| Gemma 2 2B | 100 | 0.65 | 0.588 | 1.00 | 0.741 | 35 | 0 | Safest recall, but too many false positives for main model |

## Interpretation

Qwen 1.5B remains the best main model because it has the strongest balance between accuracy, precision, recall, and F1 score.

Gemma 2 2B achieved the safest hallucination-catching behavior with zero false negatives. However, it over-flagged many supported answers as hallucinated, so it should not replace Qwen 1.5B as the main model.

Qwen2.5-3B behaved similarly to Gemma. It was conservative, but did not improve enough to replace Qwen 1.5B.

## Current Project Decision

Use Qwen 1.5B as the main local model.

Keep Qwen2.5-3B and Gemma 2 2B as future secondary-review candidates, especially for disagreement-based human review routing.

Llama 3.2 3B Instruct is still waiting for Hugging Face access and should be tested next.
