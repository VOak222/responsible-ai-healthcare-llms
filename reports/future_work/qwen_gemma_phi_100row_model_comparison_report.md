# Week 6 Model Expansion: Qwen, Gemma, and Phi Comparison

## Purpose

This comparison checks whether newer or slightly larger local open-source models improve healthcare hallucination detection compared with the current best model, Qwen 1.5B.

All models in this table are compared on the same 100-row Med-HALT balanced sample.

## Results

| Model | Rows | Accuracy | Precision | Recall | F1 | False Positives | False Negatives | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen 1.5B | 100 | 0.86 | 0.790 | 0.98 | 0.875 | 13 | 1 | Best main model so far |
| Qwen2.5-3B | 100 | 0.66 | 0.598 | 0.98 | 0.743 | 33 | 1 | Conservative secondary reviewer candidate |
| Gemma 2 2B | 100 | 0.65 | 0.588 | 1.00 | 0.741 | 35 | 0 | Safest recall, but too many false positives |
| Phi-3.5-mini | 100 | 0.67 | 0.608 | 0.96 | 0.744 | 31 | 2 | Not preferred; slower and missed more hallucinations |

## Interpretation

Qwen 1.5B remains the best main model because it has the strongest balance of accuracy, precision, recall, and F1 score.

Gemma 2 2B had the safest recall result because it produced zero false negatives. However, it over-flagged many supported answers as hallucinated, so it is better as a secondary safety reviewer than as the main model.

Qwen2.5-3B also behaved conservatively and can be used as a secondary reviewer candidate, but it did not improve over Qwen 1.5B.

Phi-3.5-mini did not beat Qwen 1.5B. It missed two hallucinated answers, produced many false positives, and was slower during local evaluation.

## Current Project Decision

Qwen 1.5B remains the main local hallucination detection model.

Gemma 2 2B and Qwen2.5-3B remain useful secondary reviewer candidates.

Phi-3.5-mini is not preferred for the current main model because it did not improve the safety or balance of the system.

Llama 3.2 3B was tested only as a 25-row pilot and was not advanced to the 100-row comparison.

Qwen3-1.7B should be tested next.
