# Qwen 0.5B Med-HALT Model Suitability Decision

## Experiment

Qwen/Qwen2.5-0.5B-Instruct was evaluated as a hallucination classifier on a balanced 100-row Med-HALT sample. A second 20-row pilot compared the original zero-shot numerical prompt with a few-shot descriptive prompt.

## Main Results

| Prompt | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Zero-shot numerical pilot | 0.50 | 0.00 | 0.00 | 0.46 |
| Few-shot descriptive pilot | 0.40 | 0.30 | 0.33 | 0.48 |

The original 100-row evaluation produced 44 false negatives and only detected 6 of the 50 hallucinated answers.

## Interpretation

The few-shot prompt increased hallucination recall, but overall discrimination remained close to random. Performance was also inconsistent between the two Med-HALT subsets. This means the improvement does not generalize reliably.

The model often defaulted to the supported label and showed poorly calibrated probabilities. Choosing a new threshold from only 20 pilot rows would risk overfitting and would not solve the inconsistent dataset behavior.

## Decision

Qwen2.5-0.5B-Instruct is not suitable as the project's Med-HALT hallucination classifier in its current form.

The results are retained as a transparent negative experiment, but they will not be combined with the Qwen EquityMedQA fairness score. The next model experiment should use a larger open-source instruction model within the project's local 0.5B-3B constraint.
