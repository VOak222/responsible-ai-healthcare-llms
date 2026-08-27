# Qwen Med-HALT Prompt Robustness Pilot

## Purpose

This pilot tests whether Qwen's weak hallucination detection was caused mainly by the original prompt and numerical labels.

The same 20 balanced cases were evaluated using:

- a zero-shot numerical prompt using `0` and `1`;
- a few-shot descriptive prompt using `SUPPORTED` and `HALLUCINATED`.

## Results

| prompt_variant | dataset_name | rows | accuracy | precision | recall | f1_score | roc_auc | predicted_hallucination_rate | mean_hallucination_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fewshot_descriptive | combined | 20 | 0.4000 | 0.3750 | 0.3000 | 0.3333 | 0.4800 | 0.4000 | 0.4413 |
| fewshot_descriptive | medhalt_reasoning_fake | 10 | 0.3000 | 0.3750 | 0.6000 | 0.4615 | 0.0000 | 0.8000 | 0.6025 |
| fewshot_descriptive | medhalt_reasoning_fct | 10 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.9200 | 0.0000 | 0.2802 |
| zero_shot_numeric | combined | 20 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4600 | 0.0000 | 0.0999 |
| zero_shot_numeric | medhalt_reasoning_fake | 10 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.0000 | 0.0836 |
| zero_shot_numeric | medhalt_reasoning_fct | 10 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1600 | 0.0000 | 0.1162 |

## Combined Comparison

| prompt_variant | dataset_name | rows | accuracy | precision | recall | f1_score | roc_auc | predicted_hallucination_rate | mean_hallucination_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fewshot_descriptive | combined | 20 | 0.4000 | 0.3750 | 0.3000 | 0.3333 | 0.4800 | 0.4000 | 0.4413 |
| zero_shot_numeric | combined | 20 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4600 | 0.0000 | 0.0999 |

## Initial Interpretation

The prompt variant with the highest combined F1 score was `fewshot_descriptive`, with an F1 score of 0.3333 and ROC-AUC of 0.4800.

This is a small prompt-robustness pilot. It should not be treated as a final model evaluation. A prompt should only replace the existing approach if it improves recall, F1, and ranking behavior without collapsing toward one prediction label.
