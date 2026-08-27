# Qwen 1.5B Med-HALT Prompt Robustness Pilot

## Purpose

This pilot tests whether Qwen's weak hallucination detection was caused mainly by the original prompt and numerical labels.

The same 20 balanced cases were evaluated using:

- a zero-shot numerical prompt using `0` and `1`;
- a few-shot descriptive prompt using `SUPPORTED` and `HALLUCINATED`.

## Results

| prompt_variant | dataset_name | rows | accuracy | precision | recall | f1_score | roc_auc | predicted_hallucination_rate | mean_hallucination_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fewshot_descriptive | combined | 20 | 0.7000 | 0.6429 | 0.9000 | 0.7500 | 0.9200 | 0.7000 | 0.6469 |
| fewshot_descriptive | medhalt_reasoning_fake | 10 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 0.8516 |
| fewshot_descriptive | medhalt_reasoning_fct | 10 | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 1.0000 | 0.4000 | 0.4422 |
| zero_shot_numeric | combined | 20 | 0.7500 | 0.6923 | 0.9000 | 0.7826 | 0.9400 | 0.6500 | 0.5935 |
| zero_shot_numeric | medhalt_reasoning_fake | 10 | 0.6000 | 0.5556 | 1.0000 | 0.7143 | 1.0000 | 0.9000 | 0.7727 |
| zero_shot_numeric | medhalt_reasoning_fct | 10 | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.9600 | 0.4000 | 0.4143 |

## Combined Comparison

| prompt_variant | dataset_name | rows | accuracy | precision | recall | f1_score | roc_auc | predicted_hallucination_rate | mean_hallucination_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fewshot_descriptive | combined | 20 | 0.7000 | 0.6429 | 0.9000 | 0.7500 | 0.9200 | 0.7000 | 0.6469 |
| zero_shot_numeric | combined | 20 | 0.7500 | 0.6923 | 0.9000 | 0.7826 | 0.9400 | 0.6500 | 0.5935 |

## Initial Interpretation

The prompt variant with the highest combined F1 score was `zero_shot_numeric`, with an F1 score of 0.7826 and ROC-AUC of 0.9400.

This is a small prompt-robustness pilot. It should not be treated as a final model evaluation. A prompt should only replace the existing approach if it improves recall, F1, and ranking behavior without collapsing toward one prediction label.
