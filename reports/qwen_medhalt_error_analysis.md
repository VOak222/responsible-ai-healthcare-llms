# Qwen Med-HALT Error Analysis

## Error Counts

| error_type     |   rows |
|:---------------|-------:|
| correct        |     52 |
| false_negative |     44 |
| false_positive |      4 |

## Probability Behavior

| dataset_name           |   is_hallucinated |   rows |   mean_hallucination_probability |   median_hallucination_probability |   minimum_probability |   maximum_probability |   predicted_hallucination_rate |   accuracy |
|:-----------------------|------------------:|-------:|---------------------------------:|-----------------------------------:|----------------------:|----------------------:|-------------------------------:|-----------:|
| medhalt_reasoning_fake |                 0 |     25 |                         0.199938 |                           0.182678 |             0.0999592 |              0.376156 |                           0    |       1    |
| medhalt_reasoning_fake |                 1 |     25 |                         0.164084 |                           0.136084 |             0.0355749 |              0.412422 |                           0    |       0    |
| medhalt_reasoning_fct  |                 0 |     25 |                         0.275598 |                           0.232711 |             0.0521835 |              0.759451 |                           0.16 |       0.84 |
| medhalt_reasoning_fct  |                 1 |     25 |                         0.359518 |                           0.398123 |             0.0397492 |              0.800429 |                           0.24 |       0.24 |

## Interpretation

Qwen2.5-0.5B-Instruct strongly favored the supported label. It predicted only 6 of 50 hallucinated answers as hallucinated.

The combined ROC-AUC was approximately 0.51, indicating that the current zero-shot numeric-label prompt provides almost no reliable separation between supported and hallucinated answers.

## Next Step

Run a smaller prompt-robustness experiment using descriptive labels and few-shot examples before deciding whether the model is unsuitable for hallucination classification.
