# Qwen Med-HALT Balanced Sample Evaluation

## Evaluation Design

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Total rows: 100
- Sampling: 25 supported and 25 hallucinated rows from each Med-HALT dataset
- Classification: 0 = supported, 1 = hallucinated
- Random state: 42

## Sample Distribution

| dataset_name           |   is_hallucinated |   rows |
|:-----------------------|------------------:|-------:|
| medhalt_reasoning_fake |                 0 |     25 |
| medhalt_reasoning_fake |                 1 |     25 |
| medhalt_reasoning_fct  |                 0 |     25 |
| medhalt_reasoning_fct  |                 1 |     25 |

## Performance

| dataset_name           |   rows |   accuracy |   precision |   recall |   f1_score |   roc_auc |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|-------:|-----------:|------------:|---------:|-----------:|----------:|----------------:|-----------------:|-----------------:|----------------:|
| medhalt_reasoning_fake |     50 |       0.5  |         0   |     0    |   0        |    0.3232 |              25 |                0 |               25 |               0 |
| medhalt_reasoning_fct  |     50 |       0.54 |         0.6 |     0.24 |   0.342857 |    0.6384 |              21 |                4 |               19 |               6 |
| combined               |    100 |       0.52 |         0.6 |     0.12 |   0.2      |    0.5096 |              46 |                4 |               44 |               6 |

## Next Step

The next step is to connect Qwen's hallucination results with the existing grounding pipeline and then calculate a same-model Clinical Trustworthiness Score.
