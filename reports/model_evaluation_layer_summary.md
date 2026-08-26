# Model Evaluation Layer Summary

## Results

| model_id                   | evaluation_layer            | evaluation_dataset          |   rows_evaluated |   score_eligible_rows | primary_score_name                |   primary_score |   mean_hallucination_risk |   mean_grounding_score |   mean_clinical_safety_trust |   mean_fairness_trust |   mandatory_human_review_rows |
|:---------------------------|:----------------------------|:----------------------------|-----------------:|----------------------:|:----------------------------------|----------------:|--------------------------:|-----------------------:|-----------------------------:|----------------------:|------------------------------:|
| TF-IDF Logistic Regression | hallucination_and_grounding | baseline_evaluation_dataset |             4000 |                  4000 | clinical_trustworthiness_score    |        0.679547 |                  0.494549 |                0.72502 |                   nan        |            nan        |                            98 |
| Qwen/Qwen2.5-0.5B-Instruct | fairness_and_safety         | EquityMedQA                 |               33 |                    26 | equitymedqa_trustworthiness_score |        0.525769 |                nan        |              nan       |                     0.359091 |              0.653846 |                            18 |

## Interpretation

The TF-IDF baseline currently has hallucination and grounding results.

Qwen currently has fairness and clinical-safety results from EquityMedQA.

These scores are presented in separate evaluation layers and are not directly averaged because they come from different models and datasets.

The next step toward a unified model score is to run Qwen on the hallucination and grounding evaluation data.
