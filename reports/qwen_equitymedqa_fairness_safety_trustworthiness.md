# Qwen EquityMedQA Fairness-Safety Trustworthiness

## Evaluation

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: EquityMedQA
- Rows reviewed: 33
- Composite-score eligible rows: 26
- Excluded non-comparable pairs: 7

## Formula

Clinical Safety Trust:

- 80% manual clinical-safety label
- 15% professional-care guidance
- 5% response completeness

EquityMedQA Trustworthiness:

- 60% Clinical Safety Trust
- 40% Fairness Trust

## Results

| model_id                   | evaluation_dataset   |   rows_reviewed |   composite_score_eligible_rows |   excluded_not_comparable_rows |   mean_clinical_safety_trust_score |   mean_fairness_trust_score |   mean_equitymedqa_trustworthiness_score |   mandatory_human_review_rows |   human_review_rows |   accept_rows |
|:---------------------------|:---------------------|----------------:|--------------------------------:|-------------------------------:|-----------------------------------:|----------------------------:|-----------------------------------------:|------------------------------:|--------------------:|--------------:|
| Qwen/Qwen2.5-0.5B-Instruct | EquityMedQA          |              33 |                              26 |                              7 |                           0.359091 |                    0.653846 |                                 0.525769 |                            18 |                  12 |             3 |

## Recommended Actions

| recommended_action     |   row_count |
|:-----------------------|------------:|
| mandatory_human_review |          18 |
| human_review           |          12 |
| accept                 |           3 |

## Limitation

This score evaluates Qwen on EquityMedQA. It is kept separate from the TF-IDF Clinical Trustworthiness Score because the models and evaluation rows are different.
