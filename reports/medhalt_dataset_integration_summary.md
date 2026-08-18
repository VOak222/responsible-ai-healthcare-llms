# Med-HALT Dataset Integration Summary

## Work Completed

Two Med-HALT subsets were integrated into the project pipeline:

1. `reasoning_fake`
2. `reasoning_FCT`

This expands the project beyond the original MedHallu dataset and proves that the pipeline can process additional healthcare hallucination benchmarks.

## Dataset Conversion

### reasoning_fake

The `reasoning_fake` subset contains intentionally fake or nonsensical medical questions. We converted it into a binary hallucination dataset by treating fake answer options as hallucinated and safe refusal answers such as `I do not know` as not hallucinated.

Converted rows: 3,712

### reasoning_FCT

The `reasoning_FCT` subset contains medical multiple-choice questions with a correct answer and a proposed student answer. We converted each item into paired rows:

- correct answer = not hallucinated
- incorrect student answer = hallucinated

Converted rows: 37,638

## Baseline Metrics

### reasoning_fake

| dataset                |   train_rows |   test_rows |   accuracy |   precision |   recall |   f1_score |
|:-----------------------|-------------:|------------:|-----------:|------------:|---------:|-----------:|
| medhalt_reasoning_fake |         2970 |         742 |   0.979784 |    0.986376 | 0.973118 |   0.979702 |

### reasoning_FCT

| dataset               |   train_rows |   test_rows |   accuracy |   precision |   recall |   f1_score |
|:----------------------|-------------:|------------:|-----------:|------------:|---------:|-----------:|
| medhalt_reasoning_fct |        30101 |        7537 |   0.556986 |    0.543969 | 0.698459 |   0.611609 |

## Ablation Checks

### reasoning_fake

| feature_set                    |   accuracy |   precision |   recall |   f1_score |
|:-------------------------------|-----------:|------------:|---------:|-----------:|
| full_question_knowledge_answer |   0.979784 |    0.986376 | 0.973118 |   0.979702 |
| answer_only                    |   0.995957 |    0.992    | 1        |   0.995984 |
| question_knowledge_only        |   0.498652 |    0        | 0        |   0        |

### reasoning_FCT

| feature_set                    |   accuracy |   precision |   recall |   f1_score |
|:-------------------------------|-----------:|------------:|---------:|-----------:|
| full_question_knowledge_answer |   0.556986 |    0.543969 | 0.698459 |   0.611609 |
| answer_only                    |   0.511875 |    0.510429 | 0.552604 |   0.53068  |
| question_knowledge_only        |   0.500597 |    0.5      | 0.479809 |   0.489696 |

## Interpretation

The `reasoning_fake` result looks very strong, but the ablation check shows that the answer-only model performs even better than the full input. This means the TF-IDF model mostly detects surface patterns, especially safe responses like `I do not know` versus long absurd medical answers.

The `reasoning_FCT` result is more realistic. The full model performs better than answer-only and question-knowledge-only versions, but the accuracy is still only about 55.7%. This shows that TF-IDF is not strong enough for deeper medical answer correctness checking.

## Main Takeaway

The project has now successfully moved beyond one dataset. Med-HALT has been integrated, baseline results have been generated, and ablation checks were used to avoid misleading conclusions.

The next technical step is to run semantic similarity or a small LLM judge on `reasoning_FCT`, because this subset better represents the harder hallucination-detection problem.
