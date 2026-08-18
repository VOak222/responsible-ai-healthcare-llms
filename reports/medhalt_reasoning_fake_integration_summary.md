# Med-HALT reasoning_fake Integration Result

## Dataset Integrated

We integrated the Med-HALT `reasoning_fake` subset into the project pipeline.

Original questions: 1,858  
Converted evaluation rows: 3,712  

The converted dataset has two answer types:
- fake answer option = hallucinated
- safe refusal / I do not know = not hallucinated

## Baseline Result

| dataset                |   train_rows |   test_rows |   accuracy |   precision |   recall |   f1_score |
|:-----------------------|-------------:|------------:|-----------:|------------:|---------:|-----------:|
| medhalt_reasoning_fake |         2970 |         742 |   0.979784 |    0.986376 | 0.973118 |   0.979702 |

## Ablation Check

| feature_set                    |   accuracy |   precision |   recall |   f1_score |
|:-------------------------------|-----------:|------------:|---------:|-----------:|
| full_question_knowledge_answer |   0.979784 |    0.986376 | 0.973118 |   0.979702 |
| answer_only                    |   0.995957 |    0.992    | 1        |   0.995984 |
| question_knowledge_only        |   0.498652 |    0        | 0        |   0        |

## Interpretation

The TF-IDF baseline achieved very high accuracy on this subset. However, the ablation test shows that the answer-only model performs even better than the full question + knowledge + answer input.

This means the model is mostly learning surface-level answer patterns. For example, `I do not know` is strongly associated with the safe class, while long absurd medical answer options are strongly associated with the hallucinated class.

So this result should not be presented as proof that the model deeply understands medical hallucination. It is better presented as a successful dataset-integration and sanity-check experiment.

## What We Learned

This experiment proves that the project can now process an additional healthcare hallucination benchmark beyond the original MedHallu dataset.

It also shows why evaluation needs ablation checks. A high accuracy score alone can be misleading if the model is using shortcuts.

## Next Step

The next dataset/subset should be harder and less shortcut-driven. The best next Med-HALT subset to test is `reasoning_FCT`, because it checks whether a model can evaluate whether a proposed answer is valid instead of only detecting fake wording.
