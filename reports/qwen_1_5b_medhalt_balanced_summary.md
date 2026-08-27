# Qwen 1.5B Med-HALT Balanced Evaluation

## Evaluation

Qwen/Qwen2.5-1.5B-Instruct was evaluated on the same balanced 100-row Med-HALT sample previously used for the 0.5B model.

The selected prompt was the zero-shot numerical prompt because it produced the strongest pilot accuracy, F1 score, and ROC-AUC.

## Results

| prompt_variant | dataset_name | rows | accuracy | precision | recall | f1_score | roc_auc | predicted_hallucination_rate | mean_hallucination_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_shot_numeric | combined | 100 | 0.8600 | 0.7903 | 0.9800 | 0.8750 | 0.9728 | 0.6200 | 0.6039 |
| zero_shot_numeric | medhalt_reasoning_fake | 50 | 0.7400 | 0.6579 | 1.0000 | 0.7937 | 0.9760 | 0.7600 | 0.7353 |
| zero_shot_numeric | medhalt_reasoning_fct | 50 | 0.9800 | 1.0000 | 0.9600 | 0.9796 | 0.9920 | 0.4800 | 0.4724 |

## Combined Result

- Rows evaluated: 100
- Accuracy: 0.8600
- Precision: 0.7903
- Recall: 0.9800
- F1 score: 0.8750
- ROC-AUC: 0.9728
- Predicted hallucination rate: 0.6200

## Interpretation

This 100-row evaluation tests whether the strong 20-row pilot result generalizes to a larger balanced sample. The results should be compared directly with the earlier Qwen 0.5B evaluation, which achieved 0.52 accuracy, 0.20 F1, and 0.5096 ROC-AUC.
