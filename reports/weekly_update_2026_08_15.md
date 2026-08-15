# Weekly Progress Update - Responsible AI Healthcare LLMs

## Week Ending

August 15, 2026

## Summary

This week, I extended the healthcare hallucination evaluation pipeline beyond the initial TF-IDF baseline. The work focused on semantic claim verification, updated clinical trustworthiness scoring, direct local LLM judging, and preparing the next phase for multi-dataset evaluation.

## Completed Work

1. Verified and consolidated the existing baseline pipeline outputs.
2. Added semantic claim verification using sentence-transformers to move beyond simple word matching.
3. Updated the clinical trustworthiness score to use semantic grounding.
4. Created model comparison outputs across baseline, grounding, semantic verification, and trustworthiness scoring.
5. Generated TP/TN/FP/FN review examples for manual inspection.
6. Added a direct local LLM judge using Qwen/Qwen2.5-0.5B-Instruct.
7. Scaled the LLM judge test from 20 examples to a balanced 100-row sample.
8. Created manager-ready reports for LLM judge experiments.
9. Prepared the dataset integration plan for adding MedHallu and other healthcare datasets.

## Key Results

- Semantic verification improved evidence-support scoring compared with earlier TF-IDF fallback.
- Updated trustworthiness scoring reduced unnecessary mandatory human-review routing compared with the older score.
- Direct LLM judging was successfully tested locally using a small open-source model.
- On a balanced 100-row test, the TF-IDF baseline outperformed the 0.5B LLM judge, showing that the small LLM is useful for experimentation but not reliable enough as the only hallucination detector.

## LLM Judge v3 Metrics

| method | rows | accuracy | precision | recall | f1_score | true_positive | true_negative | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF baseline | 100 | 0.78 | 0.818 | 0.72 | 0.766 | 36 | 42 | 8 | 14 |
| Qwen 0.5B LLM judge v3 | 99 | 0.566 | 0.556 | 0.612 | 0.583 | 30 | 26 | 24 | 19 |

## LLM Support-Level Counts

| support_level | count |
| --- | --- |
| partial | 54 |
| supported | 45 |
| unknown | 1 |

## Interpretation

The direct LLM experiment answers the question of why we had not used LLMs directly yet. We have now tested a direct local LLM judge. The result shows that a very small 0.5B LLM can provide useful evidence-support reasoning, but it does not outperform the current baseline on a larger sample. This supports a responsible AI approach where the LLM is used as a supporting evaluator rather than the only decision-maker.

## Limitations

- The current pipeline has primarily been tested on one main healthcare QA-style dataset.
- The direct LLM test used a small local model due to hardware constraints.
- Larger or more medically specialized models may perform better but need local feasibility testing.
- The next phase should test whether the pipeline generalizes across additional healthcare hallucination datasets.

## Next Week Plan

1. Integrate MedHallu as the next dataset.
2. Convert external datasets into a common schema.
3. Run baseline, grounding, semantic verification, and trustworthiness scoring per dataset.
4. Compare results by dataset instead of only reporting one overall score.
5. If hardware allows, test a stronger local LLM such as a 1.5B model.

