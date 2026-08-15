# Dataset Integration Plan

## Purpose

The current pipeline works on the initial healthcare QA/hallucination dataset, but the final project should not rely on only one dataset. The next phase is to integrate additional healthcare hallucination datasets and evaluate whether the pipeline generalizes across different data sources.

## Standard Schema

All datasets should be converted into the same format:

| column | meaning |
| --- | --- |
| record_id | Unique row identifier |
| dataset_name | Source dataset name |
| question | Healthcare question or prompt |
| evidence | Medical evidence, context, or reference answer |
| answer | Model answer or candidate answer to evaluate |
| is_hallucinated | 1 for hallucinated/unsupported, 0 for grounded/safe |
| source_label | Original label from the source dataset |

## Dataset Priority

1. MedHallu: highest priority because it directly focuses on medical hallucination.
2. PubMedQA: keep as the current primary evidence-based QA dataset.
3. HealthSearchQA: useful for consumer-style health questions.
4. Med-HALT / K-QA / EquityMedQA: secondary datasets for safety, reasoning, and fairness evaluation if time allows.

## Evaluation Plan

For each dataset, run:

1. TF-IDF baseline hallucination classifier.
2. Claim-level grounding.
3. Semantic claim verification.
4. Clinical trustworthiness scoring.
5. Optional direct LLM judge sample.

## Reporting Plan

Final results should include both per-dataset and overall metrics:

- PubMedQA performance
- MedHallu performance
- HealthSearchQA or secondary dataset performance
- Overall combined performance

## Key Reason

Using multiple datasets makes the project stronger because it shows whether the responsible AI pipeline works beyond one dataset format. This improves credibility for the final report and presentation.
