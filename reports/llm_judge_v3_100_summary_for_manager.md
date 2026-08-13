# LLM Judge v3 Larger Sample Summary

## Purpose

This experiment expanded the direct LLM judge from a 20-row sample to a larger balanced 100-row sample, with 50 hallucinated answers and 50 ground-truth answers.

## Model Used

Qwen/Qwen2.5-0.5B-Instruct was used as a small open-source local LLM judge.

## Method

The LLM was asked to judge whether each healthcare answer was supported, partial, or unsupported based only on the provided medical evidence. For healthcare safety, partial and unsupported outputs were mapped to hallucination risk.

## Results

| method | rows | accuracy | precision | recall | f1_score | true_positive | true_negative | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF baseline | 100 | 0.78 | 0.818 | 0.72 | 0.766 | 36 | 42 | 8 | 14 |
| Qwen 0.5B LLM judge v3 | 99 | 0.566 | 0.556 | 0.612 | 0.583 | 30 | 26 | 24 | 19 |

## Support Level Counts

| support_level | count |
| --- | --- |
| partial | 54 |
| supported | 45 |
| unknown | 1 |

## Interpretation

The 0.5B LLM judge worked successfully on the larger sample, producing valid judgments for 99 out of 100 rows. However, its accuracy on the larger sample was 0.566, compared with 0.780 for the TF-IDF baseline on the same 100 rows. This shows that the small LLM can provide useful evidence-support reasoning, but it is not reliable enough to replace the baseline model as the main hallucination detector.

## Key Takeaway

The direct LLM experiment answers the question of whether we have tested LLMs directly. We have now tested one locally. The result suggests that a 0.5B model is too small for dependable healthcare hallucination detection, so the next step is to test a stronger local model such as Qwen2.5-1.5B-Instruct or use the LLM judge as a supporting explanation layer combined with semantic grounding and trustworthiness scoring.
