# Direct LLM Judge Experiment Summary

## Purpose

This experiment was added to directly test an open-source LLM instead of relying only on TF-IDF and classical ML. The model used was Qwen/Qwen2.5-0.5B-Instruct, running locally.

## Approach

The LLM was used as a healthcare evidence judge. Instead of asking only for a hallucination label, the model judged whether the answer was supported, partial, or unsupported based on the provided medical evidence. Then healthcare safety logic was applied: supported = not hallucinated, partial or unsupported = hallucination risk.

## Metrics

| method | rows | accuracy | precision | recall | f1_score | true_positive | true_negative | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF baseline | 20 | 0.5 | 0.5 | 0.5 | 0.5 | 5 | 5 | 5 | 5 |
| Qwen 0.5B LLM judge v2 | 20 | 0.6 | 0.667 | 0.4 | 0.5 | 4 | 8 | 2 | 6 |

## Support Level Counts

| support_level | count |
| --- | --- |
| supported | 14 |
| partial | 6 |

## Key Takeaway

The direct LLM judge improved sample accuracy from 50% to 60% compared with the TF-IDF baseline on the same 20 reviewed examples. However, it still missed 6 hallucinated answers, which shows that the 0.5B model is useful for experimentation but not reliable enough as the only evaluator. The best direction is to combine LLM judging with semantic grounding and clinical trustworthiness scoring.
