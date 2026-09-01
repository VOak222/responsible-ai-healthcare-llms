# Project-Level Trustworthiness View

## What This Shows

This report combines the main safety evaluation layers built so far.

The project now evaluates healthcare LLM outputs across hallucination detection, evidence grounding, fairness review, and human-review routing.

## Project-Level Summary

| evaluation_area | dataset_or_layer | rows | main_metric | safety_signal | current_decision |
| --- | --- | --- | --- | --- | --- |
| Med-HALT hallucination detection | Med-HALT balanced sample | 100 | Accuracy 0.86, F1 0.875, Recall 0.98 | False negatives: 1 | Qwen 1.5B is the stronger local LLM candidate |
| Logic-aware safety routing | Qwen 1.5B Med-HALT routing | 100 | Accepted hallucinated rows: 0 | Negative exam-style prompts are sent to review | Keep logic-aware routing in the trust layer |
| EquityMedQA fairness responses | Full EquityMedQA Qwen run | 60 | Review rows: 60 | Manual review shortlist rows: 33 | Use as fairness and bias review layer |
| EquityMedQA trustworthiness routing | Qwen EquityMedQA trust scores | 33.0 | Mean trustworthiness score: 0.359 | 3 summary rows available | Connect fairness review with trustworthiness routing |
| Project-level governance | Combined current system |  | Hallucination + grounding + fairness + routing now evaluated | Unsafe or uncertain cases are routed to human review | Ready to prepare dashboard/reporting layer next |

## EquityMedQA Review Summary

| evaluation_type | overall_fairness_safety_risk | rows |
| --- | --- | --- |
| paired_prompt_comparison | low_observed_risk | 8 |
| paired_prompt_comparison | review_needed | 12 |
| single_prompt_fairness | low_observed_risk | 19 |
| single_prompt_fairness | review_needed | 21 |

## Interpretation

The strongest current model result is Qwen 1.5B on the Med-HALT balanced sample. It performed much better than Qwen 0.5B, but the important safety improvement was the logic-aware routing layer.

The logic-aware rule reduced accepted hallucinated rows to 0 by catching negative medical question patterns such as NOT correct, incorrect, except, and least likely.

EquityMedQA adds the fairness and bias-assessment side of the project. This means the framework is no longer only checking whether an answer is hallucinated. It is also checking whether model behavior changes across sensitive or demographic prompt variations.

## Current Project Position

At this point, the project has a usable responsible-AI evaluation structure:

1. Hallucination detection using baseline and local LLM evaluation.
2. Evidence grounding using semantic similarity and unsupported claim checks.
3. Clinical trustworthiness scoring.
4. Logic-aware safety routing for tricky medical prompts.
5. EquityMedQA fairness and bias review.

## Recommended Next Step

The next step is to build a simple dashboard/reporting view so these results can be shown clearly in a demo.
