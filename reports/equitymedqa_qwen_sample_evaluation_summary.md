# EquityMedQA Qwen Sample Evaluation Summary

## What Was Tested

A small Qwen/Qwen2.5-0.5B-Instruct test was run on selected EquityMedQA prompts.

The sample included:
- 5 single fairness prompts
- 3 paired demographic comparison prompts
- 8 total evaluated rows

## Why This Was Done

The goal was not to run the full EquityMedQA dataset yet. The goal was to confirm that the pipeline can:
- load selected fairness prompts,
- generate local Qwen responses,
- save model outputs,
- prepare outputs for fairness and safety review.

## Initial Result

The model generated responses for all 8 prompts successfully.

Risk summary:

| overall_fairness_risk   |   rows |
|:------------------------|-------:|
| low_observed_risk       |      4 |
| review_needed           |      4 |

## Early Observation

The model generally uses respectful language, but the clinical quality is mixed. Some answers are too generic, and some high-risk medical prompts need clearer caution to contact a healthcare professional.

This means EquityMedQA can now be used as the fairness and health-equity evaluation layer, but model responses still need structured review before we trust them.

## Next Step

Next week, the plan should be:
1. Run Qwen on the full 60-row EquityMedQA evaluation template.
2. Improve the review rubric for fairness, unsafe clinical advice, and demographic inconsistency.
3. Add these fairness signals into the trustworthiness score.
