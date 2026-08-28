# Week 3 Model Comparison Summary

## What This Summary Shows

This summary connects the main model-evaluation results completed so far.

The project is no longer only checking model accuracy. It now compares hallucination detection, evidence grounding, trustworthiness scoring, and safety routing.

## Model Comparison

| model_or_method | dataset | rows | accuracy | f1_score | recall | main_result | project_decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF baseline + trust score | Baseline healthcare evaluation set | 4000 |  |  |  | Mean trustworthiness score 0.68 | Useful baseline reference |
| Qwen 0.5B | Med-HALT balanced sample | 100 | 0.52 | 0.2 | 0.12 | Too weak for main hallucination detection | Do not use as primary detector |
| Qwen 1.5B | Med-HALT balanced sample | 100 | 0.86 | 0.875 | 0.98 | Strong improvement over Qwen 0.5B | Keep validating with safety routing |
| Qwen 1.5B + logic-aware routing | Med-HALT balanced sample | 100 |  |  |  | Accepted hallucinated rows reduced to 0 | Use logic-aware routing for negative exam-style prompts |

## Main Interpretation

Qwen 1.5B is clearly stronger than Qwen 0.5B on the Med-HALT balanced sample. The 0.5B model missed too many hallucinated answers, while the 1.5B model detected nearly all hallucinations in the current test.

However, the most important finding today was not only the higher accuracy. One hallucinated answer was still being accepted because the question used negative wording like “NOT correct.” The answer looked medically supported, but it was wrong for the question logic.

After adding logic-aware routing, accepted hallucinated rows dropped from 1 to 0.

## Project Decision

The current best approach is:

1. Keep Qwen 1.5B as the stronger local LLM candidate.
2. Keep semantic grounding as the evidence-checking layer.
3. Add logic-aware routing for tricky medical exam prompts.
4. Continue validating before treating Qwen 1.5B as final.

## Next Step

The next major step is to update the final evaluation layer so Med-HALT hallucination detection, semantic grounding, EquityMedQA fairness screening, and Qwen 1.5B logic-aware routing are shown together in one project-level trustworthiness view.
