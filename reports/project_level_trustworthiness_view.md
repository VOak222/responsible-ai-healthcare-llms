# Project-Level Trustworthiness View

## What This View Shows

This file summarizes the current Responsible AI evaluation framework at the project level.

The goal is to show how the different safety layers now work together: hallucination detection, grounding, fairness review, trustworthiness scoring, manual review, and dashboard reporting.

## Project-Level Summary

| evaluation_area | dataset_or_layer | rows | main_metric | safety_signal | current_decision |
| --- | --- | --- | --- | --- | --- |
| Med-HALT hallucination detection | Qwen 1.5B Med-HALT balanced sample | 100 | Accuracy 0.86, F1 0.875, Recall 0.98 | False negatives checked after model scoring | Qwen 1.5B is the strongest local LLM candidate so far |
| Logic-aware safety routing | Qwen 1.5B Med-HALT routing | 100 | Accepted hallucinated rows after logic-aware routing: 0 | Negative exam-style prompts are routed to review | Keep logic-aware routing in the trust layer |
| EquityMedQA manual fairness review | Completed 33-row manual review | 33 | Fail 16, needs revision 14, pass 3 | Clinical validation required for 29 rows | Use as a fairness and clinical safety review layer |
| EquityMedQA trustworthiness routing | Qwen EquityMedQA trust scores | 33 | 18 mandatory human review, 12 human review, 3 accept | Most flagged fairness responses are not automatically accepted | Connect fairness review with trustworthiness routing |
| Project-level governance | Combined current system |  | Hallucination + grounding + fairness + routing now evaluated | Unsafe, uncertain, incomplete, or unfair responses are routed to review | Ready to keep improving the dashboard and broader validation layer |

## EquityMedQA Manual Review Outcome

| manual_review_outcome | rows |
| --- | --- |
| fail | 16 |
| needs_revision | 14 |
| pass | 3 |

The EquityMedQA manual review is now more complete. Out of 33 shortlisted fairness responses, 16 failed, 14 needed revision, and 3 passed.

This shows that many responses can sound reasonable but still need human review because they may be unsafe, incomplete, unfair, stereotyped, or missing proper care guidance.

## EquityMedQA Trustworthiness Routing

| recommended_action | rows |
| --- | --- |
| mandatory_human_review | 18 |
| human_review | 12 |
| accept | 3 |

The routing result supports the manual review finding. Only 3 responses were accepted directly, while most were sent to human review or mandatory human review.

## Latest Interpretation

Qwen 1.5B is currently the strongest local model tested in this project. It performed much better than Qwen 0.5B on the Med-HALT balanced sample.

The most important safety improvement is logic-aware routing. It reduced accepted hallucinated rows from 1 to 0 by catching tricky negative medical exam-style prompts such as NOT correct, incorrect, except, and least likely.

The fairness layer is also stronger now because the EquityMedQA shortlist has been converted into a completed manual review summary. This gives the project a clearer way to explain which responses passed, which need revision, and which failed.

## Current Project Decision

The project should continue with this structure:

1. Use Qwen 1.5B as the strongest local LLM candidate so far.
2. Keep semantic grounding as the evidence-checking layer.
3. Keep logic-aware routing for tricky medical prompts.
4. Keep EquityMedQA as the fairness and health-equity review layer.
5. Use the dashboard and reports to explain the results clearly.
