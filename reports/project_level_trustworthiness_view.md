# Project-Level Trustworthiness View

## Current Project Status

This view brings together the main safety and trustworthiness checks completed so far across Med-HALT and EquityMedQA.

The project now evaluates hallucination detection, logic-aware safety routing, manual fairness review, trustworthiness routing, and explainable fairness-risk scoring.

## Project-Level Summary

| evaluation_area | dataset_or_layer | rows | main_metric | safety_signal | current_decision |
| --- | --- | --- | --- | --- | --- |
| Med-HALT hallucination detection | Qwen 1.5B Med-HALT balanced sample | 100.0 | Accuracy 0.86, F1 0.875, Recall 0.98 | False negatives checked after model scoring | Qwen 1.5B is the strongest local LLM candidate so far |
| Logic-aware safety routing | Qwen 1.5B Med-HALT routing | 100.0 | Accepted hallucinated rows after logic-aware routing: 0 | Negative exam-style prompts are routed to review | Keep logic-aware routing in the trust layer |
| EquityMedQA manual fairness review | Completed 33-row manual review | 33.0 | Fail 16, needs revision 14, pass 3 | Clinical validation required for 29 rows | Use as a fairness and clinical safety review layer |
| EquityMedQA trustworthiness routing | Qwen EquityMedQA trust scores | 33.0 | 18 mandatory human review, 12 human review, 3 accept | Most flagged fairness responses are not automatically accepted | Connect fairness review with trustworthiness routing |
| EquityMedQA manual-trust alignment | Completed EquityMedQA manual review + trust routing | 33.0 | Unsafe accepts: 0; under-escalated fails: 0 | All fail rows routed to mandatory review; accepted pass rows: 3 | Trustworthiness routing aligns with completed manual review outcomes |
| EquityMedQA explainable fairness risk | Manual review labels + explainable risk scoring | 33.0 | High 21, medium 9, low 3 | Fail rows marked high risk: 16; pass rows marked low risk: 3 | Use risk reasons to explain why cases are routed to review |
| Project-level governance | Combined current system | nan | Hallucination + grounding + fairness + routing now evaluated | Unsafe, uncertain, incomplete, or unfair responses are routed to review | Ready to keep improving the dashboard and broader validation layer |

## Latest Update

The newest update adds explainable fairness-risk scoring for the EquityMedQA manual review results.

This helps explain why certain responses are considered risky. The strongest signals were clinical validation needs, incomplete or truncated responses, unsafe clinical safety labels, missing professional care guidance, stereotype signals, and fairness concerns.

## Current Interpretation

Qwen 1.5B remains the strongest local LLM candidate tested so far on Med-HALT. The logic-aware routing layer successfully prevents tricky negative-question cases from being automatically accepted.

For EquityMedQA, the manual review and trustworthiness layers show that most risky fairness responses should go to human review instead of being accepted. The explainable fairness-risk layer now adds clearer reasoning behind those decisions.

## Next Step

The next step is to keep improving the reporting layer so the results from different datasets can be shown together clearly. The dashboard/reporting view is still in progress and can be used later when it is ready for presentation.
