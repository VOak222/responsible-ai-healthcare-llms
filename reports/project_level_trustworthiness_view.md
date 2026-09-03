# Project-Level Trustworthiness View

This view summarizes the current project-level safety and trustworthiness layers.

The latest update adds manual-trust alignment for EquityMedQA. This checks whether the completed manual review outcomes agree with the automated trustworthiness routing.

## Current Trustworthiness Layers

| evaluation_area                     | dataset_or_layer                                    |   rows | main_metric                                                  | safety_signal                                                           | current_decision                                                     |
|:------------------------------------|:----------------------------------------------------|-------:|:-------------------------------------------------------------|:------------------------------------------------------------------------|:---------------------------------------------------------------------|
| Med-HALT hallucination detection    | Qwen 1.5B Med-HALT balanced sample                  |    100 | Accuracy 0.86, F1 0.875, Recall 0.98                         | False negatives checked after model scoring                             | Qwen 1.5B is the strongest local LLM candidate so far                |
| Logic-aware safety routing          | Qwen 1.5B Med-HALT routing                          |    100 | Accepted hallucinated rows after logic-aware routing: 0      | Negative exam-style prompts are routed to review                        | Keep logic-aware routing in the trust layer                          |
| EquityMedQA manual fairness review  | Completed 33-row manual review                      |     33 | Fail 16, needs revision 14, pass 3                           | Clinical validation required for 29 rows                                | Use as a fairness and clinical safety review layer                   |
| EquityMedQA trustworthiness routing | Qwen EquityMedQA trust scores                       |     33 | 18 mandatory human review, 12 human review, 3 accept         | Most flagged fairness responses are not automatically accepted          | Connect fairness review with trustworthiness routing                 |
| Project-level governance            | Combined current system                             |    nan | Hallucination + grounding + fairness + routing now evaluated | Unsafe, uncertain, incomplete, or unfair responses are routed to review | Ready to keep improving the dashboard and broader validation layer   |
| EquityMedQA manual-trust alignment  | Completed EquityMedQA manual review + trust routing |     33 | Unsafe accepts: 0; under-escalated fails: 0                  | All fail rows routed to mandatory review; accepted pass rows: 3         | Trustworthiness routing aligns with completed manual review outcomes |

## Latest Interpretation

The EquityMedQA manual-trust alignment result is clean. All manually failed rows were routed to mandatory human review, all needs-revision rows were sent to review, and all passed rows were accepted. Most importantly, there were zero unsafe accepts.

This strengthens the project because the fairness review layer is now connected to trustworthiness routing instead of being only a separate manual checklist.
