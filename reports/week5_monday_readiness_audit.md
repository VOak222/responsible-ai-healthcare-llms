# Week 5 Monday Readiness Audit

## Purpose

This note prepares the project for Monday's report and PPT update work.

The main goal is to make sure the latest technical results are already organized before presentation preparation starts.

## Readiness Summary

| workstream | status | result | monday_use |
| --- | --- | --- | --- |
| Med-HALT Qwen 1.5B evaluation | complete | Accuracy 0.86, F1 0.875, Recall 0.98, FN 1 | Use in PPT/report as strongest local LLM result so far |
| Logic-aware safety routing | complete | Accepted hallucinated rows after routing: 0 | Use as the main safety improvement example |
| EquityMedQA manual review | complete | Fail 16, needs revision 14, pass 3 | Use to show human review was completed for flagged fairness cases |
| EquityMedQA trust routing | complete | 18 mandatory review, 12 human review, 3 accept | Use to show risky responses were not automatically accepted |
| Manual-trust alignment | complete | Unsafe accepts 0, under-escalated fails 0, accepted pass cases 3 | Use to prove routing matches manual review outcomes |
| Explainable fairness-risk scoring | complete | High 21, medium 9, low 3 | Use to explain why cases were routed for review |
| Dashboard reporting | ready for meeting review | Final dashboard updated with Week 4 metrics | Open and decide whether to show it during Tuesday meeting |

## Main Project Story For Next Week

The project has moved from simple model evaluation into a broader trustworthiness workflow.

The current evaluation now covers hallucination detection, logic-aware routing, manual fairness review, trustworthiness routing, manual-review alignment, and explainable risk scoring.

## Strongest Results To Present

| Area | Result |
|---|---|
| Qwen 1.5B Med-HALT performance | Accuracy 0.86, F1 0.875, Recall 0.98, FN 1 |
| Logic-aware routing | Accepted hallucinated rows after routing: 0 |
| EquityMedQA manual review | Fail 16, needs revision 14, pass 3 |
| Manual-trust alignment | Unsafe accepts 0, under-escalated fails 0 |
| Explainable fairness risk | High 21, medium 9, low 3 |

## Explainable Risk Reasons

The top risk reasons are:

clinical validation required, response incomplete or truncated, manual overall risk is high, manual review outcome failed, unsafe clinical safety label

These reasons help explain why certain healthcare responses should go to review instead of being accepted automatically.

## Monday Work Plan

1. Update the PPT with the latest Week 4 and Week 5 readiness results.
2. Update the written report with the manual-trust alignment and explainable risk findings.
3. Decide whether to show the dashboard in the Tuesday meeting.
4. Keep the dashboard as supporting material if the meeting is short.
5. Start the next technical phase after the meeting: broader validation across another dataset or a larger EquityMedQA run.

## Current Position

The project is ready for Monday presentation preparation. The remaining work is mainly communication, formatting, and deciding which results to highlight for Tuesday.
