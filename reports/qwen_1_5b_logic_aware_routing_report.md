# Qwen 1.5B Logic-Aware Routing Report

## Why This Was Needed

The first threshold tuning still accepted one hallucinated row.

That row was a tricky medical question because it asked which statement was NOT correct. The model selected an answer that looked medically supported, but it was wrong for the actual question logic.

This means the issue was not only hallucination probability or grounding score. It was a question-logic issue.

## Before vs After

| action_column | accepted_rows | accepted_supported_rows | accepted_hallucinated_rows | mandatory_human_review_rows | human_review_rows | revise_rows | logic_risk_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recommended_action | 25 | 24 | 1 | 68 | 6 | 1 | 4 |
| tuned_recommended_action | 25 | 24 | 1 | 68 | 6 | 1 | 4 |
| logic_aware_recommended_action | 24 | 24 | 0 | 75 | 1 | 0 | 4 |

## Logic-Aware Action Breakdown

| is_hallucinated | logic_aware_recommended_action | rows |
| --- | --- | --- |
| 0 | accept | 24 |
| 0 | mandatory_human_review | 26 |
| 1 | human_review | 1 |
| 1 | mandatory_human_review | 49 |

## Interpretation

The logic-aware routing rule catches exam-style medical prompts where words like NOT correct, incorrect, false, except, or least likely can change the meaning of the task.

This rule does not use the true label. It only uses the question text and model routing signals, so it is closer to what the final evaluation tool could do in practice.

## Project Decision

For healthcare evaluation, tricky negative questions should not be automatically accepted even when the answer seems grounded. These cases should go to human review because the answer may be factually true but still wrong for the question.
