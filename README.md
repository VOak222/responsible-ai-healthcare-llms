# Responsible AI Healthcare LLMs

This project evaluates healthcare LLM responses across Responsible AI dimensions including hallucination detection, evidence grounding, bias and health equity, clinical safety risk, transparency, and human review recommendations.

## Responsible AI Dimensions

| Dimension | Measurement Direction |
|---|---|
| Accuracy | Hallucination rate, precision, recall, F1 score, false positives, and false negatives |
| Groundedness | Evidence support score and claim-level grounding checks |
| Safety | Low, medium, and high clinical risk labels |
| Fairness | Bias and health-equity assessment across patient groups |
| Transparency | Explanation, evidence notes, and model decision interpretation |
| Human Oversight | Accept, revise, human review, or mandatory human review recommendation |

## Current Baselines

The current baseline work includes:

| Model | Purpose |
|---|---|
| TF-IDF + Logistic Regression | Classical interpretable hallucination detection baseline |
| Sentence Transformer + Logistic Regression | Semantic embedding baseline for comparison |
| Evidence overlap diagnostic | Initial groundedness diagnostic |

## Benchmark Plan

Planned evaluator types include:

| Evaluator Type | Purpose |
|---|---|
| TF-IDF + Logistic Regression | Classical interpretable baseline |
| Sentence Transformer + Logistic Regression | Semantic embedding baseline |
| Classical Transformer | Stronger supervised text classifier if compute allows |
| Phi, Gemma, Qwen, or similar small LLMs | Local LLM evaluator candidates |

## Final Deliverables

Planned final outputs are:

1. Technical report.
2. Evaluation scripts and/or notebooks.
3. Demo or dashboard.
4. Manager presentation.