# Controlled PubMedQA Model Selection and Safety Decision

## Purpose

This document records the final model-selection decision for the Responsible AI Healthcare LLMs project. The system is designed as a research-stage evidence-assistance and human-review framework. It is not an autonomous clinical diagnosis or treatment system.

## Controlled Evaluation Protocol

A controlled PubMedQA benchmark was created to compare models fairly.

- Source: official PubMedQA expert-labelled data
- Training set: 486 rows
  - 188 yes
  - 188 no
  - 110 maybe
- Validation set: 100 rows
  - 50 yes
  - 50 no
- Frozen untouched test set: 200 rows
  - 100 yes
  - 100 no

Both BioGPT-Large and Qwen2.5-3B were fine-tuned from their base checkpoints using LoRA. The test set was not used for training, validation-loss checkpoint selection, or confidence-threshold selection.

## Controlled Benchmark Results

| Model | Test Accuracy | Macro-F1 | Yes Recall | No Recall |
|---|---:|---:|---:|---:|
| Base BioGPT-Large | 52.5% | 0.531 | 57% | 48% |
| BioGPT-Large + Controlled PubMedQA LoRA | 75.5% | 0.761 | 65% | 86% |
| Base Qwen2.5-3B | 6.5% | 0.120 | 10% | 3% |
| Qwen2.5-3B + Controlled PubMedQA LoRA, Epoch 1 | 84.0% | 0.839 | 93% | 75% |

Qwen epoch 1 was selected before final testing because it had the lowest validation loss. Later epochs were not selected because validation loss increased.

## Model Selection Decision

Qwen2.5-3B with the controlled PubMedQA LoRA adapter is the primary model.

Reasons:

1. It achieved the highest controlled-test accuracy, 84.0%.
2. It achieved the highest macro-F1, 0.839, on the balanced yes/no test set.
3. It showed strong yes recall, 93%, while retaining useful no recall, 75%.
4. It fits the project constraint of using a small open-source model that can run locally.

BioGPT-Large is retained as an independent biomedical verifier.

Reasons:

1. It has biomedical pretraining and provides a second, independent model view.
2. Its no recall was higher than Qwen's, 86% versus 75%.
3. It can identify disagreement cases that require human review.

The models are not weight-averaged or presented as an unsupported ensemble. BioGPT is used as a verification and routing signal.

## Safety and Human-Review Findings

### Qwen Confidence Gate

A Qwen confidence threshold was selected using the validation set. The rule maximized validation coverage while allowing zero observed true-no to predicted-yes errors.

| Evaluation set | Auto-accepted | Coverage | Accepted accuracy | Unsafe no-to-yes accepts |
|---|---:|---:|---:|---:|
| Validation | 30 / 100 | 30.0% | 100.0% | 0 |
| Untouched test | 44 / 200 | 22.0% | 90.9% | 4 |

### Qwen and BioGPT Agreement Gate

A stricter rule required both Qwen and BioGPT to predict yes, in addition to the validation-selected Qwen confidence threshold.

| Evaluation set | Auto-accepted | Coverage | Accepted accuracy | Unsafe no-to-yes accepts |
|---|---:|---:|---:|---:|
| Validation | 30 / 100 | 30.0% | 100.0% | 0 |
| Untouched test | 47 / 200 | 23.5% | 93.6% | 3 |

The agreement gate improved accepted-answer accuracy and reduced unsafe false accepts from four to three. However, it did not eliminate unsafe errors.

## Final Safety Policy

The project does not allow either model to autonomously approve a clinical-facing yes decision.

The final workflow is:

1. Qwen produces a structured evidence-based decision and rationale.
2. BioGPT provides an independent verification signal.
3. Evidence-grounding, confidence, and model-agreement signals are recorded.
4. Any disagreement, low confidence, risky clinical content, or final clinical-facing decision is routed to a qualified human reviewer.
5. The human reviewer makes the final decision.

## Limitations

- The benchmark is based on PubMedQA evidence questions and is not clinical deployment validation.
- The test set contains 200 rows, which is useful for controlled comparison but too small to establish clinical safety.
- A validation threshold with zero observed unsafe errors still produced unsafe errors on the untouched test set.
- Model performance and routing quality must be evaluated further on additional datasets, clinical scenarios, demographic groups, and expert-reviewed cases.

## Final Conclusion

Qwen2.5-3B LoRA is selected as the primary local open-source model because it performed best on the controlled benchmark. BioGPT-Large LoRA is retained as an independent verifier. The system remains human-in-the-loop because the experiments show that confidence gating and model agreement reduce risk but do not eliminate unsafe decisions.