# Week 5 Future Work: Model and Dataset Expansion Plan

## Purpose

This note prepares the next phase of the project without changing this week's report or PPT.

The goal is to make the project stronger by adding one larger local model, one additional medical reasoning dataset, and a final governance/reporting layer.

## Environment Check

| Check | Value |
|---|---|
| Python version | 3.14.0 |
| Platform | Windows-11-10.0.26200-SP0 |
| Free disk space | 140.19 GB |
| PyTorch available | True |
| CUDA available | False |
| GPU | not available |

## Package Check

| Package | Status |
|---|---|
| torch | available |
| transformers | available |
| accelerate | available |
| datasets | available |
| pandas | available |
| sentence_transformers | available |

## Planned Additions

| Addition | Why It Matters |
|---|---|
| Qwen2.5-3B-Instruct | Adds a stronger local model comparison beyond Qwen 0.5B and Qwen 1.5B. |
| Phi-3.5-mini-instruct | Gives a backup small-model option if Qwen2.5-3B is too heavy locally. |
| MedMCQA | Adds a broader medical reasoning dataset for multiple-choice healthcare evaluation. |
| Governance/reporting layer | Makes the final project look like a professional evaluation framework, not only model testing. |

## Implementation Approach

The next step should be a small smoke test, not a full run.

First, test Qwen2.5-3B on 5 Med-HALT rows. If it runs properly, expand to 25 rows, then 100 rows.

After that, add a small MedMCQA sample and reuse the existing trustworthiness and routing structure.

## Current Decision

These additions should be treated as Week 5 future work. They should not be added to this week's PPT or report yet, but the groundwork is now prepared.
