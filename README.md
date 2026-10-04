# Responsible AI Evaluation Framework for Healthcare LLMs

This project implements a local Responsible AI evaluation framework for assessing externally generated healthcare LLM answers against controlled biomedical evidence.

The system does not generate medical advice and is not intended for clinical use. It evaluates candidate answers produced by external models and routes uncertain or conflicting cases for human review.

---

## Final System Architecture

The final prototype combines three independent evaluation signals:

1. Evidence retrieval
   - Sentence-Transformers (`all-MiniLM-L6-v2`)
   - FAISS semantic search
   - Controlled local PubMedQA train + validation corpus
   - Retrieval uses only the research question so the candidate answer cannot influence its own evidence selection.

2. PubMedQA evidence-conclusion evaluator
   - Base model: `Qwen/Qwen2.5-3B-Instruct`
   - LoRA adapter trained on the controlled PubMedQA task
   - Input: question + evidence
   - Output: `yes`, `no`, or `maybe`

3. Med-HALT hallucination evaluator
   - Base model: `Qwen/Qwen2.5-3B-Instruct`
   - Separate Med-HALT LoRA adapter
   - Input: question + evidence + candidate answer
   - Output: `supported` or `hallucinated`

A transparent rule-based consistency layer compares the candidate's explicit yes/no/maybe conclusion with the PubMedQA evidence conclusion.

The final routing layer combines retrieval strength, both model signals, label separation, and candidate-evidence consistency.

Possible dispositions include:

- `Benchmark pass — evidence aligned`
- `Benchmark fail — evidence conflict`
- `Human review required`
- `Routine human review — no escalation`
- `Mandatory human review`
- `Insufficient local evidence — human review required`

A benchmark pass is only a result within the controlled local research workflow. It is not clinical approval or a probability of medical correctness.

---

## Responsible AI Design Principles

The prototype focuses on:

- Evidence grounding
- Hallucination detection
- Independent evaluator signals
- Conservative uncertainty handling
- Transparent routing rules
- Human oversight
- Separation between answer generation and answer evaluation
- Clear distinction between research benchmark results and clinical validity

The candidate answer is treated as untrusted input.

---

## Key Evaluation Results

### PubMedQA Controlled Benchmark

Selected Qwen2.5-3B LoRA:

- Frozen test accuracy: 84.0%
- Macro-F1: 0.839

A separate 100-row diagnostic subset produced:

- Base Qwen: 3.0% accuracy
- PubMedQA LoRA: 82.0% accuracy

The base-model result reflects failure on this fixed-label task, particularly over-prediction of `maybe`; it should not be interpreted as general medical knowledge accuracy.

### Med-HALT

Selected Qwen2.5-3B LoRA:

Med-HALT FCT:
- Accuracy: 100%
- Macro-F1: 1.000

Unseen Reasoning Fake stress test:
- Accuracy: 99.1%
- Macro-F1: 0.991

These are task-specific benchmark results and are not clinical validation.

### Natural-Language Cross-Dataset Stress Test

A human-audited PubMedQA-derived counterfactual test containing 100 natural-language candidate answers was also evaluated:

Base Qwen:
- Accuracy: 63%
- Macro-F1: 0.578

Med-HALT LoRA:
- Accuracy: 86%
- Macro-F1: 0.860

This experiment demonstrates improved cross-format behavior while also showing why the system uses multiple independent signals rather than relying on one hallucination classifier alone.

---

# Running the Final Demo

## Tested Environment

The final package was reproduced successfully using:

- Windows 11
- Python 3.14.0
- NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch 2.11.0 + CUDA 12.8

The live dual-adapter evaluator requires an NVIDIA CUDA-capable GPU.

Internet access is required the first time the project runs so Hugging Face can download:

- `Qwen/Qwen2.5-3B-Instruct`
- `sentence-transformers/all-MiniLM-L6-v2`

The selected LoRA adapters and local FAISS evidence index are included in the final project package.

---

## 1. Create a Virtual Environment

From the project root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 2. Install PyTorch with CUDA 12.8

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The tested system returns:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4070 Laptop GPU
```

---

## 3. Install Project Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 4. Start the Demo

From the project root:

```powershell
python -m src.demo.app
```

Open:

```text
http://127.0.0.1:8000
```

The first model evaluation may take longer because the Qwen base model and Sentence-Transformer model may need to download from Hugging Face.

---

## Example Demo Case

Research question:

```text
Does hypoglycaemia increase the risk of cardiovascular events?
```

Candidate answer:

```text
Yes. In people with dysglycaemia and high cardiovascular risk, severe hypoglycaemia was associated with an increased risk of cardiovascular outcomes, including cardiovascular death and arrhythmic death.
```

With the included local PubMedQA evidence index, this controlled example produces:

```text
Benchmark pass — evidence aligned
```

This is a research-benchmark disposition only and is not clinical approval.

---

## Evidence Retrieval

The bundled evidence index contains PubMedQA training and validation evidence only.

Frozen benchmark test records are intentionally excluded from retrieval.

Automatic evidence retrieval:

- embeds only the user research question
- uses normalized Sentence-Transformer embeddings
- searches a FAISS inner-product index
- rejects weak nearest neighbors below the relevance threshold

If the local evidence corpus does not contain sufficiently relevant evidence, the system stops automatic evaluation and routes the case for human review.

The user can also provide evidence manually. Manually supplied evidence is explicitly labeled unverified and can never receive an automatic controlled-benchmark pass.

---

## Rebuilding the Evidence Index

If the evidence index needs to be rebuilt:

```powershell
python scripts/build_pubmedqa_evidence_index.py
```

Required source files:

```text
data/processed/controlled_benchmark_train.jsonl
data/processed/controlled_benchmark_validation.jsonl
```

---

## Required Final Runtime Assets

The local demo requires:

```text
artifacts/
├── adapters/
│   └── qwen2_5_3b_medhalt_hallucination_best_validation/
├── checkpoints/
│   └── qwen2_5_3b_controlled_benchmark/
│       └── epoch_1/
└── evidence_index/
    ├── pubmedqa_train_validation.faiss
    ├── pubmedqa_train_validation_metadata.jsonl
    └── pubmedqa_train_validation_manifest.json

data/
└── processed/
    ├── controlled_benchmark_train.jsonl
    └── controlled_benchmark_validation.jsonl
```

The full Qwen2.5-3B base model is not bundled because it is downloaded from Hugging Face when needed.

---

## Repository Scope

The repository also contains historical training, baseline, fairness, error-analysis, and evaluation scripts developed during the project.

Some historical experiments require datasets or intermediate artifacts that are not included in the lightweight final demo package.

The supported final runtime entry point is:

```powershell
python -m src.demo.app
```

---

## Important Limitations

This system is a research prototype.

It is not:

- a medical device
- a clinical decision-support system
- a diagnostic system
- a substitute for a healthcare professional
- a validated measure of medical correctness

Model preference percentages shown in the interface represent normalized relative label preferences for that case. They are not calibrated probabilities of medical correctness.

Retrieval similarity represents semantic question similarity to the controlled local corpus. It does not prove that evidence or candidate answers are medically correct.

Human review remains part of the final architecture.

---

## Reproducibility Validation

The final package was tested from a fresh Python virtual environment.

The validation included:

- clean PyTorch CUDA installation
- clean `requirements.txt` installation
- successful CUDA/GPU detection
- evidence retrieval
- loading both Qwen LoRA adapters
- PubMedQA evaluation
- Med-HALT evaluation
- candidate-evidence consistency analysis
- final routing
- successful end-to-end benchmark-pass case

This confirms that the final bundled demo can be reproduced independently of the original development virtual environment.
