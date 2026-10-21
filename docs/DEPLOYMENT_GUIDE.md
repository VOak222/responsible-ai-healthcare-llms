# Deployment Guide — Clean Windows Machine

**Responsible AI Evaluation Framework for Healthcare LLMs**  
Final Tech Mahindra AI internship research prototype, October 2026. This is **not** a clinical decision-support tool.

## 1. What you need

The **GitHub repository alone is not the complete runnable distribution**, because it does not include the selected trained LoRA weights or the prebuilt local FAISS index. Obtain the separately validated **final runtime ZIP** containing these assets. If working from a fresh GitHub checkout, place the files from that ZIP into the *same relative paths* shown below. Do not flatten the folders.

### Verified development / clean-install environment

| Item | Verified setting |
| --- | --- |
| OS | Windows 11 |
| Python | 3.14.0 |
| CUDA-enabled PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Live GPU requirement | CUDA-capable NVIDIA GPU |
| Internet | First run only, for Qwen base model and embedding-model downloads |

The live evaluator checks for CUDA and will fail if no NVIDIA CUDA GPU is available. The stated hardware/software combination is **the tested setup**, not a guarantee that other versions will work.

### Essential paths

```text
src/demo/app.py
src/demo/live_evaluator.py
src/demo/evidence_retrieval.py
scripts/build_pubmedqa_evidence_index.py
requirements.txt
artifacts/adapters/qwen2_5_3b_medhalt_hallucination_best_validation/adapter_config.json
artifacts/adapters/qwen2_5_3b_medhalt_hallucination_best_validation/adapter_model.safetensors
artifacts/checkpoints/qwen2_5_3b_controlled_benchmark/epoch_1/adapter_config.json
artifacts/checkpoints/qwen2_5_3b_controlled_benchmark/epoch_1/adapter_model.safetensors
artifacts/evidence_index/pubmedqa_train_validation.faiss
artifacts/evidence_index/pubmedqa_train_validation_metadata.jsonl
artifacts/evidence_index/pubmedqa_train_validation_manifest.json
data/processed/controlled_benchmark_train.jsonl
data/processed/controlled_benchmark_validation.jsonl
```

The complete raw PubMedQA test set, Med-HALT training/test splits, and other experiment records are for reproducibility, not required by this live demo. See [DATASET_MANIFEST.md](DATASET_MANIFEST.md).

## 2. Create and activate a new environment

Open PowerShell in the **project root**:

```powershell
cd C:\path\to\responsible-ai-healthcare-llms
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If script activation is blocked, in the **same PowerShell session**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install CUDA-enabled PyTorch and dependencies

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -m pip install -r requirements.txt
```

The verified machine printed:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4070 Laptop GPU
```

If `torch.cuda.is_available()` is `False`, check the NVIDIA driver, supported GPU, virtual environment, and CUDA-enabled PyTorch installation before attempting live inference.

## 4. Check or rebuild evidence index

The runtime ZIP includes the three evidence-index files. If they are absent, outdated, or intentionally being rebuilt, use:

```powershell
python scripts/build_pubmedqa_evidence_index.py
```

This requires the **controlled PubMedQA train and validation JSONL files** and downloads/loads the MiniLM embedding model as necessary. It indexes **586 records** (486 train + 100 validation). The **frozen PubMedQA test** is excluded on purpose.

## 5. Start the local application

```powershell
python -m src.demo.app
```

Open `http://127.0.0.1:8000` in a browser. Optional health endpoint: `http://127.0.0.1:8000/api/health`.

The first evaluation may take longer because Hugging Face downloads or loads:

- `Qwen/Qwen2.5-3B-Instruct` (the base model is **not** stored in this repository/ZIP).
- `sentence-transformers/all-MiniLM-L6-v2`.

The application runs locally after the required weights are cached; no external hosted model API is needed for the evaluator.

## 6. Sanity check: controlled alignment case

In the browser, submit:

**Research question**

> Does hypoglycaemia increase the risk of cardiovascular events?

**AI-generated candidate answer**

> Yes. Severe hypoglycaemia was associated with a higher risk of cardiovascular events and cardiovascular death, although non-severe hypoglycaemia was not associated with increased risk after adjustment.

**Evidence mode:** automatic (leave manual evidence empty).

Expected: the identical PubMedQA research question retrieves near **1.000 similarity**. If Med-HALT reports supported, PubMedQA reports yes, the explicit candidate conclusion is yes and both task margins are strong, the result is `Benchmark pass — evidence aligned`. Depending on task-specific label margins, the conservative system can instead route to human review; that is not an installation failure.

For other example cases and user-facing interpretation, read [DEMO_USER_MANUAL.md](DEMO_USER_MANUAL.md).

## 7. Stop / restart / troubleshoot

Stop: `Ctrl+C` in the serving PowerShell window. Restart in a later session by activating the existing `.venv` and running `python -m src.demo.app`.

| Symptom | Check |
| --- | --- |
| CUDA not available | NVIDIA GPU/driver, correct environment, CUDA-enabled PyTorch version. |
| Adapter not found | Selected Med-HALT folder under `artifacts/adapters/` and selected PubMedQA `epoch_1` folder under `artifacts/checkpoints/` were copied from final ZIP. |
| Evidence index not found | Three index files under `artifacts/evidence_index/` or run index-builder with PubMedQA train/validation JSONL. |
| First evaluation slow | Allow first Hugging Face downloads and initial GPU load to complete. |
| Port 8000 in use | Stop the existing service or process using that port. |
| Unrelated question yields no evidence | This is expected below the 0.70 retrieval relevance guard. It is **not** proof the candidate answer is wrong. |

## 8. Reproducibility and safety

- `docs/DATASET_MANIFEST.md`: controlled dataset inventory, split seeds, source provenance and evaluation files.
- `docs/APPLICATION_ARCHITECTURE.md`: precise code/data flow, model roles and routing gates.
- `docs/DEMO_USER_MANUAL.md`: fields, interpretation and example review cases.
- `src/demo/`: live code; `src/models/`: training scripts; `src/evaluation/`: fixed test/evaluation scripts.

**Safety boundary:** automatic pass/fail refers only to alignment/conflict *within the controlled research evidence workflow*. This is not clinical validation, diagnosis, prescribing, or medical approval. User-supplied evidence is unverified and remains routed to a human reviewer.
