# Dataset Manifest and Evaluation Provenance

**Project:** Responsible AI Evaluation Framework for Healthcare LLMs  
**Repository:** `VOak222/responsible-ai-healthcare-llms`  
**Scope:** Final controlled research benchmarks, frozen evaluation records, and provenance. This is not a clinical system.

## 1. What is versioned in GitHub

The repository versions the **project-generated controlled benchmark snapshots** and exact evaluation subsets below. These are the datasets needed to inspect the reported final experiments. Original upstream public-dataset downloads and some historical normalized intermediate files are **not** all in this repository. Do not confuse the controlled project snapshots with the full upstream datasets.

| Dataset / task | Exact repository file | Rows | Role |
| --- | --- | ---: | --- |
| PubMedQA | `data/processed/controlled_benchmark_train.jsonl` | 486 | Model training; 188 yes, 188 no, 110 maybe |
| PubMedQA | `data/processed/controlled_benchmark_validation.jsonl` | 100 | Checkpoint selection; 50 yes, 50 no |
| PubMedQA | `data/processed/controlled_benchmark_test.jsonl` | 200 | **Frozen** evaluation; 100 yes, 100 no |
| Med-HALT | `data/processed/controlled_medhalt_hallucination_train.jsonl` | 4,000 | Model training; 2,000 supported, 2,000 hallucinated |
| Med-HALT | `data/processed/controlled_medhalt_hallucination_validation.jsonl` | 1,000 | Checkpoint selection; 500 per class |
| Med-HALT FCT | `data/processed/controlled_medhalt_hallucination_test.jsonl` | 1,000 | **Frozen** FCT evaluation; 500 per class |
| Med-HALT Reasoning Fake | `data/processed/unseen_medhalt_reasoning_fake_stress_test.jsonl` | 1,000 | Separate unseen stress test; 500 per class |

The `.jsonl` files are **one JSON object per line**, not a single JSON array. PubMedQA records include `id`, `label`, `prompt`, and `response`. Med-HALT controlled records retain task prompts, labels, and origin metadata prepared by the repository scripts.

### Controlled split manifests (versioned)

- `configs/controlled_pubmedqa_binary_benchmark_manifest.json` — seed `20260916`, record IDs, counts, frozen-test policy.
- `configs/controlled_medhalt_hallucination_benchmark_manifest.json` — seed `42`, source partitions, question-group disjointness, class counts.
- `configs/unseen_medhalt_reasoning_fake_stress_test_manifest.json` — seed `113`, exclusion of Fake question groups previously used for training.

**Leakage policy:** final frozen test records were not used for checkpoint selection. The PubMedQA **retrieval** corpus uses *only* the 486 training and 100 validation rows; it excludes the frozen 200-row test set and every Med-HALT test/stress set.

## 2. Exact auxiliary evaluation datasets

| Repository file | Purpose |
| --- | --- |
| `data/evaluation/pubmedqa_100row_evaluation_set.csv` | Fixed 100-case PubMedQA diagnostic (50 yes, 50 no), sampled from the already-frozen PubMedQA test set |
| `data/evaluation/pubmedqa_100row_evaluation_set.xlsx` | Spreadsheet companion of the 100-case diagnostic |
| `data/evaluation/pubmedqa_medhalt_cross_dataset_100.csv` | Earlier constructed 100-case stance-pair cross-dataset experiment |
| `data/evaluation/pubmedqa_medhalt_cross_dataset_100.xlsx` | Spreadsheet companion |
| `data/evaluation/pubmedqa_natural_candidate_test_100.csv` | Earlier natural-language candidate draft, **not final** |
| `data/evaluation/pubmedqa_natural_candidate_test_v2_100.csv` | Intermediate V2 natural-language candidate set, **not final** |
| `data/evaluation/pubmedqa_natural_candidate_test_v3_100.csv` | **Final** 100-case natural-language stress set: 50 supported / 50 counterfactual candidates |
| `data/evaluation/pubmedqa_natural_candidate_test_v3_100.xlsx` | Spreadsheet companion of final V3 |

The V3 set followed structural checks and a manual semantic spot-check. Four inconsistent counterfactuals were repaired **before final V3 was frozen**. Earlier candidates remain for experiment history; the report's final 63% versus 86% cross-format comparison uses V3, **not** V2.

### Selected versioned result files

- PubMedQA full frozen benchmark: `results/qwen2_5_3b_controlled_benchmark_test_summary.csv` and `results/qwen2_5_3b_controlled_benchmark_test_predictions.csv`.
- BioGPT comparator: `results/biogpt_large_controlled_benchmark_test_summary.csv` and `results/biogpt_large_controlled_benchmark_test_predictions.csv`.
- Med-HALT frozen FCT: `results/qwen2_5_3b_medhalt_hallucination_test_summary.csv` and corresponding `_predictions.csv`.
- Med-HALT unseen Fake: `results/qwen2_5_3b_unseen_fake_stress_test_summary.csv` and corresponding `_predictions.csv`.
- PubMedQA 100-row diagnostic: `results/pubmedqa_100row_comparison/`.
- Constructed cross-dataset: `results/pubmedqa_medhalt_cross_dataset/`.
- Final natural-language V3: `results/pubmedqa_natural_candidate_v3/`, including `natural_v3_base_vs_medhalt_lora_comparison.csv` and `.xlsx`.
- MedHallu baseline, EquityMedQA exploratory fairness, and supporting experiment outputs are in `results/` and `reports/`. These are historical or exploratory, **not** automated inputs to final live routing.

## 3. Dataset origins and rebuilding boundaries

| Source | Role | Upstream / preparation |
| --- | --- | --- |
| **PubMedQA**, expert-labelled | Evidence-conclusion LoRA, BioGPT comparison, local retrieval | `src/models/prepare_controlled_pubmedqa_benchmark.py` downloads official `https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`; saves locally as `data/raw/ori_pqal_expert.json`; generates deterministic controlled JSONL splits and ID manifest. |
| **Med-HALT**, Reasoning FCT and Reasoning Fake | Supported-vs-hallucinated LoRA; frozen FCT and unseen Fake evaluations | `src/models/prepare_controlled_medhalt_hallucination_benchmark.py` reads local normalized `data/processed/medhalt_reasoning_fct_common.csv` and `medhalt_reasoning_fake_common.csv`. `src/models/prepare_unseen_medhalt_reasoning_fake_stress_test.py` creates the Fake stress set. The normalized `*_common.csv` inputs are **not committed**; the controlled JSONL outputs *are*. |
| **MedHallu** | TF-IDF / Logistic Regression baseline and early error analysis | `src/data/download_medhallu.py` uses Hugging Face `UTAustin-AIHealth/MedHallu` (`pqa_labeled`, `pqa_artificial`); upstream files are not committed. |
| **EquityMedQA** | Exploratory fairness-risk inventory and manual evaluation | `src/data/prepare_equitymedqa_inventory.py` expects source files in `data/raw/equitymedqa/`; original raw inventories are not committed. Selected analyses are under `results/equitymedqa_fairness/` and `reports/`. |

**Licensing and provenance:** Before redistributing *original* upstream datasets, check the current license and terms at each authoritative source. This repository is not a mirror of full upstream dataset distributions.

**Rebuild limitations:** PubMedQA controlled splits can be recreated with the supplied preparation script and upstream expert dataset. Recreating Med-HALT splits from original upstream data also requires restoring the historical normalized `*_common.csv` inputs. The committed controlled JSONL files already preserve the final Med-HALT benchmark records.

## 4. Runtime evidence index (separate from GitHub data)

The local runnable ZIP, rather than the Git repository, supplies:

- `artifacts/evidence_index/pubmedqa_train_validation.faiss`
- `artifacts/evidence_index/pubmedqa_train_validation_metadata.jsonl`
- `artifacts/evidence_index/pubmedqa_train_validation_manifest.json`

To rebuild the index from versioned PubMedQA training/validation JSONL:

```powershell
python scripts/build_pubmedqa_evidence_index.py
```

The index contains **586** question embeddings, using `sentence-transformers/all-MiniLM-L6-v2` and normalized FAISS inner-product similarity. The frozen test set is excluded. A minimum 0.70 similarity is a **prototype retrieval guard**, not a medical correctness threshold.

## 5. Preserve the evaluation protocol

1. Train only on controlled **training** records.
2. Select checkpoints only with **validation** records.
3. Use frozen test records for final reporting, not model selection.
4. Keep Med-HALT hallucination and PubMedQA evidence-conclusion metrics separate.
5. Distinguish the small audited V3 natural-language stress test from native Med-HALT results.
6. Never interpret model preference, label separation, retrieval similarity, or benchmark pass/fail as clinical certification.

See [APPLICATION_ARCHITECTURE.md](APPLICATION_ARCHITECTURE.md) and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
