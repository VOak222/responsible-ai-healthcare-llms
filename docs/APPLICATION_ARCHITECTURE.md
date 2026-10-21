# Application Architecture and Data Flow

**Responsible AI Evaluation Framework for Healthcare LLMs** — final local research prototype, October 2026.

## 1. Purpose and safety boundary

The application evaluates an **externally generated** answer to a healthcare research question. It is not a medical answer generator, diagnostic tool, clinical decision-support system, or medical device. Benchmark dispositions are internal research labels, **not** medical approval/rejection or a calibrated probability of correctness.

## 2. Final inference workflow

```text
Research question + external AI candidate answer
                         |
                         v
                 Evidence selection
     +------------------------------------------+
     | Automatic: embed QUESTION ONLY with      |
     | MiniLM -> FAISS -> PubMedQA abstract      |
     | Manual: user-pasted text (UNVERIFIED)     |
     +------------------------------------------+
                         |
                         v
             Freeze the selected evidence
                         |
             +-----------+------------+
             |                        |
             v                        v
  PubMedQA LoRA adapter       Med-HALT LoRA adapter
  question + evidence        question + evidence + candidate
  -> yes / no / maybe        -> supported / hallucinated
             |                        |
             +------------+-----------+
                          |
                          v
        Parse explicit candidate conclusion
               yes / no / maybe / uncertain
                          |
                          v
             Deterministic consistency
          consistent / contradiction / uncertain
                          |
                          v
       Retrieval gates + label separation + rules
                          |
                          v
          Benchmark pass / benchmark fail /
           human review / insufficient evidence
```

**Important independence constraint:** automatic evidence retrieval sees the research question, **not** the candidate answer. The candidate cannot steer the paper selected to judge it.

## 3. Components, algorithms, and source paths

| Component | Source | Responsibility |
| --- | --- | --- |
| FastAPI frontend / REST API / final routing | `src/demo/app.py` | Validates input, selects evidence mode, applies 0.90 automatic-decision gate, shows dispositions and reasons. |
| Semantic evidence retrieval | `src/demo/evidence_retrieval.py` | Embeds questions using `sentence-transformers/all-MiniLM-L6-v2`, normalizes vectors, FAISS inner-product search, returns abstract metadata, applies 0.70 relevance cutoff. |
| Evidence index build | `scripts/build_pubmedqa_evidence_index.py` | Builds and persists FAISS index + metadata + manifest from controlled PubMedQA **train and validation only**. |
| Dual-adapter model inference | `src/demo/live_evaluator.py` | Loads `Qwen/Qwen2.5-3B-Instruct` once with 4-bit NF4 quantization; selects separate PEFT LoRA adapters, scores their task-specific labels, parses candidate stance, computes consistency and model signal margins. |
| PubMedQA LoRA training | `src/models/train_qwen2_5_3b_controlled_benchmark.py` | Question + PubMed abstract evidence -> yes/no/maybe; checkpoint selected on validation. |
| Med-HALT LoRA training | `src/models/train_qwen2_5_3b_medhalt_hallucination.py` | Question + evidence + candidate answer -> supported/hallucinated; checkpoint selected on validation. |

The two LoRA adapters are **different tasks over the same base model**; their task-specific accuracy figures must never be averaged into a single clinical score. BioGPT-Large + LoRA was an *offline comparator*, not a third live evaluator.

## 4. Evidence retrieval and routing safeguards

Automatic retrieval uses 586 controlled PubMedQA records: 486 training plus 100 validation. **No frozen PubMedQA test item is indexed.** The system retrieves the nearest question embeddings and returns the corresponding source abstract only if the similarity guard is passed.

| Condition | Behavior |
| --- | --- |
| Best question similarity **below 0.70** | No sufficiently relevant local evidence; automatic model evaluation stops; `Insufficient local evidence — human review required`. |
| Best similarity **0.70 to below 0.90** | Show sufficiently relevant evidence but do **not** assign automatic benchmark pass/fail; human review, possibly mandatory if the underlying routing signals require it. |
| Best similarity **at least 0.90** | Eligible for controlled benchmark pass/fail **only** if other rules warrant it. |
| Explicit yes/no contradiction with PubMedQA evidence conclusion at eligible similarity | `Benchmark fail — evidence conflict`. |
| Strong Med-HALT hallucination signal at eligible similarity | `Benchmark fail — evidence conflict`. |
| Supported Med-HALT result, decisive yes/no evidence and candidate, consistent conclusions, strong margins and appropriate routing | `Benchmark pass — evidence aligned`. |
| Weak/mixed model signals or unclear candidate conclusion | Human review rather than forced automatic pass/fail. |
| Manually pasted evidence | **Always human review**: routine, required, or mandatory according to signals. Manual evidence is marked unverified and never receives automatic benchmark pass/fail. |

Thresholds are prototype control settings and not validated medical cutoffs. In code: retrieval relevance `0.70`, automatic-decision eligibility `0.90`, strong model label margin `0.75`. Label separation below `0.15` is low; `0.15` to below `0.75` is moderate; `0.75` or greater is strong.

The candidate-conclusion parser intentionally recognizes only explicit yes/no/maybe or clear uncertainty wording. Non-explicit stances return `uncertain`. That conservative restriction trades coverage for auditable reasoning.

## 5. What scores mean

- **Retrieval similarity:** semantic closeness between the submitted research question and a stored question; **not** correctness of evidence or answer.
- **Relative model preference:** relative ranking of available labels by each adapter for this case; **not** a calibrated probability of medical truth.
- **Label separation:** score gap between competing labels, indicating a more or less decisive task-specific prediction; **not** model clinical reliability.
- **Final disposition:** a transparent rules-based benchmark/review decision; **not** a diagnosis, approval, or prescription.

## 6. Runtime assets and placement

```text
src/demo/app.py
src/demo/live_evaluator.py
src/demo/evidence_retrieval.py
scripts/build_pubmedqa_evidence_index.py
artifacts/adapters/qwen2_5_3b_medhalt_hallucination_best_validation/
artifacts/checkpoints/qwen2_5_3b_controlled_benchmark/epoch_1/
artifacts/evidence_index/pubmedqa_train_validation.faiss
artifacts/evidence_index/pubmedqa_train_validation_metadata.jsonl
artifacts/evidence_index/pubmedqa_train_validation_manifest.json
data/processed/controlled_benchmark_train.jsonl
data/processed/controlled_benchmark_validation.jsonl
requirements.txt
```

**Distribution boundary:** the GitHub repo contains versioned code, dataset snapshots, manifests, benchmark outputs and documentation. The separate, validated **runtime ZIP** contains the selected adapters and prebuilt FAISS artifacts. The base Qwen and MiniLM embedding models are downloaded from Hugging Face during first use rather than bundled.

## 7. Development/evaluation lineage

1. MedHallu supported early TF-IDF + Logistic Regression baseline and error inspection.
2. Med-HALT Reasoning Fake/FCT exposed lexical shortcut risks; question/evidence/context-rich checking was retained.
3. SentenceTransformers + FAISS enabled semantic retrieval and demonstrated why nearest-neighbor matches need guards.
4. Qwen2.5-3B + separate LoRA adapters implemented task-specific evidence conclusion and hallucination checks.
5. Frozen PubMedQA, frozen Med-HALT FCT, unseen Med-HALT Fake, and cross-dataset natural-language tests established task-specific behavior and limitations.
6. Final FastAPI application combines evidence, two adapters, deterministic consistency and human-review routing.

See [DATASET_MANIFEST.md](DATASET_MANIFEST.md), [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) and [DEMO_USER_MANUAL.md](DEMO_USER_MANUAL.md).
