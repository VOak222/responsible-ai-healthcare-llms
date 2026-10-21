# Demo User Manual — Evaluating Healthcare AI Answers

**Responsible AI Evaluation Framework for Healthcare LLMs**  
Local research prototype using independent evidence and task-specific model checks. This tool does **not** generate medical advice or provide clinical approval.

## 1. Before you begin

Install and start the application following [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md). In the running browser application at `http://127.0.0.1:8000`, submit a research question and a candidate answer generated **outside** this system. The application checks an existing answer; it does not draft one.

The application exposes two evidence modes. In **automatic** mode, it searches a deliberately limited, controlled PubMedQA research corpus using the **question only**. In **manual** mode, you provide an evidence abstract or excerpt; the source is shown as **unverified** and can never receive an automatic benchmark pass/fail.

## 2. Fill in the fields

1. **Research question:** enter the original medical *research* question to be evaluated. Phrase the question as clearly and specifically as possible.
2. **AI-generated candidate answer:** paste the complete external AI response. A clear answer beginning `Yes`, `No`, or `Maybe` permits the conservative conclusion parser to compare stances; ambiguous wording may be routed to review even if medically plausible.
3. **Evidence (optional):** leave empty for automatic PubMedQA retrieval, or paste a study abstract/source excerpt for the manual-evidence route. Manual inputs are **not** provenance-verified by this prototype.
4. Submit/evaluate, then inspect the retrieved evidence, both model outputs, candidate conclusion, consistency, confidence/margin indicators, and final disposition.

The visible PubMedQA evidence-conclusion signal answers the **original research question using the selected evidence**; it does not independently validate every claim in the pasted candidate. Med-HALT checks the *candidate answer* against the question/evidence.

## 3. Reading the result

| Display element | Meaning | What it **does not** mean |
| --- | --- | --- |
| Retrieved PubMedQA evidence | An abstract associated with a stored research question; returned by semantic search of the question. | That it is the current medical consensus. |
| Retrieval similarity | Semantic closeness of questions on a normalized embedding scale. | Probability the answer is medically true. |
| PubMedQA adapter | Evidence supports `yes`, `no`, or `maybe` for the original question. | A complete medical accuracy review of the candidate's every claim. |
| Med-HALT adapter | Classifies question + evidence + candidate as `supported` or `hallucinated`. | An infallible hallucination detector. |
| Candidate conclusion | Explicit `yes` / `no` / `maybe` or `uncertain` stance recognized by a conservative parser. | A full natural-language entailment model. |
| Consistency | Compares the candidate conclusion with the PubMedQA evidence conclusion. | Proof of medical validity outside the presented evidence. |
| Relative model preference / label separation | Task-specific label ranking and decisiveness. | Calibrated medical correctness probabilities. |

**Thresholds:** automatic retrieval rejects matches below **0.70**. Matches from **0.70 to below 0.90** are displayable but require review. A match **at least 0.90** is only *eligible* for automatic benchmark pass/fail when the model and consistency conditions also justify it. These are research safeguards, not clinical cutoffs.

## 4. Outcome labels

| Disposition | Interpretation and next action |
| --- | --- |
| **Benchmark pass — evidence aligned** | Strong controlled match, supported candidate, explicit aligned yes/no conclusion and sufficiently decisive signals. **Benchmark result only**, not clinical approval. |
| **Benchmark fail — evidence conflict** | Controlled evidence conflicts with the candidate's stated yes/no conclusion or a strong Med-HALT hallucination signal fires. Review the original evidence; not a clinical rejection. |
| **Human review required** | Moderate evidence match, unclear answer stance, mixed signals, or weak label separation; the system refuses to force a binary answer. |
| **Routine human review — no escalation** | Manually supplied, unverified evidence with aligned signals; still requires human review. |
| **Mandatory human review** | A higher-priority manual-evidence conflict/hallucination trigger or other high-risk review signal. |
| **Insufficient local evidence — human review required** | No nearest question passed the 0.70 guard; automatic model evaluation stopped. This **does not** mean the candidate is incorrect. |

Manual evidence always remains under human review; the application may use a general required-review route when manual signals are uncertain.

## 5. Controlled examples for demonstration

**Case A — aligned (automatic):**

- **Question:** `Does hypoglycaemia increase the risk of cardiovascular events?`
- **Candidate:** `Yes. Severe hypoglycaemia was associated with a higher risk of cardiovascular events and cardiovascular death, although non-severe hypoglycaemia was not associated with increased risk after adjustment.`
- **Mode:** Automatic (no manual evidence).
- **Expected behavior:** exact question similarity ~1.000; with strong and aligned task-specific signals, `Benchmark pass — evidence aligned`.

**Case B — contradiction (automatic):**

- **Same question.**
- **Candidate:** `No. Hypoglycaemia was not associated with cardiovascular risk or cardiovascular death, including in severe cases.`
- **Expected:** Retrieved evidence remains the same; yes evidence versus no candidate is a contradiction, leading to `Benchmark fail — evidence conflict` on the controlled high-similarity path.

**Case C — unclear stance (automatic):**

- **Same question.**
- **Candidate:** `The study reported an association between severe hypoglycaemia and several cardiovascular outcomes, while non-severe hypoglycaemia did not show the same association after adjustment.`
- **Expected:** no explicit yes/no/maybe conclusion; `Human review required`, not automatically marked wrong.

**Case D — out of local corpus (automatic):**

- **Question:** `Does the addition of empagliflozin reduce cardiovascular death or heart-failure hospitalization in HFpEF?`
- **Expected:** the tested nearest similarity was about 0.469, below 0.70. No local evidence is accepted and automatic evaluation stops. Other medical questions absent from the small corpus may behave similarly.

**Case E — manual evidence:** paste an excerpt with the question and candidate. If signals align, expect **routine** human review, *not* benchmark pass. Contradictory manual evidence can trigger **mandatory** human review. No claim is made about the authenticity of pasted text.

Actual inferences may vary with hardware, model loading and task-specific score margins; the system intentionally prioritizes review when its conditions for an automatic disposition are not met.

## 6. Operating boundaries

- The corpus is intentionally limited to **586 controlled PubMedQA train/validation** research records; it is **not** a comprehensive biomedical search engine.
- A perfect similarity score means a stored *question* matched; it is **not** a guarantee of scientific or clinical correctness.
- The Med-HALT and PubMedQA adapters perform **different** tasks. Their metrics and raw scores are not interchangeable.
- The fairness work based on EquityMedQA is exploratory and **not** a live automatic fairness verdict.
- Never enter identifiable patient data into a demo case. This research application has not been validated to process clinical patient records.
- Qualified human judgment remains necessary for ambiguous, conflicting, or externally provided evidence.

Related references: [APPLICATION_ARCHITECTURE.md](APPLICATION_ARCHITECTURE.md) and [DATASET_MANIFEST.md](DATASET_MANIFEST.md).
