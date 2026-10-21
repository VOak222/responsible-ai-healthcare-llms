"""Read-only final dataset and runtime inventory verification.

Run from repository root:
    python tools/verify_final_submission.py

Uses only the Python standard library. Does not modify repository files.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        errors.append(description)


def records(path):
    p = ROOT / path
    if not p.is_file():
        check(False, f"Exists: {path}")
        return []
    try:
        with p.open(encoding="utf-8") as fh:
            values = [json.loads(line) for line in fh if line.strip()]
    except (OSError, ValueError) as exc:
        check(False, f"Valid JSONL: {path} ({exc})")
        return []
    check(True, f"Valid JSONL: {path}")
    return values


print("=== FINAL JSONL SPLITS ===")
specs = {
    "data/processed/controlled_benchmark_train.jsonl": (486, {"yes": 188, "no": 188, "maybe": 110}),
    "data/processed/controlled_benchmark_validation.jsonl": (100, {"yes": 50, "no": 50}),
    "data/processed/controlled_benchmark_test.jsonl": (200, {"yes": 100, "no": 100}),
    "data/processed/controlled_medhalt_hallucination_train.jsonl": (4000, {"supported": 2000, "hallucinated": 2000}),
    "data/processed/controlled_medhalt_hallucination_validation.jsonl": (1000, {"supported": 500, "hallucinated": 500}),
    "data/processed/controlled_medhalt_hallucination_test.jsonl": (1000, {"supported": 500, "hallucinated": 500}),
    "data/processed/unseen_medhalt_reasoning_fake_stress_test.jsonl": (1000, {"supported": 500, "hallucinated": 500}),
}
loaded = {}
for filename, (expected_rows, labels) in specs.items():
    rows = records(filename)
    loaded[filename] = rows
    check(len(rows) == expected_rows, f"Row count {filename}: {len(rows)}/{expected_rows}")
    observed = dict(Counter(str(row.get("label", "")).lower() for row in rows))
    check(observed == labels, f"Class counts {filename}: {observed}")

print("\n=== DISJOINT FINAL SPLITS ===")
prefix = "data/processed/controlled_benchmark_"
pubmed_ids = {
    k: {str(row.get("id", "")) for row in loaded[prefix + k + ".jsonl"]}
    for k in ("train", "validation", "test")
}
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    check(not (pubmed_ids[a] & pubmed_ids[b]), f"PubMedQA IDs disjoint: {a} / {b}")

m_prefix = "data/processed/controlled_medhalt_hallucination_"
medhalt_groups = {
    k: {str(row.get("question_group_id", "")) for row in loaded[m_prefix + k + ".jsonl"]}
    for k in ("train", "validation", "test")
}
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    check(not (medhalt_groups[a] & medhalt_groups[b]), f"Med-HALT question-group IDs disjoint: {a} / {b}")

print("\n=== FINAL V3 DATASET AND MANIFESTS ===")
v3 = ROOT / "data/evaluation/pubmedqa_natural_candidate_test_v3_100.csv"
if v3.is_file():
    with v3.open(encoding="utf-8-sig", newline="") as fh:
        v3_rows = list(csv.DictReader(fh))
    check(len(v3_rows) == 100, f"V3 cross-format set: {len(v3_rows)}/100 rows")
    counts = dict(Counter(str(r.get("expected_medhalt_label", "")).lower() for r in v3_rows))
    check(counts == {"supported": 50, "hallucinated": 50}, f"V3 label balance: {counts}")
else:
    check(False, "V3 cross-format set exists")

for path in (
    "configs/controlled_pubmedqa_binary_benchmark_manifest.json",
    "configs/controlled_medhalt_hallucination_benchmark_manifest.json",
    "configs/unseen_medhalt_reasoning_fake_stress_test_manifest.json",
):
    check((ROOT / path).is_file(), f"Manifest exists: {path}")

print("\n=== REFERENCED DELIVERABLES ===")
paths = (
    "data/evaluation/pubmedqa_100row_evaluation_set.csv",
    "data/evaluation/pubmedqa_100row_evaluation_set.xlsx",
    "data/evaluation/pubmedqa_medhalt_cross_dataset_100.csv",
    "data/evaluation/pubmedqa_natural_candidate_test_v3_100.xlsx",
    "results/pubmedqa_100row_comparison/pubmedqa_100row_model_comparison.csv",
    "results/pubmedqa_natural_candidate_v3/natural_v3_base_vs_medhalt_lora_comparison.csv",
    "src/demo/app.py",
    "src/demo/evidence_retrieval.py",
    "src/demo/live_evaluator.py",
    "scripts/build_pubmedqa_evidence_index.py",
    "requirements.txt",
    "docs/DATASET_MANIFEST.md",
    "docs/APPLICATION_ARCHITECTURE.md",
    "docs/DEPLOYMENT_GUIDE.md",
    "docs/DEMO_USER_MANUAL.md",
)
for path in paths:
    check((ROOT / path).is_file(), f"File exists: {path}")

print("\n=== LOCAL RUNTIME ARTIFACTS (NOT PART OF REMOTE GIT REPOSITORY) ===")
for path in (
    "artifacts/adapters/qwen2_5_3b_medhalt_hallucination_best_validation/adapter_config.json",
    "artifacts/adapters/qwen2_5_3b_medhalt_hallucination_best_validation/adapter_model.safetensors",
    "artifacts/checkpoints/qwen2_5_3b_controlled_benchmark/epoch_1/adapter_config.json",
    "artifacts/checkpoints/qwen2_5_3b_controlled_benchmark/epoch_1/adapter_model.safetensors",
    "artifacts/evidence_index/pubmedqa_train_validation.faiss",
    "artifacts/evidence_index/pubmedqa_train_validation_metadata.jsonl",
    "artifacts/evidence_index/pubmedqa_train_validation_manifest.json",
):
    check((ROOT / path).is_file(), f"Local asset exists: {path}")

print("\n" + ("FINAL AUDIT PASSED" if not errors else f"FINAL AUDIT FAILED ({len(errors)} issue(s))"))
sys.exit(0 if not errors else 1)
