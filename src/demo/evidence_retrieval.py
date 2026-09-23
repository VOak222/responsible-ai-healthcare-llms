"""Local, provenance-preserving retrieval for the manual evaluation demo.

Only controlled PubMedQA training and validation data are indexed.
Frozen benchmark tests and Med-HALT test files are intentionally excluded.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
INDEX_DIR = PROJECT_DIR / "artifacts" / "evidence_index"

INDEX_PATH = INDEX_DIR / "pubmedqa_train_validation.faiss"
METADATA_PATH = INDEX_DIR / "pubmedqa_train_validation_metadata.jsonl"
MANIFEST_PATH = INDEX_DIR / "pubmedqa_train_validation_manifest.json"

CORPUS_PATHS = [
    DATA_DIR / "controlled_benchmark_train.jsonl",
    DATA_DIR / "controlled_benchmark_validation.jsonl",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EvidenceRecord:
    record_id: str
    split: str
    research_question: str
    evidence: str


def read_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def parse_pubmedqa_prompt(prompt):
    """Extract the question and abstract from the existing project prompt."""

    question_match = re.search(
        r"Research question:\s*(.*?)\s*PubMed abstract evidence:",
        prompt,
        flags=re.DOTALL,
    )

    evidence_match = re.search(
        r"PubMed abstract evidence:\s*(.*?)\s*Decision:\s*$",
        prompt,
        flags=re.DOTALL,
    )

    if not question_match or not evidence_match:
        raise ValueError(
            "A controlled PubMedQA prompt did not match the expected format."
        )

    question = question_match.group(1).strip()
    evidence = evidence_match.group(1).strip()

    return question, evidence


def load_corpus():
    records = []

    for file_path in CORPUS_PATHS:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required PubMedQA corpus file was not found: {file_path}"
            )

        split = "train" if file_path.name.endswith("_train.jsonl") else "validation"

        for row in read_jsonl(file_path):
            question, evidence = parse_pubmedqa_prompt(row["prompt"])

            records.append(
                EvidenceRecord(
                    record_id=str(row["id"]),
                    split=split,
                    research_question=question,
                    evidence=evidence,
                )
            )

    return records


def build_evidence_index():
    """Build a local FAISS index from approved project research text only."""

    records = load_corpus()

    if not records:
        raise RuntimeError("No PubMedQA evidence records were available.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)

    retrieval_texts = [
        (
            f"Research question: {record.research_question}\n"
            f"PubMed abstract evidence: {record.evidence}"
        )
        for record in records
    ]

    embeddings = model.encode(
        retrieval_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "records": len(records),
        "included_files": [
            str(file_path.relative_to(PROJECT_DIR))
            for file_path in CORPUS_PATHS
        ],
        "excluded_files": [
            "controlled_benchmark_test.jsonl",
            "controlled_medhalt_hallucination_test.jsonl",
            "unseen_medhalt_reasoning_fake_stress_test.jsonl",
        ],
        "purpose": (
            "Local research-prototype retrieval for manual evaluation; "
            "not a clinical knowledge base."
        ),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest


class EvidenceRetriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Evidence index is missing. "
                "Run scripts/build_pubmedqa_evidence_index.py first."
            )

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(str(INDEX_PATH))

        self.records = []

        with open(METADATA_PATH, encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.records.append(EvidenceRecord(**json.loads(line)))

    def retrieve(self, question, candidate_answer, top_k=3):
        query = (
            f"Research question: {question}\n"
            f"Candidate answer: {candidate_answer}"
        )

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            vector,
            min(top_k, len(self.records)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            record = self.records[int(index)]

            results.append(
                {
                    "record_id": record.record_id,
                    "split": record.split,
                    "research_question": record.research_question,
                    "evidence": record.evidence,
                    "similarity_score": round(float(score), 3),
                    "source_label": "Local PubMedQA research corpus",
                }
            )

        return results