"""Build the local PubMedQA evidence index for the manual evaluation demo."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.demo.evidence_retrieval import build_evidence_index


if __name__ == "__main__":
    manifest = build_evidence_index()

    print("\nEvidence index created successfully.")
    print(f"Indexed records: {manifest['records']}")

    print("\nIncluded files:")
    for file_path in manifest["included_files"]:
        print(f"- {file_path}")

    print("\nFrozen tests were not indexed.")