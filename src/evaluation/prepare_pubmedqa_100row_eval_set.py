import csv
import json
import random
from pathlib import Path

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "controlled_benchmark_test.jsonl"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_100row_evaluation_set.csv"
)


def parse_prompt(prompt):
    question_marker = "Research question:\n"
    evidence_marker = "\n\nPubMed abstract evidence:\n"
    decision_marker = "\n\nDecision:"

    question = (
        prompt.split(question_marker, 1)[1]
        .split(evidence_marker, 1)[0]
        .strip()
    )

    evidence = (
        prompt.split(evidence_marker, 1)[1]
        .rsplit(decision_marker, 1)[0]
        .strip()
    )

    return question, evidence


with SOURCE_FILE.open("r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

yes_rows = []
no_rows = []

for row_number, record in enumerate(records, start=1):
    if record["label"] == "yes":
        yes_rows.append((row_number, record))
    elif record["label"] == "no":
        no_rows.append((row_number, record))


random.seed(SEED)

selected_yes = random.sample(yes_rows, 50)
selected_no = random.sample(no_rows, 50)

selected_rows = selected_yes + selected_no

# Keep final CSV in original frozen-test order
selected_rows.sort(key=lambda x: x[0])


output_rows = []

for sample_id, (source_row_number, record) in enumerate(
    selected_rows,
    start=1,
):
    question, evidence = parse_prompt(record["prompt"])

    output_rows.append(
        {
            "sample_id": sample_id,
            "source_row_number": source_row_number,
            "pubmed_id": record["id"],
            "ground_truth_label": record["label"],
            "research_question": question,
            "pubmed_abstract_evidence": evidence,
            "source_split": "frozen_controlled_pubmedqa_test",
            "selection_seed": SEED,
        }
    )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=output_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(output_rows)


yes_count = sum(
    row["ground_truth_label"] == "yes"
    for row in output_rows
)

no_count = sum(
    row["ground_truth_label"] == "no"
    for row in output_rows
)

print()
print("=" * 70)
print("PUBMEDQA 100-ROW EVALUATION SET CREATED")
print("=" * 70)
print(f"Source file: {SOURCE_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print()
print(f"Total rows: {len(output_rows)}")
print(f"YES rows:   {yes_count}")
print(f"NO rows:    {no_count}")
print(f"Seed:       {SEED}")
print()
print(
    "These rows come only from the already-frozen "
    "PubMedQA controlled test set."
)
print(
    "They must not be used for training, "
    "checkpoint selection, or threshold tuning."
)
print("=" * 70)