import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
RAW_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

RAW_PATH = RAW_DIR / "ori_pqal_expert.json"


def evidence_text(contexts):
    if isinstance(contexts, list):
        return " ".join(str(item) for item in contexts)

    if isinstance(contexts, dict):
        return " ".join(str(item) for item in contexts.values())

    return str(contexts)


def make_record(record_id, item):
    decision = item["final_decision"].strip().lower()

    prompt = (
        "You are a biomedical evidence assistant.\n"
        "Use only the supplied PubMed abstract evidence.\n"
        "Answer with one decision: yes, no, or maybe.\n\n"
        f"Research question:\n{item['QUESTION']}\n\n"
        f"PubMed abstract evidence:\n{evidence_text(item['CONTEXTS'])}\n\n"
        "Decision:"
    )

    return {
        "id": record_id,
        "label": decision,
        "prompt": prompt,
        "response": f" {decision}",
    }


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        print("Downloading official PubMedQA expert-labelled data...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)

    with open(RAW_PATH, encoding="utf-8") as file:
        expert_data = json.load(file)

    records_by_label = defaultdict(list)

    for record_id, item in expert_data.items():
        label = item["final_decision"].strip().lower()

        if label in {"yes", "no", "maybe"}:
            records_by_label[label].append(make_record(record_id, item))

    rng = random.Random(SEED)

    for label in records_by_label:
        rng.shuffle(records_by_label[label])

    train_records = (
        records_by_label["yes"][:200]
        + records_by_label["no"][:200]
        + records_by_label["maybe"][:110]
    )

    validation_records = (
        records_by_label["yes"][200:300]
        + records_by_label["no"][200:300]
    )

    test_records = (
        records_by_label["yes"][300:]
        + records_by_label["no"][300:]
    )

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    write_jsonl(
        train_records,
        OUTPUT_DIR / "biogpt_pubmedqa_train.jsonl",
    )
    write_jsonl(
        validation_records,
        OUTPUT_DIR / "biogpt_pubmedqa_validation.jsonl",
    )
    write_jsonl(
        test_records,
        OUTPUT_DIR / "biogpt_pubmedqa_test.jsonl",
    )

    print("\nBioGPT PubMedQA data preparation complete.")
    print("Training rows:", len(train_records), Counter(row["label"] for row in train_records))
    print(
        "Validation rows:",
        len(validation_records),
        Counter(row["label"] for row in validation_records),
    )
    print("Untouched test rows:", len(test_records), Counter(row["label"] for row in test_records))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()