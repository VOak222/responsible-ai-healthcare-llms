import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260916

RAW_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CONFIG_DIR = PROJECT_DIR / "configs"

RAW_PATH = RAW_DIR / "ori_pqal_expert.json"
MANIFEST_PATH = (
    CONFIG_DIR / "controlled_pubmedqa_binary_benchmark_manifest.json"
)

TRAIN_PATH = PROCESSED_DIR / "controlled_benchmark_train.jsonl"
VALIDATION_PATH = PROCESSED_DIR / "controlled_benchmark_validation.jsonl"
TEST_PATH = PROCESSED_DIR / "controlled_benchmark_test.jsonl"


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


def label_counts(records):
    return dict(Counter(record["label"] for record in records))


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

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

    # Frozen controlled protocol:
    # - balanced binary test: 100 yes / 100 no
    # - balanced binary validation: 50 yes / 50 no
    # - identical training rows for both models
    test_records = (
        records_by_label["yes"][:100]
        + records_by_label["no"][:100]
    )

    validation_records = (
        records_by_label["yes"][100:150]
        + records_by_label["no"][100:150]
    )

    train_records = (
        records_by_label["yes"][150:338]
        + records_by_label["no"][150:338]
        + records_by_label["maybe"][:110]
    )

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    write_jsonl(train_records, TRAIN_PATH)
    write_jsonl(validation_records, VALIDATION_PATH)
    write_jsonl(test_records, TEST_PATH)

    manifest = {
        "experiment_name": "controlled_pubmedqa_binary_benchmark",
        "seed": SEED,
        "rule": (
            "Both Qwen and BioGPT must start from their base models, "
            "use only controlled_benchmark_train.jsonl for training, "
            "use only controlled_benchmark_validation.jsonl for "
            "calibration, and never use controlled_benchmark_test.jsonl "
            "until final evaluation."
        ),
        "train_counts": label_counts(train_records),
        "validation_counts": label_counts(validation_records),
        "test_counts": label_counts(test_records),
        "train_ids": [record["id"] for record in train_records],
        "validation_ids": [record["id"] for record in validation_records],
        "test_ids": [record["id"] for record in test_records],
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nCONTROLLED BENCHMARK SPLIT FROZEN")
    print("Training rows:", len(train_records), label_counts(train_records))
    print("Validation rows:", len(validation_records), label_counts(validation_records))
    print("Balanced test rows:", len(test_records), label_counts(test_records))
    print("Manifest saved to:", MANIFEST_PATH)
    print("\nDo not use the controlled test file in training or calibration.")


if __name__ == "__main__":
    main()