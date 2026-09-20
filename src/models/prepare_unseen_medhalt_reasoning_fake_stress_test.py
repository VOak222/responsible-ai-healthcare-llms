import hashlib
import json
import re
from pathlib import Path

import pandas as pd


SEED = 113
QUESTION_GROUPS = 500

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
CONFIG_DIR = PROJECT_DIR / "configs"

SOURCE_PATH = DATA_DIR / "medhalt_reasoning_fake_common.csv"
CONTROLLED_TRAIN_PATH = (
    DATA_DIR / "controlled_medhalt_hallucination_train.jsonl"
)

OUTPUT_PATH = (
    DATA_DIR / "unseen_medhalt_reasoning_fake_stress_test.jsonl"
)

MANIFEST_PATH = (
    CONFIG_DIR / "unseen_medhalt_reasoning_fake_stress_test_manifest.json"
)


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def normalize_question(question):
    normalized = str(question).lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def make_group_id(question):
    question_hash = hashlib.sha256(
        normalize_question(question).encode("utf-8")
    ).hexdigest()[:16]

    return f"medhalt_reasoning_fake_{question_hash}"


def get_label(value):
    if int(value) == 0:
        return "supported"

    if int(value) == 1:
        return "hallucinated"

    raise ValueError(f"Unexpected is_hallucinated value: {value}")


def build_prompt_from_training_template(
    template_prompt,
    template_row,
    target_row,
):
    """
    Reuses the exact prompt wording from the controlled training benchmark,
    while replacing only question, evidence, and candidate answer values.
    """

    replacements = [
        ("question", "question"),
        ("knowledge", "knowledge"),
        ("answer", "answer"),
    ]

    prompt = template_prompt

    for template_column, target_column in replacements:
        template_value = str(template_row[template_column])
        target_value = str(target_row[target_column])

        if template_value not in prompt:
            raise RuntimeError(
                f"Could not find {template_column} in the training prompt. "
                "Do not create the stress test with a changed prompt format."
            )

        prompt = prompt.replace(template_value, target_value, 1)

    return prompt


def make_balanced_pairs(dataframe):
    selected_rows = []

    for _, group in dataframe.groupby("question_group_id", sort=True):
        supported_rows = group[
            group["label"] == "supported"
        ].sort_values("record_id")

        hallucinated_rows = group[
            group["label"] == "hallucinated"
        ].sort_values("record_id")

        if supported_rows.empty or hallucinated_rows.empty:
            continue

        selected_rows.append(supported_rows.iloc[0].to_dict())
        selected_rows.append(hallucinated_rows.iloc[0].to_dict())

    pairs = pd.DataFrame(selected_rows)

    if pairs.empty:
        raise RuntimeError("No balanced Fake question pairs were found.")

    return pairs.reset_index(drop=True)


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Fake source data was not found: {SOURCE_PATH}")

    if not CONTROLLED_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Controlled training benchmark was not found: "
            f"{CONTROLLED_TRAIN_PATH}"
        )

    print("Loading Med-HALT Reasoning Fake source...")
    fake_dataframe = pd.read_csv(SOURCE_PATH)

    required_columns = {
        "record_id",
        "dataset_name",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
        "case_type",
        "source_split",
        "source_file",
    }

    missing_columns = required_columns - set(fake_dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing source columns: {sorted(missing_columns)}")

    fake_dataframe = fake_dataframe.copy()
    fake_dataframe["record_id"] = fake_dataframe["record_id"].astype(str)
    fake_dataframe["label"] = fake_dataframe["is_hallucinated"].map(get_label)
    fake_dataframe["question_group_id"] = fake_dataframe["question"].map(
        make_group_id
    )

    print("Loading controlled training benchmark...")
    controlled_train_records = load_jsonl(CONTROLLED_TRAIN_PATH)

    fake_train_records = [
        record
        for record in controlled_train_records
        if record["dataset_name"] == "medhalt_reasoning_fake"
    ]

    if not fake_train_records:
        raise RuntimeError(
            "No Med-HALT Reasoning Fake records were found in controlled training."
        )

    used_record_ids = {
        str(record["source_record_id"])
        for record in fake_train_records
    }

    template_record = fake_train_records[0]

    template_source_rows = fake_dataframe[
        fake_dataframe["record_id"]
        == str(template_record["source_record_id"])
    ]

    if template_source_rows.empty:
        raise RuntimeError(
            "Could not match the prompt template record back to Fake source data."
        )

    template_source_row = template_source_rows.iloc[0].to_dict()

    # Remove every question group represented in Qwen training.
    used_question_groups = set(
        fake_dataframe.loc[
            fake_dataframe["record_id"].isin(used_record_ids),
            "question_group_id",
        ]
    )

    unused_fake_rows = fake_dataframe[
        ~fake_dataframe["question_group_id"].isin(used_question_groups)
    ].copy()

    unused_pairs = make_balanced_pairs(unused_fake_rows)

    available_groups = sorted(unused_pairs["question_group_id"].unique())

    if len(available_groups) < QUESTION_GROUPS:
        raise ValueError(
            f"Need {QUESTION_GROUPS} unused Fake question groups, but found "
            f"{len(available_groups)}."
        )

    selected_groups = (
        pd.Series(available_groups)
        .sample(
            n=QUESTION_GROUPS,
            random_state=SEED,
            replace=False,
        )
        .tolist()
    )

    stress_test = unused_pairs[
        unused_pairs["question_group_id"].isin(selected_groups)
    ].copy()

    stress_test = stress_test.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    label_counts = stress_test["label"].value_counts().to_dict()

    if (
        len(stress_test) != QUESTION_GROUPS * 2
        or label_counts.get("supported", 0) != QUESTION_GROUPS
        or label_counts.get("hallucinated", 0) != QUESTION_GROUPS
    ):
        raise RuntimeError("Stress test is not balanced as expected.")

    output_records = []

    for row_number, (_, row) in enumerate(stress_test.iterrows(), start=1):
        row_data = row.to_dict()

        prompt = build_prompt_from_training_template(
            template_record["prompt"],
            template_source_row,
            row_data,
        )

        output_records.append(
            {
                "id": f"fake_stress_{row_number:05d}",
                "label": row_data["label"],
                "prompt": prompt,
                "dataset_name": row_data["dataset_name"],
                "source_record_id": row_data["record_id"],
                "source_split": row_data["source_split"],
                "source_file": row_data["source_file"],
                "case_type": row_data["case_type"],
                "question_group_id": row_data["question_group_id"],
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for record in output_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark_name": "Unseen Med-HALT Reasoning Fake Stress Test",
        "seed": SEED,
        "rows": len(output_records),
        "question_groups": QUESTION_GROUPS,
        "labels": {
            "supported": label_counts["supported"],
            "hallucinated": label_counts["hallucinated"],
        },
        "source_policy": (
            "Only Med-HALT Reasoning Fake question groups excluded from the "
            "controlled Qwen training set were eligible."
        ),
        "leakage_control": (
            "Every Fake question group used during Qwen training was removed "
            "before this fixed stress-test subset was selected."
        ),
        "prompt_policy": (
            "The exact controlled-training prompt structure was reused; only "
            "the question, evidence, and candidate answer content changed."
        ),
        "output_file": str(OUTPUT_PATH.relative_to(PROJECT_DIR)),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nUNSEEN REASONING FAKE STRESS TEST FROZEN")
    print("Rows:", len(output_records))
    print("Question groups:", QUESTION_GROUPS)
    print("Labels:", label_counts)
    print("Training Fake source records excluded:", len(used_record_ids))
    print("Stress-test file:", OUTPUT_PATH)
    print("Manifest:", MANIFEST_PATH)
    print(
        "\nDo not use this stress test for LoRA training, checkpoint "
        "selection, or calibration."
    )


if __name__ == "__main__":
    main()