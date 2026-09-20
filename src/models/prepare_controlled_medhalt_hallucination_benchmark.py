import hashlib
import json
from pathlib import Path

import pandas as pd


SEED = 42

# Practical benchmark size for local Qwen LoRA training.
FCT_TRAIN_QUESTION_GROUPS = 1200
FAKE_TRAIN_QUESTION_GROUPS = 800
FCT_VALIDATION_QUESTION_GROUPS = 500
FCT_TEST_QUESTION_GROUPS = 500

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
CONFIG_DIR = PROJECT_DIR / "configs"

FCT_PATH = DATA_DIR / "medhalt_reasoning_fct_common.csv"
FAKE_PATH = DATA_DIR / "medhalt_reasoning_fake_common.csv"

TRAIN_PATH = DATA_DIR / "controlled_medhalt_hallucination_train.jsonl"
VALIDATION_PATH = DATA_DIR / "controlled_medhalt_hallucination_validation.jsonl"
TEST_PATH = DATA_DIR / "controlled_medhalt_hallucination_test.jsonl"

MANIFEST_PATH = (
    CONFIG_DIR / "controlled_medhalt_hallucination_benchmark_manifest.json"
)


def normalize_question(question):
    return " ".join(str(question).lower().split())


def make_question_group_id(dataset_name, question):
    text = f"{dataset_name}::{normalize_question(question)}"

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:20]


def make_prompt(row):
    return (
        "You are a healthcare-answer safety evaluator.\n"
        "Decide whether the candidate answer is supported by the supplied "
        "question and evidence.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Evidence:\n{row['knowledge']}\n\n"
        f"Candidate answer:\n{row['answer']}\n\n"
        "Is the candidate answer hallucinated?\n"
        "Reply with exactly one label: hallucinated or supported.\n"
        "Label:"
    )


def load_source_data(file_path):
    dataframe = pd.read_csv(file_path)

    required_columns = {
        "dataset_name",
        "record_id",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
        "case_type",
        "source_split",
        "source_file",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["question_group_id"] = dataframe.apply(
        lambda row: make_question_group_id(
            row["dataset_name"],
            row["question"],
        ),
        axis=1,
    )

    dataframe["label"] = dataframe["is_hallucinated"].map(
        {
            0: "supported",
            1: "hallucinated",
        }
    )

    if dataframe["label"].isna().any():
        raise ValueError(
            f"Unexpected is_hallucinated value in {file_path.name}"
        )

    return dataframe


def make_balanced_question_pairs(dataframe):
    """
    Keep one supported and one hallucinated answer per question group.

    This guarantees balanced labels and prevents answer variants of the same
    question from being separated across benchmark splits.
    """
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

    paired_dataframe = pd.DataFrame(selected_rows)

    if paired_dataframe.empty:
        raise RuntimeError(
            "No supported/hallucinated question pairs could be created."
        )

    return paired_dataframe.sort_values(
        ["question_group_id", "label", "record_id"]
    ).reset_index(drop=True)


def choose_question_groups(
    dataframe,
    number_of_groups,
    random_state,
    split_name,
):
    available_groups = sorted(
        dataframe["question_group_id"].unique()
    )

    if len(available_groups) < number_of_groups:
        raise ValueError(
            f"{split_name}: requested {number_of_groups} question groups, "
            f"but only {len(available_groups)} are available."
        )

    chosen_groups = (
        pd.Series(available_groups)
        .sample(
            n=number_of_groups,
            random_state=random_state,
            replace=False,
        )
        .tolist()
    )

    selected = dataframe[
        dataframe["question_group_id"].isin(chosen_groups)
    ].copy()

    return selected.sort_values(
        ["question_group_id", "label", "record_id"]
    ).reset_index(drop=True)


def remove_question_groups(dataframe, group_ids):
    return dataframe[
        ~dataframe["question_group_id"].isin(group_ids)
    ].copy()


def assert_no_question_leakage(named_splits):
    split_names = list(named_splits.keys())

    for left_index, left_name in enumerate(split_names):
        left_groups = set(
            named_splits[left_name]["question_group_id"].unique()
        )

        for right_name in split_names[left_index + 1 :]:
            right_groups = set(
                named_splits[right_name]["question_group_id"].unique()
            )

            shared_groups = left_groups & right_groups

            if shared_groups:
                raise RuntimeError(
                    f"Question leakage found between {left_name} and "
                    f"{right_name}: {len(shared_groups)} shared groups."
                )


def get_label_counts(dataframe):
    counts = dataframe["label"].value_counts().to_dict()

    return {
        "supported": int(counts.get("supported", 0)),
        "hallucinated": int(counts.get("hallucinated", 0)),
    }


def describe_split(dataframe):
    return {
        "rows": int(len(dataframe)),
        "question_groups": int(
            dataframe["question_group_id"].nunique()
        ),
        "labels": get_label_counts(dataframe),
        "dataset_rows": {
            str(dataset_name): int(count)
            for dataset_name, count in dataframe["dataset_name"]
            .value_counts()
            .sort_index()
            .items()
        },
        "original_source_splits": {
            str(source_split): int(count)
            for source_split, count in dataframe["source_split"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def build_output_records(dataframe, split_name):
    records = []

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        records.append(
            {
                "id": f"{split_name}_{row_number:05d}",
                "label": row["label"],
                "prompt": make_prompt(row),
                "dataset_name": row["dataset_name"],
                "source_record_id": row["record_id"],
                "source_split": row["source_split"],
                "source_file": row["source_file"],
                "case_type": row["case_type"],
                "question_group_id": row["question_group_id"],
            }
        )

    return records


def save_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def assert_balanced(split_name, dataframe):
    counts = get_label_counts(dataframe)

    if counts["supported"] != counts["hallucinated"]:
        raise RuntimeError(
            f"{split_name} is not balanced: {counts}"
        )


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Med-HALT Reasoning FCT...")
    fct_dataframe = load_source_data(FCT_PATH)

    print("Loading Med-HALT Reasoning Fake...")
    fake_dataframe = load_source_data(FAKE_PATH)

    fct_pairs = make_balanced_question_pairs(fct_dataframe)
    fake_pairs = make_balanced_question_pairs(fake_dataframe)

    fct_dev_pool = fct_pairs[
        fct_pairs["source_split"] == "dev"
    ].copy()

    fct_validation_pool = fct_pairs[
        fct_pairs["source_split"] == "val"
    ].copy()

    fct_test_pool = fct_pairs[
        fct_pairs["source_split"] == "test"
    ].copy()

    fake_train_pool = fake_pairs[
        fake_pairs["source_split"] == "train"
    ].copy()

    if (
        fct_dev_pool.empty
        or fct_validation_pool.empty
        or fct_test_pool.empty
        or fake_train_pool.empty
    ):
        raise RuntimeError(
            "Expected Med-HALT source partitions were not found."
        )

    # Protect every original FCT test question from train and validation,
    # even before selecting the fixed frozen-test subset.
    all_fct_test_groups = set(
        fct_test_pool["question_group_id"].unique()
    )

    fct_dev_pool = remove_question_groups(
        fct_dev_pool,
        all_fct_test_groups,
    )

    fct_validation_pool = remove_question_groups(
        fct_validation_pool,
        all_fct_test_groups,
    )

    # Training uses FCT dev plus Reasoning Fake train.
    fct_train = choose_question_groups(
        fct_dev_pool,
        FCT_TRAIN_QUESTION_GROUPS,
        SEED + 4,
        "FCT training",
    )

    fake_train = choose_question_groups(
        fake_train_pool,
        FAKE_TRAIN_QUESTION_GROUPS,
        SEED,
        "Reasoning Fake training",
    )

    train_split = pd.concat(
        [fct_train, fake_train],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # Validation and frozen test are FCT only. This makes checkpoint
    # selection match the final evaluation task family.
    validation_split = choose_question_groups(
        fct_validation_pool,
        FCT_VALIDATION_QUESTION_GROUPS,
        SEED + 2,
        "FCT validation",
    ).sample(
        frac=1,
        random_state=SEED + 1,
    ).reset_index(drop=True)

    # The frozen test comes only from original FCT test data.
    frozen_test_split = choose_question_groups(
        fct_test_pool,
        FCT_TEST_QUESTION_GROUPS,
        SEED + 3,
        "Frozen FCT test",
    )

    named_splits = {
        "training": train_split,
        "validation": validation_split,
        "frozen_test": frozen_test_split,
    }

    assert_no_question_leakage(named_splits)

    for split_name, dataframe in named_splits.items():
        assert_balanced(split_name, dataframe)

    save_jsonl(
        build_output_records(train_split, "train"),
        TRAIN_PATH,
    )

    save_jsonl(
        build_output_records(validation_split, "validation"),
        VALIDATION_PATH,
    )

    save_jsonl(
        build_output_records(frozen_test_split, "test"),
        TEST_PATH,
    )

    manifest = {
        "benchmark_name": (
            "Controlled Med-HALT Hallucination Detection Benchmark"
        ),
        "task_definition": (
            "Binary classification of a candidate healthcare answer as "
            "supported or hallucinated using the supplied question and "
            "evidence."
        ),
        "seed": SEED,
        "labels": {
            "supported": 0,
            "hallucinated": 1,
        },
        "source_policy": {
            "training": (
                "Med-HALT Reasoning FCT dev plus Med-HALT Reasoning Fake "
                "train question groups."
            ),
            "validation": (
                "Med-HALT Reasoning FCT val only, so checkpoint selection "
                "matches the same task family as the frozen FCT test."
            ),
            "frozen_test": (
                "A fixed, balanced, question-group-disjoint subset drawn "
                "only from the original Med-HALT Reasoning FCT test split."
            ),
        },
        "leakage_controls": [
            "All answer variants for the same normalized question stay in one split.",
            "No original FCT test question group is used for training or validation.",
            "Validation and frozen test are both FCT-only.",
            "The frozen test is selected deterministically with seed 42.",
            "Each selected question group contributes one supported and one hallucinated answer.",
        ],
        "requested_question_groups": {
            "fct_training": FCT_TRAIN_QUESTION_GROUPS,
            "fake_training": FAKE_TRAIN_QUESTION_GROUPS,
            "fct_validation": FCT_VALIDATION_QUESTION_GROUPS,
            "fct_frozen_test": FCT_TEST_QUESTION_GROUPS,
        },
        "splits": {
            split_name: describe_split(dataframe)
            for split_name, dataframe in named_splits.items()
        },
        "files": {
            "train": str(TRAIN_PATH.relative_to(PROJECT_DIR)),
            "validation": str(
                VALIDATION_PATH.relative_to(PROJECT_DIR)
            ),
            "frozen_test": str(TEST_PATH.relative_to(PROJECT_DIR)),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nCONTROLLED MED-HALT BENCHMARK FROZEN")

    for split_name, dataframe in named_splits.items():
        print(f"\n{split_name.replace('_', ' ').title()}:")
        print(describe_split(dataframe))

    print("\nManifest saved to:", MANIFEST_PATH)
    print(
        "\nDo not use controlled_medhalt_hallucination_test.jsonl "
        "for training, checkpoint selection, or calibration."
    )


if __name__ == "__main__":
    main()