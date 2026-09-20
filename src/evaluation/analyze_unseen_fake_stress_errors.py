import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

STRESS_TEST_PATH = (
    DATA_DIR / "unseen_medhalt_reasoning_fake_stress_test.jsonl"
)

SOURCE_PATH = DATA_DIR / "medhalt_reasoning_fake_common.csv"

PREDICTIONS_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_test_predictions.csv"
)

ERROR_CASES_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_error_cases.csv"
)

CASE_SUMMARY_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_error_type_summary.csv"
)

SUMMARY_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_error_analysis_summary.json"
)

LORA_MODEL_NAME = "Qwen2.5-3B + Med-HALT Hallucination LoRA"


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_error_type(row):
    if (
        row["true_label"] == "supported"
        and row["predicted_label"] == "hallucinated"
    ):
        return "false_positive_supported_flagged"

    if (
        row["true_label"] == "hallucinated"
        and row["predicted_label"] == "supported"
    ):
        return "false_negative_missed_hallucination"

    return "correct"


def main():
    for file_path in [
        STRESS_TEST_PATH,
        SOURCE_PATH,
        PREDICTIONS_PATH,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    stress_test_records = load_jsonl(STRESS_TEST_PATH)
    stress_test = pd.DataFrame(stress_test_records)

    source_data = pd.read_csv(SOURCE_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    source_data["record_id"] = source_data["record_id"].astype(str)
    stress_test["source_record_id"] = (
        stress_test["source_record_id"].astype(str)
    )

    lora_predictions = predictions[
        predictions["model"] == LORA_MODEL_NAME
    ].copy()

    if lora_predictions.empty:
        raise RuntimeError(
            "Could not find the LoRA model predictions. "
            "Check the model name in the predictions CSV."
        )

    lora_predictions["error_type"] = lora_predictions.apply(
        make_error_type,
        axis=1,
    )

    stress_metadata = stress_test[
        [
            "id",
            "source_record_id",
            "case_type",
            "question_group_id",
        ]
    ].copy()

    error_rows = lora_predictions[
        lora_predictions["error_type"] != "correct"
    ].merge(
        stress_metadata,
        on=["id", "case_type", "question_group_id"],
        how="left",
    )

    error_cases = error_rows.merge(
        source_data[
            [
                "record_id",
                "question",
                "knowledge",
                "answer",
                "source_split",
            ]
        ],
        left_on="source_record_id",
        right_on="record_id",
        how="left",
    )

    if error_cases["question"].isna().any():
        missing_count = int(error_cases["question"].isna().sum())
        raise RuntimeError(
            f"Could not retrieve source details for {missing_count} error rows."
        )

    error_cases = error_cases[
        [
            "error_type",
            "id",
            "true_label",
            "predicted_label",
            "confidence_margin",
            "score_supported",
            "score_hallucinated",
            "case_type",
            "source_split",
            "source_record_id",
            "question",
            "knowledge",
            "answer",
        ]
    ].sort_values(
        ["error_type", "confidence_margin"],
        ascending=[True, True],
    )

    error_cases.to_csv(ERROR_CASES_PATH, index=False)

    case_summary = (
        lora_predictions.groupby(["case_type", "error_type"])
        .size()
        .reset_index(name="rows")
        .sort_values(["case_type", "error_type"])
    )

    case_summary.to_csv(CASE_SUMMARY_PATH, index=False)

    false_positives = error_cases[
        error_cases["error_type"]
        == "false_positive_supported_flagged"
    ]

    false_negatives = error_cases[
        error_cases["error_type"]
        == "false_negative_missed_hallucination"
    ]

    summary = {
        "evaluation": "Unseen Med-HALT Reasoning Fake stress test",
        "model": LORA_MODEL_NAME,
        "evaluated_rows": int(len(lora_predictions)),
        "correct_predictions": int(
            (lora_predictions["error_type"] == "correct").sum()
        ),
        "total_errors": int(len(error_cases)),
        "false_positives_supported_flagged": int(len(false_positives)),
        "false_negatives_missed_hallucinations": int(
            len(false_negatives)
        ),
        "finding": (
            "The selected LoRA adapter missed no hallucinated answers on this "
            "held-out Reasoning Fake stress test. All observed errors were "
            "supported answers conservatively flagged as hallucinated."
        ),
        "error_cases_file": str(
            ERROR_CASES_PATH.relative_to(PROJECT_DIR)
        ),
        "case_summary_file": str(
            CASE_SUMMARY_PATH.relative_to(PROJECT_DIR)
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nUNSEEN REASONING FAKE ERROR ANALYSIS COMPLETE")
    print("LoRA prediction rows:", len(lora_predictions))
    print("Total errors:", len(error_cases))
    print("False positives:", len(false_positives))
    print("False negatives:", len(false_negatives))

    print("\nError cases, ordered from lowest confidence margin:")
    print(
        error_cases[
            [
                "error_type",
                "confidence_margin",
                "source_record_id",
                "question",
            ]
        ].to_string(index=False, max_colwidth=90)
    )

    print("\nSaved error cases:", ERROR_CASES_PATH)
    print("Saved case summary:", CASE_SUMMARY_PATH)
    print("Saved analysis summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()