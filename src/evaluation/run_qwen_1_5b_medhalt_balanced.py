from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.run_qwen_1_5b_medhalt_fewshot_pilot import (
    MODEL_ID,
    calculate_metrics,
    classify_row,
    markdown_table,
)


INPUT_PATH = Path(
    "results/qwen_medhalt/qwen_medhalt_balanced_sample_predictions.csv"
)

OUT_DIR = Path("results/qwen_medhalt")

PREDICTIONS_PATH = (
    OUT_DIR / "qwen_1_5b_medhalt_balanced_predictions.csv"
)

SUMMARY_PATH = (
    OUT_DIR / "qwen_1_5b_medhalt_balanced_summary.csv"
)

REPORT_PATH = Path(
    "reports/qwen_1_5b_medhalt_balanced_summary.md"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "dataset_name",
        "record_id",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    source_df = (
        source_df
        .drop_duplicates(subset=["record_id"])
        .reset_index(drop=True)
    )

    if PREDICTIONS_PATH.exists():
        existing_df = pd.read_csv(PREDICTIONS_PATH)
        rows = existing_df.to_dict("records")
        completed_record_ids = set(
            existing_df["record_id"].astype(str)
        )
        print(
            "Resuming from completed rows:",
            len(completed_record_ids),
        )
    else:
        rows = []
        completed_record_ids = set()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Model:", MODEL_ID)
    print("Device:", device)
    print("Total rows:", len(source_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    for index, source_row in source_df.iterrows():
        record_id = str(source_row["record_id"])

        if record_id in completed_record_ids:
            continue

        print(
            f"Evaluating {index + 1}/{len(source_df)}: "
            f"{source_row['dataset_name']} "
            f"true_label={source_row['is_hallucinated']}"
        )

        classification = classify_row(
            tokenizer,
            model,
            device,
            source_row,
            "zero_shot_numeric",
        )

        result = {
            "model_id": MODEL_ID,
            "dataset_name": source_row["dataset_name"],
            "record_id": source_row["record_id"],
            "question": source_row["question"],
            "knowledge": source_row["knowledge"],
            "answer": source_row["answer"],
            "is_hallucinated": int(
                source_row["is_hallucinated"]
            ),
            **classification,
        }

        result["is_correct_prediction"] = (
            result["predicted_label"]
            == result["is_hallucinated"]
        )

        rows.append(result)
        completed_record_ids.add(record_id)

        pd.DataFrame(rows).to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

        print(
            "prediction="
            f"{result['predicted_label']} "
            "probability="
            f"{result['hallucination_probability']:.4f}"
        )

    results_df = pd.DataFrame(rows)

    summary_rows = []

    for dataset_name, dataset_df in results_df.groupby(
        "dataset_name"
    ):
        summary_rows.append(
            calculate_metrics(dataset_df, dataset_name)
        )

    summary_rows.append(
        calculate_metrics(results_df, "combined")
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        "dataset_name"
    ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)

    combined = summary_df[
        summary_df["dataset_name"] == "combined"
    ].iloc[0]

    report = f"""# Qwen 1.5B Med-HALT Balanced Evaluation

## Evaluation

Qwen/Qwen2.5-1.5B-Instruct was evaluated on the same balanced 100-row Med-HALT sample previously used for the 0.5B model.

The selected prompt was the zero-shot numerical prompt because it produced the strongest pilot accuracy, F1 score, and ROC-AUC.

## Results

{markdown_table(summary_df)}

## Combined Result

- Rows evaluated: {int(combined["rows"])}
- Accuracy: {combined["accuracy"]:.4f}
- Precision: {combined["precision"]:.4f}
- Recall: {combined["recall"]:.4f}
- F1 score: {combined["f1_score"]:.4f}
- ROC-AUC: {combined["roc_auc"]:.4f}
- Predicted hallucination rate: {combined["predicted_hallucination_rate"]:.4f}

## Interpretation

This 100-row evaluation tests whether the strong 20-row pilot result generalizes to a larger balanced sample. The results should be compared directly with the earlier Qwen 0.5B evaluation, which achieved 0.52 accuracy, 0.20 F1, and 0.5096 ROC-AUC.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Qwen 1.5B Med-HALT Results:")
    print(summary_df.to_string(index=False))
    print()
    print("Saved:", PREDICTIONS_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
