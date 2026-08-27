from pathlib import Path
import pandas as pd

INPUT_PATH = Path(
    "results/qwen_medhalt/"
    "qwen_medhalt_balanced_sample_predictions.csv"
)

OUTPUT_DIR = Path("results/qwen_medhalt")
GROUP_PATH = (
    OUTPUT_DIR
    / "qwen_medhalt_probability_by_group.csv"
)
ERROR_PATH = (
    OUTPUT_DIR
    / "qwen_medhalt_error_analysis.csv"
)
SHORTLIST_PATH = (
    OUTPUT_DIR
    / "qwen_medhalt_error_review_shortlist.csv"
)
REPORT_PATH = Path(
    "reports/qwen_medhalt_error_analysis.md"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    df["is_hallucinated"] = (
        df["is_hallucinated"].astype(int)
    )
    df["predicted_label"] = (
        df["predicted_label"].astype(int)
    )

    df["error_type"] = "correct"

    df.loc[
        (df["is_hallucinated"] == 0)
        & (df["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"

    df.loc[
        (df["is_hallucinated"] == 1)
        & (df["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    group_summary = (
        df.groupby(
            ["dataset_name", "is_hallucinated"]
        )
        .agg(
            rows=("record_id", "size"),
            mean_hallucination_probability=(
                "hallucination_probability",
                "mean",
            ),
            median_hallucination_probability=(
                "hallucination_probability",
                "median",
            ),
            minimum_probability=(
                "hallucination_probability",
                "min",
            ),
            maximum_probability=(
                "hallucination_probability",
                "max",
            ),
            predicted_hallucination_rate=(
                "predicted_label",
                "mean",
            ),
            accuracy=(
                "is_correct_prediction",
                "mean",
            ),
        )
        .reset_index()
    )

    group_summary.to_csv(
        GROUP_PATH,
        index=False,
    )

    errors = df[
        df["error_type"] != "correct"
    ].copy()

    errors = errors.sort_values(
        [
            "error_type",
            "prediction_confidence",
        ],
        ascending=[True, False],
    )

    errors.to_csv(
        ERROR_PATH,
        index=False,
    )

    shortlist_parts = []

    for (
        dataset_name,
        error_type,
    ), group in errors.groupby(
        ["dataset_name", "error_type"]
    ):
        shortlist_parts.append(
            group.head(5)
        )

    shortlist = pd.concat(
        shortlist_parts,
        ignore_index=True,
    )

    shortlist_columns = [
        "record_id",
        "dataset_name",
        "case_type",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
        "predicted_label",
        "hallucination_probability",
        "prediction_confidence",
        "error_type",
    ]

    shortlist[shortlist_columns].to_csv(
        SHORTLIST_PATH,
        index=False,
    )

    error_counts = (
        df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="rows")
    )

    report = f"""# Qwen Med-HALT Error Analysis

## Error Counts

{error_counts.to_markdown(index=False)}

## Probability Behavior

{group_summary.to_markdown(index=False)}

## Interpretation

Qwen2.5-0.5B-Instruct strongly favored the supported label. It predicted only 6 of 50 hallucinated answers as hallucinated.

The combined ROC-AUC was approximately 0.51, indicating that the current zero-shot numeric-label prompt provides almost no reliable separation between supported and hallucinated answers.

## Next Step

Run a smaller prompt-robustness experiment using descriptive labels and few-shot examples before deciding whether the model is unsuitable for hallucination classification.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Error Counts:")
    print(error_counts.to_string(index=False))
    print()
    print("Probability Behavior:")
    print(group_summary.to_string(index=False))
    print()
    print("Saved:", GROUP_PATH)
    print("Saved:", ERROR_PATH)
    print("Saved:", SHORTLIST_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
