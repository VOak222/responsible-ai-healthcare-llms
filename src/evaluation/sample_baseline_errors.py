from pathlib import Path

import pandas as pd


BASELINE_DIR = Path("results/baseline")
OUTPUT_PATH = BASELINE_DIR / "manual_error_review_sample.csv"
SAMPLE_SIZE_PER_ERROR_TYPE = 5


def main():
    false_positives = pd.read_csv(BASELINE_DIR / "false_positives.csv")
    false_negatives = pd.read_csv(BASELINE_DIR / "false_negatives.csv")

    fp_sample = false_positives.sample(
        n=min(SAMPLE_SIZE_PER_ERROR_TYPE, len(false_positives)),
        random_state=42,
    )
    fp_sample["error_type"] = "false_positive"

    fn_sample = false_negatives.sample(
        n=min(SAMPLE_SIZE_PER_ERROR_TYPE, len(false_negatives)),
        random_state=42,
    )
    fn_sample["error_type"] = "false_negative"

    review_columns = [
        "error_type",
        "record_id",
        "source_dataset",
        "difficulty_level",
        "hallucination_category",
        "is_hallucinated",
        "predicted_label",
        "hallucination_probability",
        "question",
        "answer",
    ]

    review_sample = pd.concat([fp_sample, fn_sample], ignore_index=True)
    review_sample = review_sample[review_columns]
    review_sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved manual review sample to {OUTPUT_PATH}")
    print(f"Rows for review: {len(review_sample)}")


if __name__ == "__main__":
    main()