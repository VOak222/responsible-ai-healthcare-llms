from pathlib import Path
import pandas as pd

PREDICTIONS_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS_PATH)

print("Overall prediction correctness:")
print(df["is_correct_prediction"].value_counts())

print("\nAccuracy by difficulty level:")
print(df.groupby("difficulty_level")["is_correct_prediction"].mean().sort_values())

print("\nAccuracy by hallucination category:")
category_accuracy = (
    df.groupby("hallucination_category")["is_correct_prediction"]
    .agg(["count", "mean"])
    .sort_values("mean")
)
print(category_accuracy)

false_positives = df[
    (df["is_hallucinated"] == 0) & (df["predicted_label"] == 1)
]

false_negatives = df[
    (df["is_hallucinated"] == 1) & (df["predicted_label"] == 0)
]

false_positives.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
false_negatives.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)
category_accuracy.to_csv(OUTPUT_DIR / "category_accuracy.csv")

print("\nSaved error analysis files to results/baseline/")
print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))