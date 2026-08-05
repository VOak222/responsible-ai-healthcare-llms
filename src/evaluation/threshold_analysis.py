from pathlib import Path
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


PREDICTIONS_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS_PATH)

y_true = df["is_hallucinated"]
y_proba = df["hallucination_probability"]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []

for threshold in thresholds:
    # If probability is greater than or equal to the threshold,
    # classify the answer as hallucinated.
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })

results = pd.DataFrame(rows)

print("Threshold Analysis:")
print(results)

results_path = OUTPUT_DIR / "threshold_analysis.csv"
summary_path = OUTPUT_DIR / "threshold_analysis_summary.txt"

results.to_csv(results_path, index=False)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Threshold Analysis for Baseline Hallucination Detector\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nInterpretation:\n")
    f.write(
        "Lowering the threshold usually increases recall and reduces false negatives, "
        "but it can also increase false positives. In healthcare hallucination detection, "
        "false negatives are especially important because they allow hallucinated medical "
        "answers to pass as safe or grounded.\n"
    )

print(f"\nSaved threshold analysis to {results_path}")
print(f"Saved summary to {summary_path}")