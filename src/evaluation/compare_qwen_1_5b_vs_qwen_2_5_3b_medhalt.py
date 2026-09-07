from pathlib import Path
import pandas as pd

QWEN_15_PATH = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
QWEN_3B_PATH = Path("results/qwen_medhalt/qwen_2_5_3b_medhalt_100row_corrected_predictions.csv")

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports/future_work")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "qwen_1_5b_vs_qwen_2_5_3b_medhalt_comparison.csv"
DISAGREEMENT_PATH = OUT_DIR / "qwen_1_5b_vs_qwen_2_5_3b_disagreements.csv"
FN_PATH = OUT_DIR / "qwen_model_false_negative_comparison.csv"
REPORT_PATH = REPORT_DIR / "qwen_1_5b_vs_qwen_2_5_3b_medhalt_comparison_report.md"


def pick_col(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {label}. Tried {candidates}. Available columns: {list(df.columns)}")


def metrics(df, pred_col, label_col):
    valid = df[df[pred_col].isin([0, 1])].copy()

    y_true = valid[label_col].astype(int)
    y_pred = valid[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = round((tp + tn) / max(len(valid), 1), 3)
    precision = round(tp / max(tp + fp, 1), 3)
    recall = round(tp / max(tp + fn, 1), 3)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-9), 3)

    return {
        "rows": len(valid),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def md_table(df):
    if df.empty:
        return "_No rows available._"

    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    q15 = pd.read_csv(QWEN_15_PATH)
    q3 = pd.read_csv(QWEN_3B_PATH)

    q15_pred_col = pick_col(q15, ["prediction", "predicted_label", "model_prediction"], "Qwen 1.5B prediction")
    label_col = pick_col(q15, ["is_hallucinated", "true_label", "label"], "true label")

    q15 = q15.copy()
    q15["qwen_1_5b_prediction"] = q15[q15_pred_col].astype(int)
    q15["is_hallucinated"] = q15[label_col].astype(int)

    q3 = q3.copy()
    q3["qwen_2_5_3b_prediction"] = q3["qwen_2_5_3b_prediction"].astype(int)
    q3["is_hallucinated"] = q3["is_hallucinated"].astype(int)

    summary = pd.DataFrame([
        {
            "model": "Qwen 1.5B",
            **metrics(q15, "qwen_1_5b_prediction", "is_hallucinated"),
            "interpretation": "Best main model so far because it balances recall and fewer false positives",
        },
        {
            "model": "Qwen2.5-3B",
            **metrics(q3, "qwen_2_5_3b_prediction", "is_hallucinated"),
            "interpretation": "Strong recall but too many false positives; useful as secondary safety reviewer",
        },
    ])

    keep_cols = ["record_id", "question", "knowledge", "answer", "is_hallucinated", "qwen_1_5b_prediction"]
    keep_cols = [col for col in keep_cols if col in q15.columns]

    merged = q15[keep_cols].merge(
        q3[["record_id", "qwen_2_5_3b_prediction", "qwen_2_5_3b_raw_response"]],
        on="record_id",
        how="inner"
    )

    disagreements = merged[
        merged["qwen_1_5b_prediction"] != merged["qwen_2_5_3b_prediction"]
    ].copy()

    false_negatives = merged[
        (merged["is_hallucinated"].astype(int) == 1)
        & (
            (merged["qwen_1_5b_prediction"].astype(int) == 0)
            | (merged["qwen_2_5_3b_prediction"].astype(int) == 0)
        )
    ].copy()

    summary.to_csv(SUMMARY_PATH, index=False)
    disagreements.to_csv(DISAGREEMENT_PATH, index=False)
    false_negatives.to_csv(FN_PATH, index=False)

    same_false_negative = "unknown"
    q15_fn_ids = set(
        merged[
            (merged["is_hallucinated"].astype(int) == 1)
            & (merged["qwen_1_5b_prediction"].astype(int) == 0)
        ]["record_id"].astype(str)
    )
    q3_fn_ids = set(
        merged[
            (merged["is_hallucinated"].astype(int) == 1)
            & (merged["qwen_2_5_3b_prediction"].astype(int) == 0)
        ]["record_id"].astype(str)
    )

    if q15_fn_ids == q3_fn_ids:
        same_false_negative = "yes"
    else:
        same_false_negative = "no"

    report = f"""# Qwen 1.5B vs Qwen2.5-3B Med-HALT Comparison

## Purpose

This comparison checks whether Qwen2.5-3B should replace Qwen 1.5B as the main local open-source model for healthcare hallucination detection.

Both models are compared on the same 100-row Med-HALT balanced sample.

## Model Comparison

{md_table(summary)}

## Key Finding

Qwen2.5-3B successfully ran locally and produced parseable outputs for all rows.

However, it should not replace Qwen 1.5B as the main model yet. Qwen2.5-3B has strong recall, but it produces many more false positives. This means it is safer than a weak model, but less balanced than Qwen 1.5B.

## False Negative Check

| Check | Result |
|---|---|
| Qwen 1.5B false negative rows | {len(q15_fn_ids)} |
| Qwen2.5-3B false negative rows | {len(q3_fn_ids)} |
| Same false negative row | {same_false_negative} |

## Disagreement Cases

The two models disagreed on {len(disagreements)} rows.

These rows are useful for future manual review because disagreement often highlights uncertain cases.

## Project Decision

Qwen 1.5B remains the best main model so far.

Qwen2.5-3B should be kept as a future-work candidate or secondary safety reviewer because it is conservative and catches hallucinations well, but it over-flags supported responses.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Qwen 1.5B vs Qwen2.5-3B comparison complete")
    print()
    print(summary.to_string(index=False))
    print()
    print("Disagreement rows:", len(disagreements))
    print("Qwen 1.5B false negative rows:", len(q15_fn_ids))
    print("Qwen2.5-3B false negative rows:", len(q3_fn_ids))
    print("Same false negative row:", same_false_negative)
    print()
    print("Saved:", SUMMARY_PATH)
    print("Saved:", DISAGREEMENT_PATH)
    print("Saved:", FN_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
