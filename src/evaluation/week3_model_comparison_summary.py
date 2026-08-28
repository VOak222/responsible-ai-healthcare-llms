from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")
OUT_DIR = Path("results/comparison")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "week3_model_comparison_summary.csv"
REPORT_PATH = REPORT_DIR / "week3_model_comparison_summary.md"

QWEN_05_SUMMARY = Path("results/qwen_medhalt/qwen_medhalt_balanced_sample_summary.csv")
QWEN_15_PRED = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
QWEN_15_LOGIC = Path("results/trustworthiness/qwen_1_5b_logic_aware_routing_summary.csv")
QWEN_15_TRUST = Path("results/trustworthiness/qwen_1_5b_medhalt_trustworthiness_overall.csv")
BASELINE_TRUST = Path("results/trustworthiness/clinical_trustworthiness_overall.csv")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None


def classification_metrics(df):
    pred_col = pick_col(df, ["prediction", "predicted_label", "model_prediction"])
    label_col = pick_col(df, ["is_hallucinated", "true_label", "label"])

    if pred_col is None or label_col is None or df.empty:
        return {}

    y_true = df[label_col].astype(int)
    y_pred = df[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(df), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "rows": len(df),
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def get_qwen_05():
    df = read_csv(QWEN_05_SUMMARY)
    if df.empty:
        return None

    row = df[df.get("dataset_name", "") == "combined"]
    if row.empty:
        row = df.head(1)
    row = row.iloc[0]

    return {
        "model_or_method": "Qwen 0.5B",
        "dataset": "Med-HALT balanced sample",
        "rows": row.get("rows", ""),
        "accuracy": round(float(row.get("accuracy", 0)), 3),
        "f1_score": round(float(row.get("f1_score", 0)), 3),
        "recall": round(float(row.get("recall", 0)), 3),
        "main_result": "Too weak for main hallucination detection",
        "project_decision": "Do not use as primary detector",
    }


def get_qwen_15():
    pred = read_csv(QWEN_15_PRED)
    metrics = classification_metrics(pred)
    if not metrics:
        return None

    return {
        "model_or_method": "Qwen 1.5B",
        "dataset": "Med-HALT balanced sample",
        "rows": metrics["rows"],
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "recall": metrics["recall"],
        "main_result": "Strong improvement over Qwen 0.5B",
        "project_decision": "Keep validating with safety routing",
    }


def get_logic_result():
    df = read_csv(QWEN_15_LOGIC)
    if df.empty:
        return None

    row = df[df["action_column"] == "logic_aware_recommended_action"]
    if row.empty:
        return None
    row = row.iloc[0]

    return {
        "model_or_method": "Qwen 1.5B + logic-aware routing",
        "dataset": "Med-HALT balanced sample",
        "rows": 100,
        "accuracy": "",
        "f1_score": "",
        "recall": "",
        "main_result": f"Accepted hallucinated rows reduced to {int(row['accepted_hallucinated_rows'])}",
        "project_decision": "Use logic-aware routing for negative exam-style prompts",
    }


def get_baseline_trust():
    df = read_csv(BASELINE_TRUST)
    if df.empty:
        return None
    row = df.iloc[0]

    score_col = pick_col(df, [
        "mean_clinical_trustworthiness_score",
        "clinical_trustworthiness_score"
    ])

    score = round(float(row.get(score_col, 0)), 3) if score_col else ""

    return {
        "model_or_method": "TF-IDF baseline + trust score",
        "dataset": "Baseline healthcare evaluation set",
        "rows": int(row.get("rows_scored", 0)),
        "accuracy": "",
        "f1_score": "",
        "recall": "",
        "main_result": f"Mean trustworthiness score {score}",
        "project_decision": "Useful baseline reference",
    }


def md_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main():
    rows = []

    for item in [
        get_baseline_trust(),
        get_qwen_05(),
        get_qwen_15(),
        get_logic_result(),
    ]:
        if item:
            rows.append(item)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)

    report = f"""# Week 3 Model Comparison Summary

## What This Summary Shows

This summary connects the main model-evaluation results completed so far.

The project is no longer only checking model accuracy. It now compares hallucination detection, evidence grounding, trustworthiness scoring, and safety routing.

## Model Comparison

{md_table(summary)}

## Main Interpretation

Qwen 1.5B is clearly stronger than Qwen 0.5B on the Med-HALT balanced sample. The 0.5B model missed too many hallucinated answers, while the 1.5B model detected nearly all hallucinations in the current test.

However, the most important finding today was not only the higher accuracy. One hallucinated answer was still being accepted because the question used negative wording like “NOT correct.” The answer looked medically supported, but it was wrong for the question logic.

After adding logic-aware routing, accepted hallucinated rows dropped from 1 to 0.

## Project Decision

The current best approach is:

1. Keep Qwen 1.5B as the stronger local LLM candidate.
2. Keep semantic grounding as the evidence-checking layer.
3. Add logic-aware routing for tricky medical exam prompts.
4. Continue validating before treating Qwen 1.5B as final.

## Next Step

The next major step is to update the final evaluation layer so Med-HALT hallucination detection, semantic grounding, EquityMedQA fairness screening, and Qwen 1.5B logic-aware routing are shown together in one project-level trustworthiness view.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Week 3 model comparison summary complete")
    print()
    print(summary.to_string(index=False))
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
