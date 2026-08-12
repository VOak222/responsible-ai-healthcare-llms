from pathlib import Path
import pandas as pd

BASELINE_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
CLAIM_GROUNDING_PATH = Path("results/grounding/claim_level_grounding_scores.csv")
SEMANTIC_PATH = Path("results/grounding/semantic_claim_verification_scores.csv")
TRUSTWORTHINESS_PATH = Path("results/trustworthiness/clinical_trustworthiness_scores.csv")
OUTPUT_DIR = Path("results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def safe_mean(df, column):
    if column not in df.columns:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").mean())

def safe_sum(df, column):
    if column not in df.columns:
        return None
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())

def binary_metrics(df):
    actual = df["is_hallucinated"].astype(int)
    predicted = df["predicted_label"].astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }

def main():
    baseline_df = require_csv(BASELINE_PATH)
    claim_df = require_csv(CLAIM_GROUNDING_PATH)
    trust_df = require_csv(TRUSTWORTHINESS_PATH)
    semantic_df = pd.read_csv(SEMANTIC_PATH) if SEMANTIC_PATH.exists() else None

    metrics = binary_metrics(baseline_df)
    actions = trust_df["recommended_action"].value_counts().to_dict()

    comparison_rows = [
        {
            "component": "TF-IDF + Logistic Regression",
            "role_in_pipeline": "Predicts whether the full answer is hallucinated",
            "main_metric": "F1 score",
            "main_value": metrics["f1_score"],
            "secondary_metric": "Accuracy",
            "secondary_value": metrics["accuracy"],
            "risk_signal": "False negatives",
            "risk_value": metrics["false_negatives"],
            "interpretation": "Good first baseline, but missed hallucinated healthcare answers still require extra review logic.",
        },
        {
            "component": "Claim-level grounding",
            "role_in_pipeline": "Breaks answers into claims and checks support from evidence",
            "main_metric": "Mean grounding score",
            "main_value": safe_mean(claim_df, "answer_claim_grounding_score"),
            "secondary_metric": "Unsupported claims",
            "secondary_value": safe_sum(claim_df, "unsupported_claim_count"),
            "risk_signal": "Unsupported high-impact claims",
            "risk_value": safe_sum(claim_df, "unsupported_high_impact_claim_count"),
            "interpretation": "Adds evidence-level checking so the system is not only relying on the classifier label.",
        },
    ]

    if semantic_df is not None:
        comparison_rows.append({
            "component": "Semantic claim verification",
            "role_in_pipeline": "Compares each claim with the closest evidence sentence",
            "main_metric": "Mean combined support score",
            "main_value": safe_mean(semantic_df, "combined_answer_grounding_score"),
            "secondary_metric": "Mean semantic support score",
            "secondary_value": safe_mean(semantic_df, "semantic_answer_grounding_score"),
            "risk_signal": "Combined unsupported claims",
            "risk_value": safe_sum(semantic_df, "combined_unsupported_claim_count"),
            "interpretation": "First semantic evaluator step. Current run may be strict because it uses TF-IDF fallback.",
        })

    comparison_rows.append({
        "component": "Clinical trustworthiness score",
        "role_in_pipeline": "Combines hallucination, grounding, safety, and confidence",
        "main_metric": "Mean trustworthiness score",
        "main_value": safe_mean(trust_df, "clinical_trustworthiness_score"),
        "secondary_metric": "Human review rows",
        "secondary_value": actions.get("human_review", 0) + actions.get("mandatory_human_review", 0),
        "risk_signal": "Mandatory human review rows",
        "risk_value": actions.get("mandatory_human_review", 0),
        "interpretation": "Turns model outputs into an operational decision: accept, revise, human review, or mandatory human review.",
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "model_evaluation_comparison.csv", index=False)

    overall_df = pd.DataFrame([{
        "rows_evaluated": len(baseline_df),
        "baseline_accuracy": metrics["accuracy"],
        "baseline_f1_score": metrics["f1_score"],
        "baseline_false_positives": metrics["false_positives"],
        "baseline_false_negatives": metrics["false_negatives"],
        "mean_claim_grounding_score": safe_mean(claim_df, "answer_claim_grounding_score"),
        "mean_trustworthiness_score": safe_mean(trust_df, "clinical_trustworthiness_score"),
        "accept_rows": actions.get("accept", 0),
        "revise_rows": actions.get("revise", 0),
        "human_review_rows": actions.get("human_review", 0),
        "mandatory_human_review_rows": actions.get("mandatory_human_review", 0),
    }])
    overall_df.to_csv(OUTPUT_DIR / "model_evaluation_overall.csv", index=False)

    action_summary = (
        trust_df["recommended_action"]
        .value_counts()
        .rename_axis("recommended_action")
        .reset_index(name="row_count")
    )
    action_summary.to_csv(OUTPUT_DIR / "trustworthiness_action_summary.csv", index=False)

    print("Model Evaluation Comparison:")
    print(comparison_df.to_string(index=False))
    print()
    print("Overall Metrics:")
    print(overall_df.to_string(index=False))
    print()
    print("Saved outputs to results/comparison")

if __name__ == "__main__":
    main()
