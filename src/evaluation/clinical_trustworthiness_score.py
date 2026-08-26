from pathlib import Path
import pandas as pd

BASELINE_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
CLAIM_PATH = Path("results/grounding/claim_level_grounding_scores.csv")
SEMANTIC_PATH = Path("results/grounding/semantic_claim_verification_scores.csv")
OUTPUT_DIR = Path("results/trustworthiness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))

def action_from_score(row):
    score = row["clinical_trustworthiness_score"]
    high_risk = row["unsupported_high_impact_claim_count"] > 0
    contradiction_risk = row["contradiction_risk"] == 1

    if high_risk or contradiction_risk or score < 0.45:
        return "mandatory_human_review"
    if score < 0.65:
        return "human_review"
    if score < 0.80:
        return "revise"
    return "accept"

def main():
    baseline_df = require_csv(BASELINE_PATH)
    claim_df = require_csv(CLAIM_PATH)

    if SEMANTIC_PATH.exists():
        semantic_df = pd.read_csv(SEMANTIC_PATH)
        grounding_df = claim_df.merge(
            semantic_df[[
                "record_id",
                "semantic_answer_grounding_score",
                "combined_answer_grounding_score",
                "semantic_unsupported_claim_count",
                "combined_unsupported_claim_count"
            ]],
            on="record_id",
            how="left"
        )
        grounding_df["final_grounding_score"] = grounding_df["combined_answer_grounding_score"].fillna(
            grounding_df["answer_claim_grounding_score"]
        )
        grounding_df["final_unsupported_claim_count"] = grounding_df["combined_unsupported_claim_count"].fillna(
            grounding_df["unsupported_claim_count"]
        )
        grounding_source = "semantic_plus_claim_grounding"
    else:
        grounding_df = claim_df.copy()
        grounding_df["final_grounding_score"] = grounding_df["answer_claim_grounding_score"]
        grounding_df["final_unsupported_claim_count"] = grounding_df["unsupported_claim_count"]
        grounding_source = "claim_grounding_only"

    df = baseline_df.merge(grounding_df, on="record_id", how="left", suffixes=("", "_grounding"))

    if "hallucination_probability" in df.columns:
        df["model_hallucination_risk"] = df["hallucination_probability"]
    else:
        df["model_hallucination_risk"] = df["predicted_label"]

    df["grounding_risk"] = 1 - df["final_grounding_score"].fillna(0.0)

    df["unsupported_high_impact_claim_count"] = df.get(
        "unsupported_high_impact_claim_count", 0
    )

    df["contradiction_risk"] = (
        (df["predicted_label"].astype(int) == 0) &
        (df["final_unsupported_claim_count"].fillna(0) > 0)
    ).astype(int)

    df["clinical_trustworthiness_score"] = df.apply(
        lambda row: clamp(
            1
            - (0.45 * row["model_hallucination_risk"])
            - (0.35 * row["grounding_risk"])
            - (0.15 * min(row["final_unsupported_claim_count"], 3) / 3)
            - (0.05 * row["contradiction_risk"])
        ),
        axis=1
    )

    df["recommended_action"] = df.apply(action_from_score, axis=1)

    output_cols = [
        "record_id",
        "is_hallucinated",
        "predicted_label",
        "model_hallucination_risk",
        "final_grounding_score",
        "final_unsupported_claim_count",
        "unsupported_high_impact_claim_count",
        "contradiction_risk",
        "clinical_trustworthiness_score",
        "recommended_action"
    ]

    df[output_cols].to_csv(OUTPUT_DIR / "clinical_trustworthiness_scores.csv", index=False)

    action_summary = (
        df["recommended_action"]
        .value_counts()
        .rename_axis("recommended_action")
        .reset_index(name="row_count")
    )
    action_summary.to_csv(OUTPUT_DIR / "clinical_trustworthiness_action_summary.csv", index=False)

    overall = pd.DataFrame([{
        "rows_scored": len(df),
        "grounding_source": grounding_source,
        "mean_clinical_trustworthiness_score": df["clinical_trustworthiness_score"].mean(),
        "mean_hallucination_risk": df["model_hallucination_risk"].mean(),
        "mean_final_grounding_score": df["final_grounding_score"].mean(),
        "unsupported_high_impact_claim_rows": int((df["unsupported_high_impact_claim_count"] > 0).sum()),
        "contradiction_risk_rows": int(df["contradiction_risk"].sum()),
        "mandatory_human_review_rows": int((df["recommended_action"] == "mandatory_human_review").sum())
    }])
    overall.to_csv(OUTPUT_DIR / "clinical_trustworthiness_overall.csv", index=False)

    print("Clinical Trustworthiness Overall:")
    print(overall.to_string(index=False))
    print()
    print("Recommended Action Summary:")
    print(action_summary.to_string(index=False))
    print()
    print("Saved trustworthiness outputs to results/trustworthiness")

if __name__ == "__main__":
    main()

