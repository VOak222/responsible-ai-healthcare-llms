from pathlib import Path
import pandas as pd

BASELINE_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
TRUST_PATH = Path("results/trustworthiness/clinical_trustworthiness_scores.csv")
SEMANTIC_CLAIMS_PATH = Path("results/grounding/semantic_claim_verification_claims.csv")
OUTPUT_DIR = Path("results/manual_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def short_text(value, limit=350):
    text = str(value).replace("\n", " ").strip()
    return text[:limit] + "..." if len(text) > limit else text

def case_type(row):
    actual = int(row["is_hallucinated"])
    predicted = int(row["predicted_label"])

    if actual == 1 and predicted == 1:
        return "true_positive"
    if actual == 0 and predicted == 0:
        return "true_negative"
    if actual == 0 and predicted == 1:
        return "false_positive"
    return "false_negative"

def main():
    baseline_df = pd.read_csv(BASELINE_PATH)
    trust_df = pd.read_csv(TRUST_PATH)

    trust_cols = [
        "record_id",
        "clinical_trustworthiness_score",
        "recommended_action",
        "final_grounding_score",
        "final_unsupported_claim_count",
        "contradiction_risk"
    ]
    trust_cols = [col for col in trust_cols if col in trust_df.columns]

    df = baseline_df.merge(trust_df[trust_cols], on="record_id", how="left")
    df["case_type"] = df.apply(case_type, axis=1)

    if SEMANTIC_CLAIMS_PATH.exists():
        claims_df = pd.read_csv(SEMANTIC_CLAIMS_PATH)
        score_col = "combined_claim_support_score"
        if score_col not in claims_df.columns:
            score_col = "semantic_claim_support_score"

        weakest_claims = (
            claims_df.sort_values(score_col)
            .groupby("record_id")
            .head(1)[["record_id", "claim_text", score_col, "best_evidence_sentence"]]
            .rename(columns={
                "claim_text": "weakest_claim_text",
                score_col: "weakest_claim_support_score"
            })
        )
        df = df.merge(weakest_claims, on="record_id", how="left")

    selected_parts = []

    for label in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        part = df[df["case_type"] == label].copy()

        if label == "true_positive":
            part = part.sort_values("hallucination_probability", ascending=False)
        elif label == "true_negative":
            part = part.sort_values("hallucination_probability", ascending=True)
        elif label == "false_positive":
            part = part.sort_values("hallucination_probability", ascending=False)
        else:
            part = part.sort_values("hallucination_probability", ascending=True)

        selected_parts.append(part.head(5))

    examples = pd.concat(selected_parts, ignore_index=True)

    text_cols = ["question", "knowledge", "answer", "weakest_claim_text", "best_evidence_sentence"]
    for col in text_cols:
        if col in examples.columns:
            examples[col] = examples[col].apply(short_text)

    keep_cols = [
        "case_type",
        "record_id",
        "is_hallucinated",
        "predicted_label",
        "hallucination_probability",
        "clinical_trustworthiness_score",
        "recommended_action",
        "final_grounding_score",
        "final_unsupported_claim_count",
        "contradiction_risk",
        "difficulty_level",
        "hallucination_category",
        "question",
        "answer",
        "knowledge",
        "weakest_claim_text",
        "weakest_claim_support_score",
        "best_evidence_sentence"
    ]
    keep_cols = [col for col in keep_cols if col in examples.columns]

    examples[keep_cols].to_csv(OUTPUT_DIR / "tp_tn_fp_fn_review_examples.csv", index=False)

    summary = (
        df["case_type"]
        .value_counts()
        .rename_axis("case_type")
        .reset_index(name="row_count")
    )
    summary.to_csv(OUTPUT_DIR / "tp_tn_fp_fn_counts.csv", index=False)

    print("TP/TN/FP/FN Counts:")
    print(summary.to_string(index=False))
    print()
    print("Saved examples to results/manual_review/tp_tn_fp_fn_review_examples.csv")

if __name__ == "__main__":
    main()
