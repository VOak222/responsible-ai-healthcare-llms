from pathlib import Path
import re

import pandas as pd


PREDICTIONS_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
DATA_PATH = Path("data/processed/medhallu_binary.csv")
OUTPUT_DIR = Path("results/grounding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "as", "from", "it", "its", "into", "than", "then", "can",
    "may", "might", "should", "would", "could", "also", "not", "there", "their",
    "they", "them", "which", "who", "what", "when", "where", "how",
}

NEGATION_TERMS = {
    "no", "not", "never", "without", "none", "negative", "absent",
    "lack", "lacks", "lacked",
}

UNCERTAINTY_TERMS = {
    "may", "might", "could", "possible", "possibly", "suggest", "suggests",
    "associated", "likely", "unlikely", "unclear", "unknown", "limited",
}

CLINICAL_DIRECTION_TERMS = {
    "increase", "increases", "increased", "decrease", "decreases", "decreased",
    "improve", "improves", "improved", "worsen", "worsens", "worsened",
    "reduce", "reduces", "reduced", "prevent", "prevents", "prevented",
    "cause", "causes", "caused", "treat", "treats", "treated",
    "recommend", "recommends", "recommended", "contraindicated",
}

HIGH_IMPACT_TERMS = {
    "diagnosis", "diagnose", "diagnosed", "dose", "dosage", "mg", "treatment",
    "prescribe", "prescribed", "contraindicated", "emergency", "risk", "death",
    "mortality", "adverse", "side", "effect", "symptom", "therapy",
}


def split_claims(answer):
    text = str(answer).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 12]


def tokenize(text, keep_negations=False):
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(text).lower())
    filtered_tokens = set()

    for token in tokens:
        if token in STOPWORDS and not (keep_negations and token in NEGATION_TERMS):
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        filtered_tokens.add(token)

    return filtered_tokens


def extract_numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", str(text).lower()))


def overlap_score(left_tokens, right_tokens):
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def term_alignment(claim_tokens, evidence_tokens, term_set):
    claim_terms = claim_tokens & term_set
    evidence_terms = evidence_tokens & term_set

    if not claim_terms:
        return 1.0
    if not evidence_terms:
        return 0.0

    return len(claim_terms & evidence_terms) / len(claim_terms)


def numeric_alignment(claim_text, evidence_text):
    claim_numbers = extract_numbers(claim_text)
    evidence_numbers = extract_numbers(evidence_text)

    if not claim_numbers:
        return 1.0
    if not evidence_numbers:
        return 0.0

    return len(claim_numbers & evidence_numbers) / len(claim_numbers)


def classify_support(support_score):
    if support_score >= 0.70:
        return "supported"
    if support_score >= 0.40:
        return "partial"
    return "unsupported"


def score_claim(claim_text, evidence_text):
    claim_tokens = tokenize(claim_text, keep_negations=True)
    evidence_tokens = tokenize(evidence_text, keep_negations=True)

    token_support = overlap_score(claim_tokens, evidence_tokens)
    direction_support = term_alignment(
        claim_tokens,
        evidence_tokens,
        CLINICAL_DIRECTION_TERMS,
    )
    negation_support = term_alignment(claim_tokens, evidence_tokens, NEGATION_TERMS)
    uncertainty_support = term_alignment(
        claim_tokens,
        evidence_tokens,
        UNCERTAINTY_TERMS,
    )
    number_support = numeric_alignment(claim_text, evidence_text)

    high_impact_claim = bool(claim_tokens & HIGH_IMPACT_TERMS)

    claim_support_score = (
        0.55 * token_support
        + 0.20 * direction_support
        + 0.10 * negation_support
        + 0.10 * number_support
        + 0.05 * uncertainty_support
    )

    unsupported_high_impact = high_impact_claim and claim_support_score < 0.40

    contradiction_risk = (
        direction_support < 0.50
        or negation_support < 0.50
        or number_support < 0.50
    )

    return {
        "claim_text": claim_text,
        "claim_token_support": token_support,
        "clinical_direction_support": direction_support,
        "negation_support": negation_support,
        "numeric_support": number_support,
        "uncertainty_support": uncertainty_support,
        "claim_support_score": claim_support_score,
        "claim_support_label": classify_support(claim_support_score),
        "high_impact_claim": high_impact_claim,
        "unsupported_high_impact_claim": unsupported_high_impact,
        "contradiction_risk": contradiction_risk,
    }


def load_input_data():
    if PREDICTIONS_PATH.exists():
        df = pd.read_csv(PREDICTIONS_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if "knowledge" not in df.columns or "answer" not in df.columns:
        source_df = pd.read_csv(DATA_PATH)
        merge_columns = ["record_id", "question", "knowledge", "answer", "is_hallucinated"]
        df = df.merge(source_df[merge_columns], on="record_id", how="left")

    return df


def main():
    df = load_input_data()

    required_columns = {"record_id", "is_hallucinated", "knowledge", "answer"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    row_summaries = []
    claim_rows = []

    for _, row in df.iterrows():
        claims = split_claims(row["answer"])
        claim_scores = [
            score_claim(claim, row["knowledge"])
            for claim in claims
        ]

        if claim_scores:
            answer_grounding_score = sum(
                claim["claim_support_score"] for claim in claim_scores
            ) / len(claim_scores)
        else:
            answer_grounding_score = 0.0

        unsupported_claim_count = sum(
            claim["claim_support_label"] == "unsupported"
            for claim in claim_scores
        )
        partial_claim_count = sum(
            claim["claim_support_label"] == "partial"
            for claim in claim_scores
        )
        supported_claim_count = sum(
            claim["claim_support_label"] == "supported"
            for claim in claim_scores
        )
        unsupported_high_impact_count = sum(
            claim["unsupported_high_impact_claim"]
            for claim in claim_scores
        )
        contradiction_risk_count = sum(
            claim["contradiction_risk"]
            for claim in claim_scores
        )

        row_summaries.append({
            "record_id": row["record_id"],
            "is_hallucinated": row["is_hallucinated"],
            "predicted_label": row.get("predicted_label"),
            "hallucination_probability": row.get("hallucination_probability"),
            "claim_count": len(claim_scores),
            "supported_claim_count": supported_claim_count,
            "partial_claim_count": partial_claim_count,
            "unsupported_claim_count": unsupported_claim_count,
            "unsupported_high_impact_claim_count": unsupported_high_impact_count,
            "contradiction_risk_count": contradiction_risk_count,
            "answer_claim_grounding_score": answer_grounding_score,
        })

        for claim_index, claim_score in enumerate(claim_scores, start=1):
            claim_rows.append({
                "record_id": row["record_id"],
                "claim_index": claim_index,
                "is_hallucinated": row["is_hallucinated"],
                **claim_score,
            })

    row_summary_df = pd.DataFrame(row_summaries)
    claim_detail_df = pd.DataFrame(claim_rows)

    row_summary_df.to_csv(
        OUTPUT_DIR / "claim_level_grounding_scores.csv",
        index=False,
    )
    claim_detail_df.to_csv(
        OUTPUT_DIR / "claim_level_grounding_claims.csv",
        index=False,
    )

    label_summary = (
        claim_detail_df.groupby(["is_hallucinated", "claim_support_label"])
        .size()
        .reset_index(name="claim_count")
        .sort_values(["is_hallucinated", "claim_support_label"])
    )

    overall = pd.DataFrame([
        {
            "rows_scored": len(row_summary_df),
            "claims_scored": len(claim_detail_df),
            "mean_answer_claim_grounding_score": (
                row_summary_df["answer_claim_grounding_score"].mean()
            ),
            "mean_unsupported_claim_count": (
                row_summary_df["unsupported_claim_count"].mean()
            ),
            "mean_unsupported_high_impact_claim_count": (
                row_summary_df["unsupported_high_impact_claim_count"].mean()
            ),
            "rows_with_contradiction_risk": (
                row_summary_df["contradiction_risk_count"].gt(0).sum()
            ),
        }
    ])

    label_summary.to_csv(
        OUTPUT_DIR / "claim_level_grounding_label_summary.csv",
        index=False,
    )
    overall.to_csv(
        OUTPUT_DIR / "claim_level_grounding_overall.csv",
        index=False,
    )

    print("Claim-Level Grounding Overall:")
    print(overall.to_string(index=False))

    print("\nClaim Support Label Summary:")
    print(label_summary.to_string(index=False))

    print(f"\nSaved row scores to {OUTPUT_DIR / 'claim_level_grounding_scores.csv'}")
    print(f"Saved claim details to {OUTPUT_DIR / 'claim_level_grounding_claims.csv'}")
    print(f"Saved label summary to {OUTPUT_DIR / 'claim_level_grounding_label_summary.csv'}")
    print(f"Saved overall metrics to {OUTPUT_DIR / 'claim_level_grounding_overall.csv'}")


if __name__ == "__main__":
    main()