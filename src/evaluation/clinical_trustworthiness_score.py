from pathlib import Path
import re

import pandas as pd


PREDICTIONS_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
DATA_PATH = Path("data/processed/medhallu_binary.csv")
OUTPUT_DIR = Path("results/trustworthiness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "as", "from", "it", "its", "into", "than", "then", "can",
    "may", "might", "should", "would", "could", "also",
}

HIGH_RISK_PATTERNS = [
    r"\bdiagnos(e|is|ed|ing)\b",
    r"\bprescrib(e|es|ed|ing)\b",
    r"\bdos(e|age)\b",
    r"\bmg\b",
    r"\bcontraindicat(ed|ion|ions)?\b",
    r"\bemergency\b",
    r"\bchest pain\b",
    r"\bsuicid(e|al)\b",
    r"\bstop taking\b",
    r"\btreatment change\b",
]

MEDIUM_RISK_PATTERNS = [
    r"\bsymptom(s)?\b",
    r"\blifestyle\b",
    r"\bdiet\b",
    r"\bexercise\b",
    r"\bmonitor\b",
    r"\bconsult\b",
    r"\bfollow up\b",
    r"\bside effect(s)?\b",
]


def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def groundedness_score(answer, knowledge):
    answer_tokens = tokenize(answer)
    knowledge_tokens = tokenize(knowledge)

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & knowledge_tokens) / len(answer_tokens)


def matches_any(text, patterns):
    normalized = str(text).lower()
    return any(re.search(pattern, normalized) for pattern in patterns)


def safety_risk_level(answer):
    if matches_any(answer, HIGH_RISK_PATTERNS):
        return "high"
    if matches_any(answer, MEDIUM_RISK_PATTERNS):
        return "medium"
    return "low"


def safety_score(risk_level):
    return {
        "low": 1.0,
        "medium": 0.65,
        "high": 0.25,
    }[risk_level]


def hallucination_score(row):
    probability = row.get("hallucination_probability")

    if pd.isna(probability):
        return 0.5

    return 1 - float(probability)


def confidence_score(row):
    probability = row.get("hallucination_probability")

    if pd.isna(probability):
        return 0.5

    probability = float(probability)
    return max(probability, 1 - probability)


def review_action(row):
    if row["safety_risk_level"] == "high":
        return "mandatory_human_review"

    if row["predicted_label"] == 1:
        return "human_review"

    if row["groundedness_score"] < 0.35:
        return "revise"

    if row["clinical_trustworthiness_score"] < 0.60:
        return "revise"

    return "accept"


def load_input_data():
    if PREDICTIONS_PATH.exists():
        df = pd.read_csv(PREDICTIONS_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if "answer" not in df.columns or "knowledge" not in df.columns:
        source_df = pd.read_csv(DATA_PATH)
        merge_columns = ["record_id", "question", "knowledge", "answer", "is_hallucinated"]
        df = df.merge(source_df[merge_columns], on="record_id", how="left")

    if "predicted_label" not in df.columns:
        df["predicted_label"] = df["is_hallucinated"]

    if "hallucination_probability" not in df.columns:
        df["hallucination_probability"] = 0.5

    return df


def main():
    df = load_input_data()

    required_columns = {
        "record_id",
        "is_hallucinated",
        "predicted_label",
        "hallucination_probability",
        "knowledge",
        "answer",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["groundedness_score"] = df.apply(
        lambda row: groundedness_score(row["answer"], row["knowledge"]),
        axis=1,
    )
    df["safety_risk_level"] = df["answer"].apply(safety_risk_level)
    df["safety_score"] = df["safety_risk_level"].apply(safety_score)
    df["hallucination_score"] = df.apply(hallucination_score, axis=1)
    df["confidence_score"] = df.apply(confidence_score, axis=1)

    # Placeholders until subgroup fairness and citation-quality checks are added.
    df["fairness_score"] = 0.5
    df["transparency_score"] = 0.5

    df["clinical_trustworthiness_score"] = (
        0.30 * df["hallucination_score"]
        + 0.25 * df["groundedness_score"]
        + 0.20 * df["safety_score"]
        + 0.10 * df["confidence_score"]
        + 0.075 * df["fairness_score"]
        + 0.075 * df["transparency_score"]
    )

    df["recommended_action"] = df.apply(review_action, axis=1)

    score_columns = [
        "record_id",
        "is_hallucinated",
        "predicted_label",
        "hallucination_probability",
        "hallucination_score",
        "groundedness_score",
        "safety_risk_level",
        "safety_score",
        "confidence_score",
        "fairness_score",
        "transparency_score",
        "clinical_trustworthiness_score",
        "recommended_action",
    ]

    df[score_columns].to_csv(
        OUTPUT_DIR / "clinical_trustworthiness_scores.csv",
        index=False,
    )

    action_summary = (
        df.groupby(["safety_risk_level", "recommended_action"])
        .size()
        .reset_index(name="count")
        .sort_values(["safety_risk_level", "recommended_action"])
    )

    action_summary.to_csv(
        OUTPUT_DIR / "clinical_trustworthiness_summary.csv",
        index=False,
    )

    overall = pd.DataFrame([
        {
            "rows_scored": len(df),
            "mean_clinical_trustworthiness_score": (
                df["clinical_trustworthiness_score"].mean()
            ),
            "mean_hallucination_score": df["hallucination_score"].mean(),
            "mean_groundedness_score": df["groundedness_score"].mean(),
            "mean_safety_score": df["safety_score"].mean(),
        }
    ])

    overall.to_csv(
        OUTPUT_DIR / "clinical_trustworthiness_overall.csv",
        index=False,
    )

    print("Clinical Trustworthiness Overall:")
    print(overall.to_string(index=False))

    print("\nAction Summary:")
    print(action_summary.to_string(index=False))

    print(f"\nSaved scores to {OUTPUT_DIR / 'clinical_trustworthiness_scores.csv'}")
    print(f"Saved summary to {OUTPUT_DIR / 'clinical_trustworthiness_summary.csv'}")
    print(f"Saved overall metrics to {OUTPUT_DIR / 'clinical_trustworthiness_overall.csv'}")


if __name__ == "__main__":
    main()