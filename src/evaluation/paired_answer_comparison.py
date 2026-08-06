from pathlib import Path
import re

import pandas as pd


DATA_PATH = Path("data/processed/medhallu_binary.csv")
OUTPUT_DIR = Path("results/grounding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "as", "from", "it", "its", "into", "than", "then", "can",
    "may", "might", "should", "would", "could", "also", "not",
}

CLINICAL_DIRECTION_TERMS = {
    "increase", "increases", "increased", "decrease", "decreases", "decreased",
    "improve", "improves", "improved", "worsen", "worsens", "worsened",
    "reduce", "reduces", "reduced", "prevent", "prevents", "prevented",
    "cause", "causes", "caused", "treat", "treats", "treated",
    "contraindicated", "recommended", "diagnosis", "dosage", "dose",
}


def normalize_record_id(record_id):
    return (
        str(record_id)
        .replace("_ground_truth", "")
        .replace("_hallucinated", "")
    )


def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def jaccard_similarity(left_text, right_text):
    left_tokens = tokenize(left_text)
    right_tokens = tokenize(right_text)

    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def clinical_direction_overlap(left_text, right_text):
    left_terms = tokenize(left_text) & CLINICAL_DIRECTION_TERMS
    right_terms = tokenize(right_text) & CLINICAL_DIRECTION_TERMS

    if not left_terms and not right_terms:
        return 1.0
    if not left_terms or not right_terms:
        return 0.0

    return len(left_terms & right_terms) / len(left_terms | right_terms)


def main():
    df = pd.read_csv(DATA_PATH)

    required_columns = {"record_id", "question", "knowledge", "answer", "is_hallucinated"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["pair_id"] = df["record_id"].apply(normalize_record_id)

    grounded = (
        df[df["is_hallucinated"] == 0]
        .set_index("pair_id")
        .add_prefix("grounded_")
    )

    hallucinated = (
        df[df["is_hallucinated"] == 1]
        .set_index("pair_id")
        .add_prefix("hallucinated_")
    )

    pairs = grounded.join(hallucinated, how="inner")

    pairs["answer_similarity"] = pairs.apply(
        lambda row: jaccard_similarity(
            row["grounded_answer"],
            row["hallucinated_answer"],
        ),
        axis=1,
    )

    pairs["knowledge_grounded_similarity"] = pairs.apply(
        lambda row: jaccard_similarity(
            row["grounded_knowledge"],
            row["grounded_answer"],
        ),
        axis=1,
    )

    pairs["knowledge_hallucinated_similarity"] = pairs.apply(
        lambda row: jaccard_similarity(
            row["grounded_knowledge"],
            row["hallucinated_answer"],
        ),
        axis=1,
    )

    pairs["clinical_direction_similarity"] = pairs.apply(
        lambda row: clinical_direction_overlap(
            row["grounded_answer"],
            row["hallucinated_answer"],
        ),
        axis=1,
    )

    pairs["grounding_similarity_delta"] = (
        pairs["knowledge_grounded_similarity"]
        - pairs["knowledge_hallucinated_similarity"]
    )

    output_columns = [
        "grounded_record_id",
        "grounded_source_dataset",
        "grounded_difficulty_level",
        "grounded_hallucination_category",
        "answer_similarity",
        "knowledge_grounded_similarity",
        "knowledge_hallucinated_similarity",
        "grounding_similarity_delta",
        "clinical_direction_similarity",
        "grounded_question",
        "grounded_answer",
        "hallucinated_answer",
    ]

    available_output_columns = [
        column for column in output_columns if column in pairs.columns
    ]

    pairs[available_output_columns].to_csv(
        OUTPUT_DIR / "paired_answer_comparison.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "paired_examples": len(pairs),
            "mean_answer_similarity": pairs["answer_similarity"].mean(),
            "median_answer_similarity": pairs["answer_similarity"].median(),
            "mean_grounded_answer_knowledge_similarity": pairs[
                "knowledge_grounded_similarity"
            ].mean(),
            "mean_hallucinated_answer_knowledge_similarity": pairs[
                "knowledge_hallucinated_similarity"
            ].mean(),
            "mean_grounding_similarity_delta": pairs[
                "grounding_similarity_delta"
            ].mean(),
            "mean_clinical_direction_similarity": pairs[
                "clinical_direction_similarity"
            ].mean(),
        }
    ])

    summary.to_csv(
        OUTPUT_DIR / "paired_answer_comparison_summary.csv",
        index=False,
    )

    print("Paired Answer Comparison Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed pairs to {OUTPUT_DIR / 'paired_answer_comparison.csv'}")
    print(f"Saved summary to {OUTPUT_DIR / 'paired_answer_comparison_summary.csv'}")


if __name__ == "__main__":
    main()