from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLAIMS_PATH = Path("results/grounding/claim_level_grounding_claims.csv")
PREDICTIONS_PATH = Path("results/baseline/baseline_tfidf_predictions.csv")
OUTPUT_DIR = Path("results/grounding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_evidence_sentences(text):
    text = str(text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
    return sentences or [text]

def classify_support(score):
    if score >= 0.60:
        return "supported"
    if score >= 0.35:
        return "partial"
    return "unsupported"

def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Using sentence-transformers semantic model.")
        return model
    except Exception as e:
        print("Sentence-transformers unavailable. Falling back to TF-IDF.")
        print(e)
        return None

def tfidf_best_sentence_score(claim, evidence_sentences):
    if not evidence_sentences:
        return 0.0, ""
    texts = [str(claim), *evidence_sentences]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    best_index = int(scores.argmax())
    return float(scores[best_index]), evidence_sentences[best_index]

def semantic_scores_for_record(model, claims, evidence_sentences):
    if not evidence_sentences:
        return [(0.0, "") for _ in claims]

    if model is None:
        return [tfidf_best_sentence_score(claim, evidence_sentences) for claim in claims]

    claim_embeddings = model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)
    evidence_embeddings = model.encode(evidence_sentences, convert_to_tensor=True, normalize_embeddings=True)

    from sentence_transformers import util
    score_matrix = util.cos_sim(claim_embeddings, evidence_embeddings)

    results = []
    for row in score_matrix:
        best_index = int(row.argmax())
        results.append((float(row[best_index]), evidence_sentences[best_index]))
    return results

def main():
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError("Run claim_level_grounding.py first.")

    claims_df = pd.read_csv(CLAIMS_PATH)
    pred_df = pd.read_csv(PREDICTIONS_PATH)

    source_cols = ["record_id"]
    for col in ["knowledge", "predicted_label", "hallucination_probability"]:
        if col in pred_df.columns and col not in claims_df.columns:
            source_cols.append(col)
        elif col in pred_df.columns:
            source_cols.append(col)

    pred_small = pred_df[source_cols].drop_duplicates("record_id")
    df = claims_df.merge(pred_small, on="record_id", how="left", suffixes=("", "_pred"))

    if "knowledge" not in df.columns and "knowledge_pred" in df.columns:
        df["knowledge"] = df["knowledge_pred"]

    model = load_sentence_model()

    scored_parts = []
    for record_id, group in df.groupby("record_id", sort=False):
        evidence_sentences = split_evidence_sentences(group["knowledge"].iloc[0])
        claims = group["claim_text"].astype(str).tolist()
        scored = semantic_scores_for_record(model, claims, evidence_sentences)

        temp = group.copy()
        temp["semantic_claim_support_score"] = [s[0] for s in scored]
        temp["best_evidence_sentence"] = [s[1] for s in scored]
        scored_parts.append(temp)

    df = pd.concat(scored_parts, ignore_index=True)

    df["semantic_claim_support_label"] = df["semantic_claim_support_score"].apply(classify_support)

    lexical_score = df["claim_support_score"] if "claim_support_score" in df.columns else 0
    df["combined_claim_support_score"] = (
        0.70 * df["semantic_claim_support_score"] + 0.30 * lexical_score
    )
    df["combined_claim_support_label"] = df["combined_claim_support_score"].apply(classify_support)

    row_summary = (
        df.groupby("record_id")
        .agg(
            semantic_answer_grounding_score=("semantic_claim_support_score", "mean"),
            combined_answer_grounding_score=("combined_claim_support_score", "mean"),
            semantic_unsupported_claim_count=("semantic_claim_support_label", lambda x: (x == "unsupported").sum()),
            combined_unsupported_claim_count=("combined_claim_support_label", lambda x: (x == "unsupported").sum()),
            claim_count=("claim_index", "count")
        )
        .reset_index()
    )

    label_summary = (
        df.groupby("combined_claim_support_label")
        .size()
        .reset_index(name="claim_count")
    )

    overall = pd.DataFrame([{
        "rows_scored": row_summary["record_id"].nunique(),
        "claims_scored": len(df),
        "mean_semantic_claim_support_score": df["semantic_claim_support_score"].mean(),
        "mean_combined_claim_support_score": df["combined_claim_support_score"].mean(),
        "combined_unsupported_claims": int((df["combined_claim_support_label"] == "unsupported").sum())
    }])

    df.to_csv(OUTPUT_DIR / "semantic_claim_verification_claims.csv", index=False)
    row_summary.to_csv(OUTPUT_DIR / "semantic_claim_verification_scores.csv", index=False)
    label_summary.to_csv(OUTPUT_DIR / "semantic_claim_verification_label_summary.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "semantic_claim_verification_overall.csv", index=False)

    print("Semantic Claim Verification Overall:")
    print(overall.to_string(index=False))
    print()
    print("Combined Claim Support Label Summary:")
    print(label_summary.to_string(index=False))
    print()
    print("Saved semantic verification outputs to results/grounding")

if __name__ == "__main__":
    main()
