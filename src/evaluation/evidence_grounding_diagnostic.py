from pathlib import Path
import re
import pandas as pd


DATA_PATH = Path("data/processed/medhallu_binary.csv")
OUTPUT_DIR = Path("results/grounding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "as", "from", "it", "its", "into", "than", "then"
}


def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def overlap_score(answer, knowledge):
    answer_tokens = tokenize(answer)
    knowledge_tokens = tokenize(knowledge)

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(knowledge_tokens)
    return len(overlap) / len(answer_tokens)


df = pd.read_csv(DATA_PATH)

df["answer_knowledge_overlap"] = df.apply(
    lambda row: overlap_score(row["answer"], row["knowledge"]),
    axis=1,
)

summary = (
    df.groupby("is_hallucinated")["answer_knowledge_overlap"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

summary["label"] = summary["is_hallucinated"].map({
    0: "Not Hallucinated",
    1: "Hallucinated",
})

summary = summary[
    ["label", "count", "mean", "median", "min", "max"]
]

output_path = OUTPUT_DIR / "evidence_grounding_overlap_summary.csv"
summary.to_csv(output_path, index=False)

print("Evidence Grounding Overlap Summary:")
print(summary)
print(f"\nSaved summary to {output_path}")