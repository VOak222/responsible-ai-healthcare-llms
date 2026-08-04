from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path("data/raw/medhallu_hf")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "pqa_labeled": RAW_DIR / "pqa_labeled_train.jsonl",
    "pqa_artificial": RAW_DIR / "pqa_artificial_train.jsonl",
}


def clean_knowledge(value):
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return ""
    return str(value)


records = []

for source_name, path in files.items():
    df = pd.read_json(path, lines=True)

    for idx, row in df.iterrows():
        question = row["Question"]
        knowledge = clean_knowledge(row["Knowledge"])
        difficulty = row["Difficulty Level"]

        records.append({
            "record_id": f"{source_name}_{idx}_ground_truth",
            "source_dataset": source_name,
            "question": question,
            "knowledge": knowledge,
            "answer": row["Ground Truth"],
            "difficulty_level": difficulty,
            "is_hallucinated": 0,
            "hallucination_category": "No Hallucination",
        })

        records.append({
            "record_id": f"{source_name}_{idx}_hallucinated",
            "source_dataset": source_name,
            "question": question,
            "knowledge": knowledge,
            "answer": row["Hallucinated Answer"],
            "difficulty_level": difficulty,
            "is_hallucinated": 1,
            "hallucination_category": row["Category of Hallucination"],
        })

processed_df = pd.DataFrame(records)

csv_path = PROCESSED_DIR / "medhallu_binary.csv"
jsonl_path = PROCESSED_DIR / "medhallu_binary.jsonl"

processed_df.to_csv(csv_path, index=False)
processed_df.to_json(jsonl_path, orient="records", lines=True)

print("Processed MedHallu dataset created.")
print("Shape:", processed_df.shape)
print("\nClass distribution:")
print(processed_df["is_hallucinated"].value_counts())
print("\nSource distribution:")
print(processed_df["source_dataset"].value_counts())
print("\nHallucination categories:")
print(processed_df["hallucination_category"].value_counts())
print(f"\nSaved CSV: {csv_path}")
print(f"Saved JSONL: {jsonl_path}")