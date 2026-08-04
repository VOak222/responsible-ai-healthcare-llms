from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/medhallu_hf")
OUTPUT_DIR = Path("results/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "pqa_labeled": DATA_DIR / "pqa_labeled_train.jsonl",
    "pqa_artificial": DATA_DIR / "pqa_artificial_train.jsonl",
}

summary_lines = []

for name, path in files.items():
    print(f"\n=== {name} ===")

    df = pd.read_json(path, lines=True)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes)

    summary_lines.append(f"\n=== {name} ===")
    summary_lines.append(f"Shape: {df.shape}")
    summary_lines.append(f"Columns: {df.columns.tolist()}")
    summary_lines.append("\nMissing values:")
    summary_lines.append(str(df.isna().sum()))
    summary_lines.append("\nData types:")
    summary_lines.append(str(df.dtypes))

    for col in df.columns:
        sample_values = df[col].dropna().head(10).tolist()

        has_list_or_dict = any(isinstance(value, (list, dict)) for value in sample_values)

        if has_list_or_dict:
            print(f"\nSkipping value counts for {col} because it contains lists/dicts.")
            summary_lines.append(f"\nSkipping value counts for {col} because it contains lists/dicts.")
            continue

        if df[col].dtype == "object" and df[col].nunique() <= 30:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False))

            summary_lines.append(f"\nValue counts for {col}:")
            summary_lines.append(str(df[col].value_counts(dropna=False)))

    print("\nSample rows:")
    print(df.head(3))

    df.head(20).to_csv(OUTPUT_DIR / f"{name}_sample.csv", index=False)

with open(OUTPUT_DIR / "medhallu_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nEDA outputs saved to results/eda/")