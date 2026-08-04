from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/MedHallu")
DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}

print("MedHallu project files found:\n")

for path in DATA_DIR.rglob("*"):
    if ".git" in path.parts:
        continue
    if path.is_file():
        print(path)

print("\nChecking for dataset files...\n")

dataset_files = [
    path for path in DATA_DIR.rglob("*")
    if ".git" not in path.parts and path.suffix.lower() in DATA_EXTENSIONS
]

if not dataset_files:
    print("No CSV, JSON, JSONL, or Parquet dataset files found.")
else:
    for path in dataset_files:
        print(f"\nDataset file: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            continue

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head())