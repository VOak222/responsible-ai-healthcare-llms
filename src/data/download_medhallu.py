from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/raw/medhallu_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["pqa_labeled", "pqa_artificial"]

for config in CONFIGS:
    print(f"\nDownloading MedHallu config: {config}")
    dataset_dict = load_dataset("UTAustin-AIHealth/MedHallu", config)

    print(dataset_dict)

    for split_name, dataset in dataset_dict.items():
        output_path = OUTPUT_DIR / f"{config}_{split_name}.jsonl"

        print(f"Saving {config}/{split_name}")
        print("Rows:", len(dataset))
        print("Columns:", dataset.column_names)

        dataset.to_json(str(output_path), orient="records", lines=True)

        print(f"Saved to: {output_path}")