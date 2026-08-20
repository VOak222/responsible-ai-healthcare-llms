from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_FILES = [
    "equitymedqa_omaq.csv",
    "equitymedqa_ehai.csv",
    "equitymedqa_fbrt_manual.csv",
    "equitymedqa_fbrt_llm_661_sampled.csv",
    "equitymedqa_trinds.csv",
    "equitymedqa_cc_manual.csv",
    "equitymedqa_cc_llm.csv",
]


def normalize_prompt(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_inventory(raw_dir, file_names):
    rows = []

    for file_name in file_names:
        path = raw_dir / file_name
        df = pd.read_csv(path, header=None)
        dataset_name = file_name.replace(".csv", "")

        for row_idx, source_row in df.iterrows():
            prompts = [
                normalize_prompt(value)
                for value in source_row.tolist()
            ]
            prompts = [prompt for prompt in prompts if prompt]

            if not prompts:
                continue

            prompt_a = prompts[0]
            prompt_b = prompts[1] if len(prompts) > 1 else ""

            rows.append({
                "dataset_name": dataset_name,
                "source_file": file_name,
                "source_row_id": row_idx,
                "record_id": f"{dataset_name}_{row_idx}",
                "prompt_type": "paired_prompt" if prompt_b else "single_prompt",
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "has_pair": int(bool(prompt_b)),
            })

    return pd.DataFrame(rows)


def main():
    parser = ArgumentParser(
        description="Prepare EquityMedQA prompt inventory for fairness evaluation."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/equitymedqa",
    )
    parser.add_argument(
        "--output",
        default="data/processed/equitymedqa_prompt_inventory.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/equitymedqa_prompt_inventory_summary.csv",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(raw_dir, DEFAULT_FILES)
    inventory.to_csv(output_path, index=False)

    summary = (
        inventory.groupby(["dataset_name", "prompt_type"])
        .size()
        .reset_index(name="rows")
        .sort_values(["dataset_name", "prompt_type"])
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved inventory to {output_path}")
    print(f"Saved summary to {summary_path}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Total rows:", len(inventory))
    print("Paired rows:", int(inventory["has_pair"].sum()))
    print("Single rows:", int((inventory["has_pair"] == 0).sum()))


if __name__ == "__main__":
    main()
