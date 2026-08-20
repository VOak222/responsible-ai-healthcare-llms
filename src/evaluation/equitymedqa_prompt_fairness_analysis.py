from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


CATEGORY_KEYWORDS = {
    "race_ethnicity": [
        "black", "white", "asian", "latina", "latino", "hispanic",
        "middle eastern", "south asian", "native", "race", "ethnicity",
    ],
    "gender_sexuality": [
        "woman", "women", "female", "male", "men", "trans", "queer",
        "lgbtq", "asexual", "intersex", "pregnancy", "pregnant",
    ],
    "disability": [
        "disabled", "disability", "blind", "deaf", "wheelchair",
        "autism", "neurodivergent",
    ],
    "income_access": [
        "low income", "lower ses", "no health insurance", "uninsured",
        "access", "resources", "housing", "food", "education",
    ],
    "geography": [
        "india", "karnataka", "uige", "province", "rural", "urban",
        "beach", "outside", "agriculture",
    ],
    "religion_culture": [
        "muslim", "christian", "jewish", "hindu", "religion",
        "culture", "cultural",
    ],
    "age": [
        "year old", "adolescent", "child", "children", "older",
        "elderly", "teen",
    ],
}


def normalize(text):
    return str(text).lower()


def detect_categories(text):
    lowered = normalize(text)
    categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            categories.append(category)

    if not categories:
        categories.append("general_equity_or_clinical_prompt")

    return categories


def main():
    parser = ArgumentParser(
        description="Analyze EquityMedQA prompts by rough fairness/equity category."
    )
    parser.add_argument(
        "--input",
        default="data/processed/equitymedqa_prompt_inventory.csv",
    )
    parser.add_argument(
        "--output",
        default="results/equitymedqa_fairness/equitymedqa_prompt_fairness_tags.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/equitymedqa_fairness/equitymedqa_prompt_fairness_summary.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    rows = []
    for _, row in df.iterrows():
        combined_prompt = " ".join([
            str(row.get("prompt_a", "")),
            str(row.get("prompt_b", "")),
        ])

        categories = detect_categories(combined_prompt)

        for category in categories:
            rows.append({
                "record_id": row["record_id"],
                "dataset_name": row["dataset_name"],
                "prompt_type": row["prompt_type"],
                "has_pair": row["has_pair"],
                "fairness_category": category,
                "prompt_a": row.get("prompt_a", ""),
                "prompt_b": row.get("prompt_b", ""),
            })

    tagged = pd.DataFrame(rows)
    tagged.to_csv(output_path, index=False)

    summary = (
        tagged.groupby(["fairness_category"])
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    summary.to_csv(summary_path, index=False)

    by_dataset = (
        tagged.groupby(["dataset_name", "fairness_category"])
        .size()
        .reset_index(name="rows")
        .sort_values(["dataset_name", "rows"], ascending=[True, False])
    )

    by_dataset_path = summary_path.with_name("equitymedqa_prompt_fairness_by_dataset.csv")
    by_dataset.to_csv(by_dataset_path, index=False)

    print("Saved tagged prompts to:", output_path)
    print("Saved category summary to:", summary_path)
    print("Saved dataset/category summary to:", by_dataset_path)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
