from pathlib import Path
import pandas as pd

TAGS_PATH = Path("results/equitymedqa_fairness/equitymedqa_prompt_fairness_tags.csv")
INVENTORY_PATH = Path("data/processed/equitymedqa_prompt_inventory.csv")
OUT_DIR = Path("results/equitymedqa_fairness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tags = pd.read_csv(TAGS_PATH)
inventory = pd.read_csv(INVENTORY_PATH)

# Sample a few prompts from each fairness category.
samples = []
for category, group in tags.groupby("fairness_category"):
    n = min(5, len(group))
    samples.append(group.sample(n=n, random_state=42))

category_sample = pd.concat(samples, ignore_index=True)
category_sample = category_sample.drop_duplicates(subset=["record_id", "fairness_category"])

# Also keep paired prompts separately because they are useful for fairness comparison.
paired_sample = inventory[inventory["has_pair"] == 1].sample(
    n=min(20, (inventory["has_pair"] == 1).sum()),
    random_state=42
)

category_out = OUT_DIR / "equitymedqa_fairness_review_sample.csv"
paired_out = OUT_DIR / "equitymedqa_paired_prompt_review_sample.csv"
summary_out = OUT_DIR / "equitymedqa_review_sample_summary.csv"

category_sample.to_csv(category_out, index=False)
paired_sample.to_csv(paired_out, index=False)

summary = pd.DataFrame([
    {"file": category_out.name, "rows": len(category_sample), "purpose": "Manual review sample by fairness category"},
    {"file": paired_out.name, "rows": len(paired_sample), "purpose": "Paired prompt sample for demographic comparison"}
])
summary.to_csv(summary_out, index=False)

print("Saved:", category_out)
print("Saved:", paired_out)
print("Saved:", summary_out)
print()
print("Category sample rows:", len(category_sample))
print("Paired sample rows:", len(paired_sample))
print()
print(category_sample["fairness_category"].value_counts().to_string())
