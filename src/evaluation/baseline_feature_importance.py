from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/medhallu_binary.csv")
OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_input_text(row):
    return (
        f"Question: {row['question']}\n"
        f"Knowledge: {row['knowledge']}\n"
        f"Answer: {row['answer']}"
    )


df = pd.read_csv(DATA_PATH)

df["input_text"] = df.apply(build_input_text, axis=1)

df["group_id"] = df["record_id"].str.replace("_ground_truth", "", regex=False)
df["group_id"] = df["group_id"].str.replace("_hallucinated", "", regex=False)

X = df["input_text"]
y = df["is_hallucinated"]
groups = df["group_id"]

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

print("Training baseline model for feature importance...")
model.fit(X_train, y_train)

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

features = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})

top_hallucination_features = (
    features.sort_values("coefficient", ascending=False)
    .head(50)
)

top_grounded_features = (
    features.sort_values("coefficient", ascending=True)
    .head(50)
)

hallucination_path = OUTPUT_DIR / "top_hallucination_features.csv"
grounded_path = OUTPUT_DIR / "top_grounded_features.csv"

top_hallucination_features.to_csv(hallucination_path, index=False)
top_grounded_features.to_csv(grounded_path, index=False)

print("\nTop features associated with hallucinated answers:")
print(top_hallucination_features.head(15))

print("\nTop features associated with grounded answers:")
print(top_grounded_features.head(15))

print(f"\nSaved hallucination features to {hallucination_path}")
print(f"Saved grounded features to {grounded_path}")