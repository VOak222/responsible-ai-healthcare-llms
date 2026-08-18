from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/processed/medhalt_reasoning_fct_common.csv")
OUT_DIR = Path("results/semantic_medhalt_reasoning_fct")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def base_source_id(value):
    text = str(value)
    return text.replace("_correct", "").replace("_student", "")


def build_reference_answers(df):
    df = df.copy()
    df["group_id"] = df["source_record_id"].apply(base_source_id)

    refs = (
        df[df["case_type"].eq("correct_answer")][["group_id", "answer"]]
        .drop_duplicates("group_id")
        .rename(columns={"answer": "reference_answer"})
    )

    df = df.merge(refs, on="group_id", how="left")
    df = df.dropna(subset=["answer", "reference_answer", "is_hallucinated"]).copy()
    df["answer"] = df["answer"].astype(str)
    df["reference_answer"] = df["reference_answer"].astype(str)
    df["is_hallucinated"] = df["is_hallucinated"].astype(int)
    return df


def cosine_similarity_scores(model, answers, references):
    answer_embeddings = model.encode(
        answers,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    reference_embeddings = model.encode(
        references,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.sum(answer_embeddings * reference_embeddings, axis=1)


def predict(scores, threshold):
    # Low similarity to benchmark answer means likely hallucinated/incorrect.
    return (scores < threshold).astype(int)


def metrics_dict(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = build_reference_answers(df)

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(
        splitter.split(df, df["is_hallucinated"], groups=df["group_id"])
    )

    print("Loading semantic model:", args.model_name)
    model = SentenceTransformer(args.model_name)

    print("Scoring answer similarity...")
    df["semantic_similarity"] = cosine_similarity_scores(
        model,
        df["answer"].tolist(),
        df["reference_answer"].tolist(),
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    sweep_rows = []
    for threshold in np.arange(0.30, 0.96, 0.01):
        threshold = round(float(threshold), 2)
        train_pred = predict(train["semantic_similarity"].to_numpy(), threshold)
        sweep_rows.append({
            "threshold": threshold,
            "train_accuracy": accuracy_score(train["is_hallucinated"], train_pred),
            "train_precision": precision_score(train["is_hallucinated"], train_pred, zero_division=0),
            "train_recall": recall_score(train["is_hallucinated"], train_pred, zero_division=0),
            "train_f1_score": f1_score(train["is_hallucinated"], train_pred, zero_division=0),
        })

    sweep = pd.DataFrame(sweep_rows)
    sweep = sweep.sort_values(
        ["train_f1_score", "train_accuracy"],
        ascending=[False, False],
    )
    selected_threshold = float(sweep.iloc[0]["threshold"])

    test_pred = predict(test["semantic_similarity"].to_numpy(), selected_threshold)
    test["semantic_predicted_label"] = test_pred
    test["semantic_correct_prediction"] = test["is_hallucinated"] == test["semantic_predicted_label"]

    metric_row = {
        "method": "Semantic similarity",
        "dataset": "medhalt_reasoning_fct",
        "threshold_selected_on_train": selected_threshold,
        "train_rows": len(train),
        "test_rows": len(test),
        **metrics_dict(test["is_hallucinated"], test_pred, "test"),
    }

    metrics = pd.DataFrame([metric_row])

    metrics.to_csv(out_dir / "semantic_fct_clean_test_metrics.csv", index=False)
    test.to_csv(out_dir / "semantic_fct_test_predictions.csv", index=False)
    sweep.to_csv(out_dir / "semantic_fct_threshold_train_sweep.csv", index=False)

    print()
    print(metrics.to_string(index=False))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(test["is_hallucinated"], test_pred))
    print()
    print(classification_report(
        test["is_hallucinated"],
        test_pred,
        target_names=["not_hallucinated", "hallucinated"],
        zero_division=0,
    ))
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
