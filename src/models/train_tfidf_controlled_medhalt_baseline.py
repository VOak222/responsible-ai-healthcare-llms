import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

TRAIN_PATH = DATA_DIR / "controlled_medhalt_hallucination_train.jsonl"
VALIDATION_PATH = (
    DATA_DIR / "controlled_medhalt_hallucination_validation.jsonl"
)
TEST_PATH = DATA_DIR / "controlled_medhalt_hallucination_test.jsonl"

PREDICTIONS_PATH = (
    RESULT_DIR / "tfidf_controlled_medhalt_hallucination_predictions.csv"
)
SUMMARY_PATH = (
    RESULT_DIR / "tfidf_controlled_medhalt_hallucination_summary.json"
)

LABELS = ["supported", "hallucinated"]
CANDIDATE_C_VALUES = [0.1, 0.3, 1.0, 3.0]
RANDOM_STATE = 42


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def extract_text(records):
    return [record["prompt"] for record in records]


def extract_labels(records):
    return [record["label"] for record in records]


def calculate_metrics(true_labels, predicted_labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS,
    )

    label_metrics = {
        label: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }

    missed_hallucinations = int(
        (
            (pd.Series(true_labels) == "hallucinated")
            & (pd.Series(predicted_labels) == "supported")
        ).sum()
    )

    supported_answers_flagged = int(
        (
            (pd.Series(true_labels) == "supported")
            & (pd.Series(predicted_labels) == "hallucinated")
        ).sum()
    )

    return {
        "rows": int(len(true_labels)),
        "accuracy": round(
            float(accuracy_score(true_labels, predicted_labels)),
            4,
        ),
        "macro_f1": round(float(f1.mean()), 4),
        "supported_precision": label_metrics["supported"]["precision"],
        "supported_recall": label_metrics["supported"]["recall"],
        "supported_f1": label_metrics["supported"]["f1"],
        "hallucination_precision": (
            label_metrics["hallucinated"]["precision"]
        ),
        "hallucination_recall": (
            label_metrics["hallucinated"]["recall"]
        ),
        "hallucination_f1": label_metrics["hallucinated"]["f1"],
        "missed_hallucinations": missed_hallucinations,
        "supported_answers_flagged_as_hallucinations": (
            supported_answers_flagged
        ),
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_labels": matrix.tolist(),
        },
    }


def build_prediction_rows(records, model, matrix, split_name):
    predicted_labels = model.predict(matrix)
    probabilities = model.predict_proba(matrix)

    class_positions = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    rows = []

    for record, prediction, probability_row in zip(
        records,
        predicted_labels,
        probabilities,
    ):
        probability_supported = float(
            probability_row[class_positions["supported"]]
        )
        probability_hallucinated = float(
            probability_row[class_positions["hallucinated"]]
        )

        rows.append(
            {
                "split": split_name,
                "id": record["id"],
                "true_label": record["label"],
                "predicted_label": prediction,
                "probability_supported": round(
                    probability_supported,
                    6,
                ),
                "probability_hallucinated": round(
                    probability_hallucinated,
                    6,
                ),
                "confidence_margin": round(
                    abs(
                        probability_hallucinated
                        - probability_supported
                    ),
                    6,
                ),
                "dataset_name": record["dataset_name"],
                "source_split": record["source_split"],
                "case_type": record["case_type"],
                "question_group_id": record["question_group_id"],
            }
        )

    return rows, list(predicted_labels)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen controlled Med-HALT benchmark...")

    train_records = load_jsonl(TRAIN_PATH)
    validation_records = load_jsonl(VALIDATION_PATH)
    test_records = load_jsonl(TEST_PATH)

    print("Training rows:", len(train_records))
    print("Validation rows:", len(validation_records))
    print("Frozen test rows:", len(test_records))

    train_text = extract_text(train_records)
    validation_text = extract_text(validation_records)
    test_text = extract_text(test_records)

    train_labels = extract_labels(train_records)
    validation_labels = extract_labels(validation_records)
    test_labels = extract_labels(test_records)

    print("\nBuilding TF-IDF text features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=60000,
        sublinear_tf=True,
    )

    train_matrix = vectorizer.fit_transform(train_text)
    validation_matrix = vectorizer.transform(validation_text)
    test_matrix = vectorizer.transform(test_text)

    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Training feature matrix:", train_matrix.shape)

    print("\nSelecting Logistic Regression regularization on validation data...")

    candidate_results = []
    best_model = None
    best_c = None
    best_validation_macro_f1 = -1.0

    for c_value in CANDIDATE_C_VALUES:
        candidate_model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

        candidate_model.fit(train_matrix, train_labels)

        validation_predictions = candidate_model.predict(
            validation_matrix
        )

        validation_metrics = calculate_metrics(
            validation_labels,
            validation_predictions,
        )

        candidate_results.append(
            {
                "C": c_value,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_hallucination_recall": (
                    validation_metrics["hallucination_recall"]
                ),
                "validation_missed_hallucinations": (
                    validation_metrics["missed_hallucinations"]
                ),
            }
        )

        print(
            f"C={c_value}: "
            f"validation accuracy={validation_metrics['accuracy']:.3f}, "
            f"macro-F1={validation_metrics['macro_f1']:.3f}, "
            f"hallucination recall="
            f"{validation_metrics['hallucination_recall']:.3f}"
        )

        if validation_metrics["macro_f1"] > best_validation_macro_f1:
            best_model = candidate_model
            best_c = c_value
            best_validation_macro_f1 = validation_metrics["macro_f1"]

    print("\nSelected C value:", best_c)

    validation_rows, validation_predictions = build_prediction_rows(
        validation_records,
        best_model,
        validation_matrix,
        "validation",
    )

    # The frozen test data is used only after all model settings have been
    # selected using training and validation data.
    test_rows, test_predictions = build_prediction_rows(
        test_records,
        best_model,
        test_matrix,
        "frozen_test",
    )

    validation_metrics = calculate_metrics(
        validation_labels,
        validation_predictions,
    )
    test_metrics = calculate_metrics(
        test_labels,
        test_predictions,
    )

    predictions = pd.DataFrame(validation_rows + test_rows)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    summary = {
        "model": "TF-IDF + Logistic Regression",
        "task": (
            "Controlled Med-HALT binary hallucination detection: "
            "supported versus hallucinated candidate answer."
        ),
        "feature_configuration": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 60000,
            "sublinear_tf": True,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "model_selection": {
            "selection_data": "validation split only",
            "candidate_c_values": CANDIDATE_C_VALUES,
            "selected_c": best_c,
            "candidates": candidate_results,
        },
        "validation": validation_metrics,
        "frozen_test": test_metrics,
        "safety_interpretation": (
            "Missed hallucinations are rows where a hallucinated answer was "
            "incorrectly predicted as supported. These require particular "
            "attention because the unsafe answer would not be flagged."
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    comparison = pd.DataFrame(
        [
            {
                "split": "Validation",
                "accuracy": validation_metrics["accuracy"],
                "macro_f1": validation_metrics["macro_f1"],
                "hallucination_recall": (
                    validation_metrics["hallucination_recall"]
                ),
                "missed_hallucinations": (
                    validation_metrics["missed_hallucinations"]
                ),
            },
            {
                "split": "Frozen test",
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "hallucination_recall": (
                    test_metrics["hallucination_recall"]
                ),
                "missed_hallucinations": (
                    test_metrics["missed_hallucinations"]
                ),
            },
        ]
    )

    print("\nTF-IDF BASELINE COMPLETE")
    print("\nFinal selected-model comparison")
    print(comparison.to_string(index=False))

    print("\nFrozen test confusion matrix")
    print(
        pd.DataFrame(
            test_metrics["confusion_matrix"]["rows_true_labels"],
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        ).to_string()
    )

    print("\nSaved predictions:", PREDICTIONS_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()