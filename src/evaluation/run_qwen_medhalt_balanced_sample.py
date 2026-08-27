from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RANDOM_STATE = 42
ROWS_PER_DATASET_LABEL = 25

FAKE_PATH = Path(
    "data/processed/medhalt_reasoning_fake_common.csv"
)
FCT_PATH = Path(
    "data/processed/medhalt_reasoning_fct_common.csv"
)

OUTPUT_DIR = Path("results/qwen_medhalt")
PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "qwen_medhalt_balanced_sample_predictions.csv"
)
SUMMARY_PATH = (
    OUTPUT_DIR
    / "qwen_medhalt_balanced_sample_summary.csv"
)
REPORT_PATH = Path(
    "reports/qwen_medhalt_balanced_sample_summary.md"
)

QUESTION_TOKEN_LIMIT = 120
KNOWLEDGE_TOKEN_LIMIT = 300
ANSWER_TOKEN_LIMIT = 160


def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path, low_memory=False)


def truncate_text(tokenizer, value, max_tokens):
    value = str(value)

    token_ids = tokenizer(
        value,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) <= max_tokens:
        return value

    token_ids = token_ids[:max_tokens]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


def create_balanced_sample():
    frames = []

    for path in [FAKE_PATH, FCT_PATH]:
        df = require_csv(path)

        df = df.dropna(
            subset=[
                "record_id",
                "question",
                "knowledge",
                "answer",
                "is_hallucinated",
            ]
        ).copy()

        df["is_hallucinated"] = (
            df["is_hallucinated"].astype(int)
        )

        for label in [0, 1]:
            group = df[
                df["is_hallucinated"] == label
            ].copy()

            if len(group) < ROWS_PER_DATASET_LABEL:
                raise ValueError(
                    f"Not enough label {label} rows in {path}"
                )

            sample = group.sample(
                n=ROWS_PER_DATASET_LABEL,
                random_state=(
                    RANDOM_STATE
                    + label
                    + len(frames)
                ),
            )

            frames.append(sample)

    sample_df = pd.concat(
        frames,
        ignore_index=True,
    )

    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return sample_df


def build_prompt(
    tokenizer,
    question,
    knowledge,
    answer,
):
    question = truncate_text(
        tokenizer,
        question,
        QUESTION_TOKEN_LIMIT,
    )

    knowledge = truncate_text(
        tokenizer,
        knowledge,
        KNOWLEDGE_TOKEN_LIMIT,
    )

    answer = truncate_text(
        tokenizer,
        answer,
        ANSWER_TOKEN_LIMIT,
    )

    user_prompt = f"""You are evaluating whether a medical answer is supported by the supplied knowledge.

Use only the supplied knowledge.

Classification labels:
0 = The answer is supported by the knowledge.
1 = The answer contains unsupported, contradictory, fabricated, or nonsensical medical information.

Question:
{question}

Knowledge:
{knowledge}

Answer being evaluated:
{answer}

Return only the classification label 0 or 1."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical "
                "hallucination evaluator."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_log_probability(
    model,
    tokenizer,
    prompt_text,
    candidate,
):
    device = next(model.parameters()).device

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    candidate_ids = tokenizer(
        candidate,
        add_special_tokens=False,
    )["input_ids"]

    candidate_tensor = torch.tensor(
        [candidate_ids],
        dtype=torch.long,
        device=device,
    )

    full_ids = torch.cat(
        [prompt_ids, candidate_tensor],
        dim=1,
    )

    attention_mask = torch.ones_like(full_ids)

    with torch.inference_mode():
        output = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

    log_probs = F.log_softmax(
        output.logits,
        dim=-1,
    )

    prompt_length = prompt_ids.shape[1]
    token_log_probs = []

    for index, token_id in enumerate(candidate_ids):
        prediction_position = (
            prompt_length + index - 1
        )

        token_log_probs.append(
            log_probs[
                0,
                prediction_position,
                token_id,
            ]
        )

    return torch.stack(
        token_log_probs
    ).mean().item()


def classify_row(
    model,
    tokenizer,
    row,
):
    prompt_text = build_prompt(
        tokenizer,
        row["question"],
        row["knowledge"],
        row["answer"],
    )

    supported_score = candidate_log_probability(
        model,
        tokenizer,
        prompt_text,
        "0",
    )

    hallucinated_score = candidate_log_probability(
        model,
        tokenizer,
        prompt_text,
        "1",
    )

    probabilities = torch.softmax(
        torch.tensor(
            [
                supported_score,
                hallucinated_score,
            ]
        ),
        dim=0,
    ).numpy()

    supported_probability = float(probabilities[0])
    hallucination_probability = float(
        probabilities[1]
    )

    predicted_label = int(
        hallucination_probability >= 0.5
    )

    return {
        "predicted_label": predicted_label,
        "supported_probability":
            supported_probability,
        "hallucination_probability":
            hallucination_probability,
        "prediction_confidence": float(
            max(probabilities)
        ),
        "decision_margin": float(
            abs(
                hallucination_probability
                - supported_probability
            )
        ),
    }


def calculate_metrics(df, dataset_name):
    true_labels = df["is_hallucinated"].astype(int)
    predictions = df["predicted_label"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    try:
        roc_auc = roc_auc_score(
            true_labels,
            df["hallucination_probability"],
        )
    except ValueError:
        roc_auc = np.nan

    return {
        "dataset_name": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(
            true_labels,
            predictions,
        ),
        "precision": precision_score(
            true_labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            true_labels,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_df = create_balanced_sample()

    print("Balanced sample:")
    print(
        sample_df.groupby(
            ["dataset_name", "is_hallucinated"]
        )
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )
    print()

    print("Loading model:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        )
        model.to("cpu")

    model.eval()

    existing_rows = []

    if PREDICTIONS_PATH.exists():
        existing = pd.read_csv(PREDICTIONS_PATH)
        existing_rows = existing.to_dict("records")
        completed_ids = set(
            existing["record_id"].astype(str)
        )

        print(
            "Resuming existing run:",
            len(existing_rows),
            "completed rows",
        )
    else:
        completed_ids = set()

    rows = list(existing_rows)

    for index, row in sample_df.iterrows():
        record_id = str(row["record_id"])

        if record_id in completed_ids:
            continue

        print(
            f"Evaluating {len(rows) + 1}/"
            f"{len(sample_df)}:",
            row["dataset_name"],
            "true_label=",
            row["is_hallucinated"],
        )

        prediction = classify_row(
            model,
            tokenizer,
            row,
        )

        output_row = row.to_dict()
        output_row["model_id"] = MODEL_ID
        output_row.update(prediction)

        output_row["is_correct_prediction"] = (
            int(output_row["predicted_label"])
            == int(output_row["is_hallucinated"])
        )

        rows.append(output_row)

        pd.DataFrame(rows).to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

        print(
            "prediction=",
            prediction["predicted_label"],
            "hallucination_probability=",
            round(
                prediction[
                    "hallucination_probability"
                ],
                4,
            ),
        )

    predictions_df = pd.DataFrame(rows)

    predictions_df = (
        predictions_df.drop_duplicates(
            subset=["record_id"],
            keep="last",
        )
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metric_rows = []

    for dataset_name, group in predictions_df.groupby(
        "dataset_name"
    ):
        metric_rows.append(
            calculate_metrics(
                group,
                dataset_name,
            )
        )

    metric_rows.append(
        calculate_metrics(
            predictions_df,
            "combined",
        )
    )

    summary_df = pd.DataFrame(metric_rows)

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    sample_counts = (
        predictions_df.groupby(
            ["dataset_name", "is_hallucinated"]
        )
        .size()
        .reset_index(name="rows")
    )

    report = f"""# Qwen Med-HALT Balanced Sample Evaluation

## Evaluation Design

- Model: {MODEL_ID}
- Total rows: {len(predictions_df)}
- Sampling: 25 supported and 25 hallucinated rows from each Med-HALT dataset
- Classification: 0 = supported, 1 = hallucinated
- Random state: {RANDOM_STATE}

## Sample Distribution

{sample_counts.to_markdown(index=False)}

## Performance

{summary_df.to_markdown(index=False)}

## Next Step

The next step is to connect Qwen's hallucination results with the existing grounding pipeline and then calculate a same-model Clinical Trustworthiness Score.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Qwen Med-HALT Results:")
    print(summary_df.to_string(index=False))
    print()
    print("Saved:", PREDICTIONS_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
