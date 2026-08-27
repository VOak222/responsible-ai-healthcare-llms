from pathlib import Path
import math

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_PATH = Path(
    "results/qwen_medhalt/qwen_medhalt_balanced_sample_predictions.csv"
)

OUT_DIR = Path("results/qwen_medhalt")
PREDICTIONS_PATH = OUT_DIR / "qwen_medhalt_fewshot_pilot_predictions.csv"
SUMMARY_PATH = OUT_DIR / "qwen_medhalt_prompt_comparison_summary.csv"
REPORT_PATH = Path("reports/qwen_medhalt_prompt_robustness_pilot.md")

ROWS_PER_GROUP = 5
MAX_INPUT_TOKENS = 2048


def clean_text(value, max_characters):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if len(text) > max_characters:
        return text[:max_characters] + "..."

    return text


def build_zero_shot_prompt(row):
    question = clean_text(row.get("question", ""), 1800)
    knowledge = clean_text(row.get("knowledge", ""), 3000)
    answer = clean_text(row.get("answer", ""), 2200)

    return f"""Determine whether the candidate medical answer is supported by the reference evidence.

Labels:
0 = supported by the evidence
1 = hallucinated, contradicted, fabricated, or unsupported

Question:
{question}

Reference evidence:
{knowledge}

Candidate answer:
{answer}

Return only 0 or 1."""


def build_fewshot_prompt(row):
    question = clean_text(row.get("question", ""), 1800)
    knowledge = clean_text(row.get("knowledge", ""), 3000)
    answer = clean_text(row.get("answer", ""), 2200)

    return f"""Decide whether a candidate medical answer is supported by the supplied reference evidence.

Use exactly one label:

SUPPORTED = the answer agrees with and is supported by the evidence.
HALLUCINATED = the answer contradicts the evidence, invents facts, or is not supported by the evidence.

Example 1:
Question: What hormone lowers blood glucose?
Reference evidence: Insulin lowers blood glucose by helping cells absorb glucose.
Candidate answer: Insulin lowers blood glucose.
Label: SUPPORTED

Example 2:
Question: Do antibiotics cure influenza?
Reference evidence: Influenza is caused by a virus. Antibiotics treat bacterial infections and do not cure influenza.
Candidate answer: Antibiotics cure influenza.
Label: HALLUCINATED

Now evaluate this case.

Question:
{question}

Reference evidence:
{knowledge}

Candidate answer:
{answer}

Return only SUPPORTED or HALLUCINATED."""


def prepare_prompt(tokenizer, user_prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical evidence evaluator. "
                "Judge only from the reference evidence provided."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_mean_log_probability(
    tokenizer,
    model,
    device,
    formatted_prompt,
    candidate,
):
    prompt_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=False,
    )

    candidate_ids = tokenizer(
        candidate,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prompt_ids = prompt_inputs["input_ids"]
    prompt_length = prompt_ids.shape[1]
    candidate_length = candidate_ids.shape[1]

    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    candidate_ids = candidate_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits[
        0,
        prompt_length - 1 : prompt_length + candidate_length - 1,
        :
    ]

    log_probabilities = torch.log_softmax(logits, dim=-1)

    token_log_probabilities = log_probabilities.gather(
        1,
        candidate_ids[0].unsqueeze(1),
    ).squeeze(1)

    return float(token_log_probabilities.mean().cpu())


def normalized_hallucination_probability(
    supported_score,
    hallucinated_score,
):
    maximum = max(supported_score, hallucinated_score)

    supported_weight = math.exp(supported_score - maximum)
    hallucinated_weight = math.exp(hallucinated_score - maximum)

    return hallucinated_weight / (
        supported_weight + hallucinated_weight
    )


def classify_row(
    tokenizer,
    model,
    device,
    row,
    prompt_variant,
):
    if prompt_variant == "zero_shot_numeric":
        user_prompt = build_zero_shot_prompt(row)
        supported_candidate = "0"
        hallucinated_candidate = "1"
    else:
        user_prompt = build_fewshot_prompt(row)
        supported_candidate = "SUPPORTED"
        hallucinated_candidate = "HALLUCINATED"

    formatted_prompt = prepare_prompt(tokenizer, user_prompt)

    supported_score = candidate_mean_log_probability(
        tokenizer,
        model,
        device,
        formatted_prompt,
        supported_candidate,
    )

    hallucinated_score = candidate_mean_log_probability(
        tokenizer,
        model,
        device,
        formatted_prompt,
        hallucinated_candidate,
    )

    probability = normalized_hallucination_probability(
        supported_score,
        hallucinated_score,
    )

    prediction = int(probability >= 0.5)

    return {
        "prompt_variant": prompt_variant,
        "supported_candidate_score": supported_score,
        "hallucinated_candidate_score": hallucinated_score,
        "hallucination_probability": probability,
        "predicted_label": prediction,
    }


def calculate_metrics(df, dataset_name):
    true_labels = df["is_hallucinated"].astype(int)
    predictions = df["predicted_label"].astype(int)
    probabilities = df["hallucination_probability"].astype(float)

    try:
        auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "prompt_variant": df["prompt_variant"].iloc[0],
        "dataset_name": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(true_labels, predictions),
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
        "roc_auc": auc,
        "predicted_hallucination_rate": predictions.mean(),
        "mean_hallucination_probability": probabilities.mean(),
    }


def markdown_table(df):
    headers = list(df.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = []

        for column in headers:
            value = row[column]

            if isinstance(value, float):
                value = f"{value:.4f}"

            values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "dataset_name",
        "record_id",
        "question",
        "knowledge",
        "answer",
        "is_hallucinated",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    pilot_df = (
        source_df
        .sort_values(
            ["dataset_name", "is_hallucinated", "record_id"]
        )
        .groupby(
            ["dataset_name", "is_hallucinated"],
            group_keys=False,
        )
        .head(ROWS_PER_GROUP)
        .reset_index(drop=True)
    )

    print("Pilot rows:", len(pilot_df))
    print()
    print(
        pilot_df.groupby(
            ["dataset_name", "is_hallucinated"]
        )
        .size()
        .to_string()
    )
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading model:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    rows = []
    total_evaluations = len(pilot_df) * 2
    completed = 0

    for _, source_row in pilot_df.iterrows():
        for prompt_variant in [
            "zero_shot_numeric",
            "fewshot_descriptive",
        ]:
            completed += 1

            print(
                f"Evaluating {completed}/{total_evaluations}: "
                f"{source_row['dataset_name']} "
                f"label={source_row['is_hallucinated']} "
                f"variant={prompt_variant}"
            )

            classification = classify_row(
                tokenizer,
                model,
                device,
                source_row,
                prompt_variant,
            )

            result = {
                "model_id": MODEL_ID,
                "dataset_name": source_row["dataset_name"],
                "record_id": source_row["record_id"],
                "question": source_row["question"],
                "knowledge": source_row["knowledge"],
                "answer": source_row["answer"],
                "is_hallucinated": int(
                    source_row["is_hallucinated"]
                ),
                **classification,
            }

            result["is_correct_prediction"] = (
                result["predicted_label"]
                == result["is_hallucinated"]
            )

            rows.append(result)

            pd.DataFrame(rows).to_csv(
                PREDICTIONS_PATH,
                index=False,
            )

            print(
                "prediction="
                f"{result['predicted_label']} "
                "probability="
                f"{result['hallucination_probability']:.4f}"
            )

    results_df = pd.DataFrame(rows)
    summary_rows = []

    for prompt_variant in results_df[
        "prompt_variant"
    ].unique():
        variant_df = results_df[
            results_df["prompt_variant"] == prompt_variant
        ]

        for dataset_name, dataset_df in variant_df.groupby(
            "dataset_name"
        ):
            summary_rows.append(
                calculate_metrics(dataset_df, dataset_name)
            )

        summary_rows.append(
            calculate_metrics(variant_df, "combined")
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["prompt_variant", "dataset_name"]
    ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)

    combined_summary = summary_df[
        summary_df["dataset_name"] == "combined"
    ].copy()

    best_f1_row = combined_summary.loc[
        combined_summary["f1_score"].idxmax()
    ]

    report = f"""# Qwen Med-HALT Prompt Robustness Pilot

## Purpose

This pilot tests whether Qwen's weak hallucination detection was caused mainly by the original prompt and numerical labels.

The same 20 balanced cases were evaluated using:

- a zero-shot numerical prompt using `0` and `1`;
- a few-shot descriptive prompt using `SUPPORTED` and `HALLUCINATED`.

## Results

{markdown_table(summary_df)}

## Combined Comparison

{markdown_table(combined_summary)}

## Initial Interpretation

The prompt variant with the highest combined F1 score was `{best_f1_row["prompt_variant"]}`, with an F1 score of {best_f1_row["f1_score"]:.4f} and ROC-AUC of {best_f1_row["roc_auc"]:.4f}.

This is a small prompt-robustness pilot. It should not be treated as a final model evaluation. A prompt should only replace the existing approach if it improves recall, F1, and ranking behavior without collapsing toward one prediction label.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print()
    print("Prompt Comparison Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Saved:", PREDICTIONS_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
