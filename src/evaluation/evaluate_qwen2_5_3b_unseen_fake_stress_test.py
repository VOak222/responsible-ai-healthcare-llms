import gc
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 768
CANDIDATE_LABELS = ["supported", "hallucinated"]

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
RESULT_DIR = PROJECT_DIR / "results"

TEST_PATH = (
    DATA_DIR / "unseen_medhalt_reasoning_fake_stress_test.jsonl"
)

BEST_ADAPTER_DIR = (
    ARTIFACT_DIR
    / "adapters"
    / "qwen2_5_3b_medhalt_hallucination_best_validation"
)

PREDICTIONS_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_test_predictions.csv"
)

SUMMARY_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_test_summary.csv"
)

BASE_MODEL_NAME = "Base Qwen2.5-3B"
LORA_MODEL_NAME = "Qwen2.5-3B + Med-HALT Hallucination LoRA"


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_quantization_config():
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


@torch.inference_mode()
def score_candidate_labels(model, tokenizer, prompt):
    candidate_token_ids = [
        tokenizer(
            f" {label}",
            add_special_tokens=False,
        )["input_ids"]
        for label in CANDIDATE_LABELS
    ]

    longest_candidate = max(
        len(token_ids)
        for token_ids in candidate_token_ids
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH - longest_candidate,
    )["input_ids"]

    sequences = [
        prompt_ids + label_token_ids
        for label_token_ids in candidate_token_ids
    ]

    max_sequence_length = max(
        len(sequence)
        for sequence in sequences
    )

    input_ids = torch.full(
        (len(sequences), max_sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (len(sequences), max_sequence_length),
        dtype=torch.long,
    )

    for row_index, sequence in enumerate(sequences):
        sequence_length = len(sequence)

        input_ids[row_index, :sequence_length] = torch.tensor(
            sequence,
            dtype=torch.long,
        )

        attention_mask[row_index, :sequence_length] = 1

    outputs = model(
        input_ids=input_ids.to("cuda"),
        attention_mask=attention_mask.to("cuda"),
    )

    scores = {}

    for row_index, (label, label_token_ids) in enumerate(
        zip(CANDIDATE_LABELS, candidate_token_ids)
    ):
        token_log_probabilities = []

        for token_offset, token_id in enumerate(label_token_ids):
            token_position = len(prompt_ids) - 1 + token_offset

            token_logits = outputs.logits[
                row_index,
                token_position,
            ]

            token_log_probability = torch.log_softmax(
                token_logits,
                dim=-1,
            )[token_id]

            token_log_probabilities.append(
                token_log_probability.item()
            )

        scores[label] = sum(token_log_probabilities) / len(
            token_log_probabilities
        )

    prediction = max(scores, key=scores.get)

    return prediction, scores


def calculate_metrics(true_labels, predicted_labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CANDIDATE_LABELS,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CANDIDATE_LABELS,
    )

    label_metrics = {
        label: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(CANDIDATE_LABELS)
    }

    true_series = pd.Series(true_labels)
    predicted_series = pd.Series(predicted_labels)

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
        "missed_hallucinations": int(
            (
                (true_series == "hallucinated")
                & (predicted_series == "supported")
            ).sum()
        ),
        "supported_answers_flagged_as_hallucinations": int(
            (
                (true_series == "supported")
                & (predicted_series == "hallucinated")
            ).sum()
        ),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(model, tokenizer, records, model_name):
    rows = []

    for row_number, record in enumerate(
        tqdm(records, desc=f"Scoring {model_name}"),
        start=1,
    ):
        prediction, scores = score_candidate_labels(
            model,
            tokenizer,
            record["prompt"],
        )

        rows.append(
            {
                "model": model_name,
                "row_number": row_number,
                "id": record["id"],
                "true_label": record["label"],
                "predicted_label": prediction,
                "score_supported": round(
                    scores["supported"],
                    6,
                ),
                "score_hallucinated": round(
                    scores["hallucinated"],
                    6,
                ),
                "confidence_margin": round(
                    abs(
                        scores["hallucinated"]
                        - scores["supported"]
                    ),
                    6,
                ),
                "case_type": record["case_type"],
                "question_group_id": record["question_group_id"],
            }
        )

    predictions = pd.DataFrame(rows)

    metrics = calculate_metrics(
        predictions["true_label"],
        predictions["predicted_label"],
    )

    return predictions, metrics


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_quantization_config(),
        device_map="auto",
    )


def clear_memory(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def print_confusion_matrix(model_name, metrics):
    print(f"\n{model_name} confusion matrix")

    print(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["true_supported", "true_hallucinated"],
            columns=["pred_supported", "pred_hallucinated"],
        ).to_string()
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Stress test not found: {TEST_PATH}")

    if not BEST_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Best LoRA adapter not found: {BEST_ADAPTER_DIR}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(TEST_PATH)

    if len(records) != 1000:
        raise RuntimeError(
            f"Expected 1,000 frozen stress-test rows, found {len(records)}."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Unseen Reasoning Fake stress-test rows:", len(records))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\nLoading base Qwen2.5-3B...")
    base_model = load_base_model()
    base_model.eval()

    base_predictions, base_metrics = evaluate_model(
        base_model,
        tokenizer,
        records,
        BASE_MODEL_NAME,
    )

    clear_memory(base_model)

    print("\nLoading Qwen2.5-3B with the selected LoRA adapter...")
    base_model = load_base_model()

    lora_model = PeftModel.from_pretrained(
        base_model,
        str(BEST_ADAPTER_DIR),
    )

    lora_model.eval()

    lora_predictions, lora_metrics = evaluate_model(
        lora_model,
        tokenizer,
        records,
        LORA_MODEL_NAME,
    )

    clear_memory(lora_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    all_predictions = pd.concat(
        [base_predictions, lora_predictions],
        ignore_index=True,
    )

    all_predictions.to_csv(PREDICTIONS_PATH, index=False)

    summary = pd.DataFrame(
        [
            {"model": BASE_MODEL_NAME, **base_metrics},
            {"model": LORA_MODEL_NAME, **lora_metrics},
        ]
    )

    summary.drop(
        columns=["confusion_matrix"],
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nUNSEEN REASONING FAKE STRESS-TEST COMPARISON")
    print(
        summary.drop(
            columns=["confusion_matrix"],
        ).to_string(index=False)
    )

    print_confusion_matrix(BASE_MODEL_NAME, base_metrics)
    print_confusion_matrix(LORA_MODEL_NAME, lora_metrics)

    print("\nSaved predictions:", PREDICTIONS_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()