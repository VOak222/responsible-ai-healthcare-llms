import gc
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 768

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

# Epoch 1 was chosen before the final test evaluation because it had
# the lowest validation loss.
ADAPTER_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "checkpoints"
    / "qwen2_5_3b_controlled_benchmark"
    / "epoch_1"
)

VALIDATION_DATA_PATH = DATA_DIR / "controlled_benchmark_validation.jsonl"

VALIDATION_PREDICTIONS_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_human_review_validation_predictions.csv"
)
TEST_PREDICTIONS_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_benchmark_test_predictions.csv"
)
TEST_ROUTING_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_human_review_test_routing.csv"
)
SUMMARY_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_human_review_calibration_summary.json"
)

CANDIDATE_LABELS = ["yes", "no", "maybe"]
LORA_MODEL_NAME = (
    "Qwen2.5-3B + Controlled PubMedQA LoRA "
    "(best validation checkpoint)"
)


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


@torch.inference_mode()
def score_candidates(model, tokenizer, prompt):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH - 8,
    )["input_ids"]

    candidate_ids = [
        tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        for label in CANDIDATE_LABELS
    ]

    sequences = [prompt_ids + label_ids for label_ids in candidate_ids]
    max_sequence_length = max(len(sequence) for sequence in sequences)

    input_ids = torch.full(
        (len(sequences), max_sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_sequence_length),
        dtype=torch.long,
    )

    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence)
        attention_mask[index, :length] = 1

    outputs = model(
        input_ids=input_ids.to("cuda"),
        attention_mask=attention_mask.to("cuda"),
    )

    scores = {}

    for row_index, (label, label_ids) in enumerate(
        zip(CANDIDATE_LABELS, candidate_ids)
    ):
        log_probabilities = []

        for token_offset, token_id in enumerate(label_ids):
            position = len(prompt_ids) - 1 + token_offset
            token_logits = outputs.logits[row_index, position]

            log_probability = torch.log_softmax(
                token_logits,
                dim=-1,
            )[token_id]

            log_probabilities.append(log_probability.item())

        scores[label] = sum(log_probabilities) / len(log_probabilities)

    prediction = max(scores, key=scores.get)
    return prediction, scores


def predict_validation(model, tokenizer, records):
    if VALIDATION_PREDICTIONS_PATH.exists():
        existing = pd.read_csv(VALIDATION_PREDICTIONS_PATH)

        if len(existing) == len(records):
            print("Using saved validation predictions.")
            return existing

    rows = []

    for row_number, record in enumerate(
        tqdm(records, desc="Scoring validation rows"),
        start=1,
    ):
        prediction, scores = score_candidates(
            model,
            tokenizer,
            record["prompt"],
        )

        rows.append(
            {
                "row_number": row_number,
                "id": record["id"],
                "true_label": record["label"],
                "predicted_label": prediction,
                "score_yes": scores["yes"],
                "score_no": scores["no"],
                "score_maybe": scores["maybe"],
            }
        )

        if row_number % 20 == 0:
            pd.DataFrame(rows).to_csv(
                VALIDATION_PREDICTIONS_PATH,
                index=False,
            )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    return predictions


def add_confidence_margin(predictions):
    predictions = predictions.copy()

    def margin(row):
        scores = sorted(
            [row["score_yes"], row["score_no"], row["score_maybe"]],
            reverse=True,
        )
        return scores[0] - scores[1]

    predictions["confidence_margin"] = predictions.apply(
        margin,
        axis=1,
    )

    return predictions


def choose_threshold(validation_predictions):
    predicted_yes = validation_predictions[
        validation_predictions["predicted_label"] == "yes"
    ]

    candidate_thresholds = sorted(
        predicted_yes["confidence_margin"].unique()
    )

    best_threshold = None
    best_accepted_rows = 0

    for threshold in candidate_thresholds:
        accepted = predicted_yes[
            predicted_yes["confidence_margin"] >= threshold
        ]

        unsafe_false_accepts = (
            (accepted["true_label"] == "no").sum()
        )

        # Safety-first rule: maximize validation coverage while allowing
        # zero observed true-no -> predicted-yes errors.
        if (
            len(accepted) > best_accepted_rows
            and unsafe_false_accepts == 0
        ):
            best_threshold = float(threshold)
            best_accepted_rows = len(accepted)

    return best_threshold


def make_routing_summary(predictions, threshold):
    predictions = predictions.copy()
    predictions["routing_decision"] = "human_review"

    if threshold is not None:
        auto_accept_mask = (
            (predictions["predicted_label"] == "yes")
            & (predictions["confidence_margin"] >= threshold)
        )

        predictions.loc[
            auto_accept_mask,
            "routing_decision",
        ] = "provisional_accept"

    accepted = predictions[
        predictions["routing_decision"] == "provisional_accept"
    ]

    accepted_rows = len(accepted)
    total_rows = len(predictions)

    accepted_accuracy = None

    if accepted_rows > 0:
        accepted_accuracy = round(
            (accepted["true_label"] == accepted["predicted_label"]).mean(),
            3,
        )

    unsafe_false_accepts = int(
        (
            (accepted["true_label"] == "no")
            & (accepted["predicted_label"] == "yes")
        ).sum()
    )

    summary = {
        "rows": total_rows,
        "accepted_rows": accepted_rows,
        "human_review_rows": total_rows - accepted_rows,
        "automated_coverage": round(accepted_rows / total_rows, 3),
        "accepted_accuracy": accepted_accuracy,
        "unsafe_false_accepts": unsafe_false_accepts,
    }

    return predictions, summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    validation_records = load_jsonl(VALIDATION_DATA_PATH)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Validation rows:", len(validation_records))
    print("Best validation checkpoint exists:", ADAPTER_DIR.exists())

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\nLoading fine-tuned Qwen2.5-3B best-validation checkpoint...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_quantization_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
    )
    model.eval()

    validation_predictions = predict_validation(
        model,
        tokenizer,
        validation_records,
    )

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    validation_predictions = add_confidence_margin(
        validation_predictions
    )

    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    test_predictions = test_predictions[
        test_predictions["model"] == LORA_MODEL_NAME
    ].copy()

    if len(test_predictions) == 0:
        raise ValueError(
            "Could not find the fine-tuned Qwen rows in the saved "
            "controlled-benchmark prediction file."
        )

    test_predictions = add_confidence_margin(test_predictions)

    threshold = choose_threshold(validation_predictions)

    validation_routing, validation_summary = make_routing_summary(
        validation_predictions,
        threshold,
    )

    test_routing, test_summary = make_routing_summary(
        test_predictions,
        threshold,
    )

    validation_routing.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )
    test_routing.to_csv(
        TEST_ROUTING_PATH,
        index=False,
    )

    summary = {
        "model": LORA_MODEL_NAME,
        "checkpoint_selection_rule": (
            "Epoch 1 selected using lowest validation loss before "
            "final test evaluation."
        ),
        "threshold_selection_rule": (
            "Maximum validation coverage with zero observed "
            "true-no to predicted-yes false accepts."
        ),
        "selected_confidence_margin": threshold,
        "validation": validation_summary,
        "untouched_test": test_summary,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nHUMAN-REVIEW CALIBRATION COMPLETE")
    print("Selected confidence margin:", threshold)
    print("\nValidation routing:", validation_summary)
    print("Untouched test routing:", test_summary)
    print("\nSaved validation predictions:", VALIDATION_PREDICTIONS_PATH)
    print("Saved test routing:", TEST_ROUTING_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()