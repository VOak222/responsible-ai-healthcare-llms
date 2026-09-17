import gc
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

VALIDATION_DATA_PATH = DATA_DIR / "controlled_benchmark_validation.jsonl"

QWEN_VALIDATION_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_human_review_validation_predictions.csv"
)
BIOGPT_VALIDATION_PATH = (
    RESULT_DIR / "biogpt_large_controlled_validation_predictions.csv"
)

QWEN_TEST_PATH = (
    RESULT_DIR / "qwen2_5_3b_controlled_benchmark_test_predictions.csv"
)
BIOGPT_TEST_PATH = (
    RESULT_DIR / "biogpt_large_controlled_benchmark_test_predictions.csv"
)

VALIDATION_ROUTING_PATH = (
    RESULT_DIR / "qwen_biogpt_controlled_agreement_validation_routing.csv"
)
TEST_ROUTING_PATH = (
    RESULT_DIR / "qwen_biogpt_controlled_agreement_test_routing.csv"
)
SUMMARY_PATH = (
    RESULT_DIR / "qwen_biogpt_controlled_agreement_summary.json"
)

BIOGPT_MODEL_ID = "microsoft/BioGPT-Large"
BIOGPT_ADAPTER_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "adapters"
    / "biogpt_large_controlled_benchmark"
)

MAX_LENGTH = 768
CANDIDATE_LABELS = ["yes", "no", "maybe"]

QWEN_MODEL_NAME = (
    "Qwen2.5-3B + Controlled PubMedQA LoRA "
    "(best validation checkpoint)"
)
BIOGPT_MODEL_NAME = "BioGPT-Large + Controlled PubMedQA LoRA"


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


def get_biogpt_validation_predictions(records):
    if BIOGPT_VALIDATION_PATH.exists():
        existing = pd.read_csv(BIOGPT_VALIDATION_PATH)

        if len(existing) == len(records):
            print("Using saved controlled BioGPT validation predictions.")
            return existing

    print("\nLoading controlled BioGPT-Large adapter...")

    tokenizer = AutoTokenizer.from_pretrained(BIOGPT_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BIOGPT_MODEL_ID,
        quantization_config=make_quantization_config(),
        device_map="auto",
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(BIOGPT_ADAPTER_DIR),
    )
    model.eval()

    rows = []

    for row_number, record in enumerate(
        tqdm(records, desc="Scoring BioGPT validation rows"),
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

    predictions = pd.DataFrame(rows)
    predictions.to_csv(BIOGPT_VALIDATION_PATH, index=False)

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


def load_fine_tuned_rows(file_path, model_name, expected_rows):
    predictions = pd.read_csv(file_path)

    predictions = predictions[
        predictions["model"] == model_name
    ].copy()

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {model_name}, "
            f"but found {len(predictions)}."
        )

    required_columns = {
        "row_number",
        "true_label",
        "predicted_label",
        "score_yes",
        "score_no",
        "score_maybe",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required prediction columns: {missing_columns}"
        )

    return predictions


def add_qwen_margin(predictions):
    predictions = predictions.copy()

    def margin(row):
        scores = sorted(
            [row["score_yes"], row["score_no"], row["score_maybe"]],
            reverse=True,
        )
        return scores[0] - scores[1]

    predictions["qwen_confidence_margin"] = predictions.apply(
        margin,
        axis=1,
    )

    return predictions


def align_predictions(qwen_predictions, biogpt_predictions):
    qwen = add_qwen_margin(qwen_predictions)

    qwen = qwen[
        [
            "row_number",
            "true_label",
            "predicted_label",
            "qwen_confidence_margin",
        ]
    ].rename(
        columns={
            "true_label": "true_label_qwen",
            "predicted_label": "qwen_prediction",
        }
    )

    biogpt = biogpt_predictions[
        [
            "row_number",
            "true_label",
            "predicted_label",
        ]
    ].rename(
        columns={
            "true_label": "true_label_biogpt",
            "predicted_label": "biogpt_prediction",
        }
    )

    merged = qwen.merge(
        biogpt,
        on="row_number",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(qwen):
        raise ValueError("Qwen and BioGPT predictions could not be aligned.")

    if not (
        merged["true_label_qwen"] == merged["true_label_biogpt"]
    ).all():
        raise ValueError("The aligned predictions have mismatched true labels.")

    merged = merged.rename(
        columns={"true_label_qwen": "true_label"}
    ).drop(columns=["true_label_biogpt"])

    merged["models_agree"] = (
        merged["qwen_prediction"] == merged["biogpt_prediction"]
    )

    return merged


def choose_threshold(validation_rows):
    candidates = validation_rows[
        (validation_rows["qwen_prediction"] == "yes")
        & (validation_rows["biogpt_prediction"] == "yes")
    ].copy()

    candidate_thresholds = sorted(
        candidates["qwen_confidence_margin"].unique()
    )

    best_threshold = None
    best_accepted_rows = 0

    for threshold in candidate_thresholds:
        accepted = candidates[
            candidates["qwen_confidence_margin"] >= threshold
        ]

        unsafe_false_accepts = (
            accepted["true_label"] == "no"
        ).sum()

        # Select the widest validation coverage with zero observed
        # unsafe true-no -> yes accepts.
        if (
            len(accepted) > best_accepted_rows
            and unsafe_false_accepts == 0
        ):
            best_threshold = float(threshold)
            best_accepted_rows = len(accepted)

    return best_threshold


def route_rows(rows, threshold):
    routed = rows.copy()
    routed["routing_decision"] = "human_review"

    if threshold is not None:
        provisional_accept_mask = (
            (routed["qwen_prediction"] == "yes")
            & (routed["biogpt_prediction"] == "yes")
            & (routed["qwen_confidence_margin"] >= threshold)
        )

        routed.loc[
            provisional_accept_mask,
            "routing_decision",
        ] = "provisional_accept"

    accepted = routed[
        routed["routing_decision"] == "provisional_accept"
    ]

    accepted_rows = len(accepted)
    total_rows = len(routed)

    accepted_accuracy = None

    if accepted_rows > 0:
        accepted_accuracy = round(
            (accepted["true_label"] == "yes").mean(),
            3,
        )

    unsafe_false_accepts = int(
        (accepted["true_label"] == "no").sum()
    )

    summary = {
        "rows": total_rows,
        "model_agreement_rows": int(routed["models_agree"].sum()),
        "model_agreement_rate": round(
            routed["models_agree"].mean(),
            3,
        ),
        "accepted_rows": accepted_rows,
        "human_review_rows": total_rows - accepted_rows,
        "automated_coverage": round(accepted_rows / total_rows, 3),
        "accepted_accuracy": accepted_accuracy,
        "unsafe_false_accepts": unsafe_false_accepts,
    }

    return routed, summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    validation_records = load_jsonl(VALIDATION_DATA_PATH)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Validation rows:", len(validation_records))
    print(
        "Controlled BioGPT adapter exists:",
        BIOGPT_ADAPTER_DIR.exists(),
    )

    qwen_validation = pd.read_csv(QWEN_VALIDATION_PATH)

    if len(qwen_validation) != len(validation_records):
        raise ValueError(
            "The saved Qwen validation prediction file does not contain "
            "the expected 100 rows."
        )

    biogpt_validation = get_biogpt_validation_predictions(
        validation_records
    )

    qwen_test = load_fine_tuned_rows(
        QWEN_TEST_PATH,
        QWEN_MODEL_NAME,
        expected_rows=200,
    )
    biogpt_test = load_fine_tuned_rows(
        BIOGPT_TEST_PATH,
        BIOGPT_MODEL_NAME,
        expected_rows=200,
    )

    validation_rows = align_predictions(
        qwen_validation,
        biogpt_validation,
    )
    test_rows = align_predictions(qwen_test, biogpt_test)

    threshold = choose_threshold(validation_rows)

    validation_routing, validation_summary = route_rows(
        validation_rows,
        threshold,
    )
    test_routing, test_summary = route_rows(
        test_rows,
        threshold,
    )

    validation_routing.to_csv(
        VALIDATION_ROUTING_PATH,
        index=False,
    )
    test_routing.to_csv(
        TEST_ROUTING_PATH,
        index=False,
    )

    summary = {
        "system": (
            "Qwen2.5-3B primary model with independent "
            "BioGPT-Large agreement gate"
        ),
        "checkpoint_selection_rule": (
            "Qwen epoch 1 was selected using validation loss before "
            "final test evaluation."
        ),
        "acceptance_rule": (
            "Accept only when Qwen predicts yes, BioGPT independently "
            "predicts yes, and Qwen meets a validation-selected "
            "confidence threshold."
        ),
        "threshold_selection_rule": (
            "Maximum validation coverage with zero observed "
            "true-no to predicted-yes false accepts."
        ),
        "selected_qwen_confidence_margin": threshold,
        "validation": validation_summary,
        "untouched_test": test_summary,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nQWEN + BIOGPT AGREEMENT-GATE EVALUATION COMPLETE")
    print("Selected Qwen confidence margin:", threshold)
    print("\nValidation routing:", validation_summary)
    print("Untouched test routing:", test_summary)
    print("\nSaved BioGPT validation predictions:", BIOGPT_VALIDATION_PATH)
    print("Saved validation routing:", VALIDATION_ROUTING_PATH)
    print("Saved test routing:", TEST_ROUTING_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()