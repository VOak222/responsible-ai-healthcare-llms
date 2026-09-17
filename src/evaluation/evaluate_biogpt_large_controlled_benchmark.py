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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

MODEL_ID = "microsoft/BioGPT-Large"
MAX_LENGTH = 768

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "controlled_benchmark_test.jsonl"
)
ADAPTER_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "adapters"
    / "biogpt_large_controlled_benchmark"
)
RESULT_DIR = PROJECT_DIR / "results"

BASE_PROGRESS_PATH = RESULT_DIR / "biogpt_large_controlled_base_test_progress.csv"
ADAPTER_PROGRESS_PATH = RESULT_DIR / "biogpt_large_controlled_lora_test_progress.csv"
PREDICTIONS_PATH = RESULT_DIR / "biogpt_large_controlled_benchmark_test_predictions.csv"
SUMMARY_PATH = RESULT_DIR / "biogpt_large_controlled_benchmark_test_summary.csv"

EVAL_LABELS = ["yes", "no"]
CANDIDATE_LABELS = ["yes", "no", "maybe"]


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    sequences = [prompt_ids + ids for ids in candidate_ids]
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

    logits = outputs.logits
    scores = {}

    for row_index, (label, label_ids) in enumerate(
        zip(CANDIDATE_LABELS, candidate_ids)
    ):
        token_log_probabilities = []

        for token_offset, token_id in enumerate(label_ids):
            prediction_position = len(prompt_ids) - 1 + token_offset
            token_logits = logits[row_index, prediction_position]
            token_log_probability = torch.log_softmax(
                token_logits,
                dim=-1,
            )[token_id]

            token_log_probabilities.append(token_log_probability.item())

        scores[label] = sum(token_log_probabilities) / len(
            token_log_probabilities
        )

    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores


def evaluate_model(model, model_name, records, tokenizer, progress_path):
    rows = []

    if progress_path.exists():
        existing = pd.read_csv(progress_path)

        if len(existing) > 0 and existing["model"].iloc[0] == model_name:
            rows = existing.to_dict("records")
            print(f"{model_name}: resuming from {len(rows)}/{len(records)} rows")

    completed_rows = len(rows)
    model.eval()

    for row_number, record in enumerate(records, start=1):
        if row_number <= completed_rows:
            continue

        prediction, scores = score_candidates(
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
                "is_correct": prediction == record["label"],
                "score_yes": scores["yes"],
                "score_no": scores["no"],
                "score_maybe": scores["maybe"],
            }
        )

        if row_number % 20 == 0:
            pd.DataFrame(rows).to_csv(progress_path, index=False)
            print(f"{model_name}: completed {row_number}/{len(records)} rows")

    results = pd.DataFrame(rows)
    results.to_csv(progress_path, index=False)
    return results


def summarize_results(results_df):
    y_true = results_df["true_label"]
    y_pred = results_df["predicted_label"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=EVAL_LABELS,
        zero_division=0,
    )

    summary = {
        "model": results_df["model"].iloc[0],
        "rows": len(results_df),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_precision_yes_no": round(precision.mean(), 3),
        "macro_recall_yes_no": round(recall.mean(), 3),
        "macro_f1_yes_no": round(f1.mean(), 3),
        "predicted_maybe_count": int(
            (results_df["predicted_label"] == "maybe").sum()
        ),
    }

    for label, p, r, score, count in zip(
        EVAL_LABELS,
        precision,
        recall,
        f1,
        support,
    ):
        summary[f"{label}_precision"] = round(p, 3)
        summary[f"{label}_recall"] = round(r, 3)
        summary[f"{label}_f1"] = round(score, 3)
        summary[f"{label}_support"] = int(count)

    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter not found: {ADAPTER_DIR}")

    records = load_jsonl(DATA_PATH)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Untouched test rows:", len(records))
    print("Adapter found:", ADAPTER_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # 1. Evaluate the untouched base BioGPT-Large model.
    print("\nLoading base BioGPT-Large...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_quantization_config(),
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    base_results = evaluate_model(
        base_model,
        "Base BioGPT-Large",
        records,
        tokenizer,
        BASE_PROGRESS_PATH,
    )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate the locally fine-tuned LoRA adapter.
    print("\nLoading BioGPT-Large with PubMedQA LoRA adapter...")
    adapter_base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_quantization_config(),
        device_map="auto",
    )
    adapter_base_model.config.pad_token_id = tokenizer.pad_token_id

    adapter_model = PeftModel.from_pretrained(
        adapter_base_model,
        str(ADAPTER_DIR),
    )

    adapter_results = evaluate_model(
        adapter_model,
        "BioGPT-Large + Controlled PubMedQA LoRA",
        records,
        tokenizer,
        ADAPTER_PROGRESS_PATH,
    )

    del adapter_model
    del adapter_base_model
    gc.collect()
    torch.cuda.empty_cache()

    all_results = pd.concat(
        [base_results, adapter_results],
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        [
            summarize_results(base_results),
            summarize_results(adapter_results),
        ]
    )

    all_results.to_csv(PREDICTIONS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nFINAL UNTOUCHED EXPERT-TEST COMPARISON")
    print(summary_df.to_string(index=False))

    for model_name, model_results in all_results.groupby("model"):
        matrix = confusion_matrix(
            model_results["true_label"],
            model_results["predicted_label"],
            labels=CANDIDATE_LABELS,
        )

        print(f"\n{model_name} confusion matrix")
        print(
            pd.DataFrame(
                matrix,
                index=[f"true_{label}" for label in CANDIDATE_LABELS],
                columns=[f"pred_{label}" for label in CANDIDATE_LABELS],
            )
        )

    print("\nSaved predictions:", PREDICTIONS_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()