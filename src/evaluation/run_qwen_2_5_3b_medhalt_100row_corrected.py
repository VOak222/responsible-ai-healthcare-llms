from pathlib import Path
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

INPUT_PATH = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
OUT_DIR = Path("results/qwen_medhalt")
REPORT_DIR = Path("reports/future_work")

OUT_PATH = OUT_DIR / "qwen_2_5_3b_medhalt_100row_corrected_predictions.csv"
ERROR_PATH = OUT_DIR / "qwen_2_5_3b_medhalt_100row_error_review.csv"
REPORT_PATH = REPORT_DIR / "qwen_2_5_3b_medhalt_100row_corrected_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def make_prompt(row):
    question = str(row.get("question", "")).strip()
    knowledge = str(row.get("knowledge", "")).strip()
    answer = str(row.get("answer", "")).strip()

    return f"""You are evaluating a healthcare answer for hallucination risk.

Use only the provided medical knowledge, question, and answer.

Medical knowledge:
{knowledge}

Question:
{question}

Answer to evaluate:
{answer}

Task:
Decide if the answer is supported by the medical knowledge or hallucinated.

Return only one label:
0 = supported
1 = hallucinated
"""


def extract_prediction(text):
    cleaned = str(text).strip().lower()

    if cleaned.startswith("0"):
        return 0
    if cleaned.startswith("1"):
        return 1
    if "supported" in cleaned and "hallucinated" not in cleaned:
        return 0
    if "hallucinated" in cleaned:
        return 1

    return -1


def compute_metrics(result):
    valid = result[result["qwen_2_5_3b_prediction"].isin([0, 1])].copy()
    parse_success = round(len(valid) / max(len(result), 1), 3)

    if valid.empty:
        return {
            "parse_success": parse_success,
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    y_true = valid["is_hallucinated"].astype(int)
    y_pred = valid["qwen_2_5_3b_prediction"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = round((tp + tn) / max(len(valid), 1), 3)
    precision = round(tp / max(tp + fp, 1), 3)
    recall = round(tp / max(tp + fn, 1), 3)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-9), 3)

    return {
        "parse_success": parse_success,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH).head(100).copy()

    existing = pd.DataFrame()
    completed_ids = set()

    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        if "record_id" in existing.columns:
            completed_ids = set(existing["record_id"].astype(str))
            print(f"Resuming from checkpoint. Completed rows found: {len(completed_ids)}")

    print("Loading tokenizer:", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model:", MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    outputs = existing.to_dict("records") if not existing.empty else []
    start_time = time.time()

    for position, (idx, row) in enumerate(df.iterrows(), start=1):
        row_id = str(row.get("record_id", idx))

        if row_id in completed_ids:
            print(f"Row {position}/100 already completed, skipping")
            continue

        messages = [
            {"role": "system", "content": "You are a careful healthcare AI safety evaluator."},
            {"role": "user", "content": make_prompt(row)},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        row_start = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        prediction = extract_prediction(response)

        elapsed = round(time.time() - row_start, 2)

        outputs.append({
            "record_id": row.get("record_id", idx),
            "question": row.get("question", ""),
            "knowledge": row.get("knowledge", ""),
            "answer": row.get("answer", ""),
            "is_hallucinated": row.get("is_hallucinated", ""),
            "qwen_2_5_3b_raw_response": response,
            "qwen_2_5_3b_prediction": prediction,
            "runtime_seconds": elapsed,
        })

        pd.DataFrame(outputs).to_csv(OUT_PATH, index=False)

        print(f"Row {position}/100 complete | prediction={prediction} | response={response!r} | seconds={elapsed}")

    result = pd.DataFrame(outputs)
    result.to_csv(OUT_PATH, index=False)

    metrics = compute_metrics(result)

    valid = result[result["qwen_2_5_3b_prediction"].isin([0, 1])].copy()
    error_review = valid[
        valid["is_hallucinated"].astype(int) != valid["qwen_2_5_3b_prediction"].astype(int)
    ].copy()
    error_review.to_csv(ERROR_PATH, index=False)

    total_runtime = round(time.time() - start_time, 2)
    avg_runtime = round(result["runtime_seconds"].mean(), 2)

    report = f"""# Qwen2.5-3B Med-HALT 100-Row Corrected Evaluation

## Purpose

This evaluation tests Qwen2.5-3B-Instruct as a stronger local open-source model for healthcare hallucination detection.

The same 100-row Med-HALT balanced sample is used so it can be compared with the earlier Qwen 0.5B and Qwen 1.5B results.

## Results

| Metric | Value |
|---|---:|
| Rows tested | {len(result)} |
| Parse success | {metrics["parse_success"]} |
| Accuracy | {metrics["accuracy"]} |
| Precision | {metrics["precision"]} |
| Recall | {metrics["recall"]} |
| F1 score | {metrics["f1"]} |
| True positives | {metrics["tp"]} |
| True negatives | {metrics["tn"]} |
| False positives | {metrics["fp"]} |
| False negatives | {metrics["fn"]} |
| Error review rows | {len(error_review)} |
| Total runtime seconds | {total_runtime} |
| Average runtime per row | {avg_runtime} |

## Interpretation

This result should be treated as future-work evidence until it is reviewed and added to the main project comparison.

If the false negative count stays low, Qwen2.5-3B is useful as a conservative healthcare safety model. If false positives are high, it may still be valuable for review routing, but not necessarily for automatic acceptance.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print()
    print("Qwen2.5-3B 100-row corrected evaluation complete")
    print("parse_success:", metrics["parse_success"])
    print("accuracy:", metrics["accuracy"])
    print("precision:", metrics["precision"])
    print("recall:", metrics["recall"])
    print("f1:", metrics["f1"])
    print("tp:", metrics["tp"])
    print("tn:", metrics["tn"])
    print("fp:", metrics["fp"])
    print("fn:", metrics["fn"])
    print("error_review_rows:", len(error_review))
    print("total_runtime_seconds:", total_runtime)
    print("average_runtime_per_row:", avg_runtime)
    print("Saved:", OUT_PATH)
    print("Saved:", ERROR_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
