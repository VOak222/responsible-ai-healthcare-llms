from pathlib import Path
import sys
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_PATH = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_predictions.csv")
OUT_DIR = Path("results/model_expansion")
REPORT_DIR = Path("reports/future_work")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def make_prompt(row):
    return f"""You are evaluating a healthcare answer for hallucination risk.

Use only the provided medical knowledge, question, and answer.

Medical knowledge:
{str(row.get("knowledge", "")).strip()}

Question:
{str(row.get("question", "")).strip()}

Answer to evaluate:
{str(row.get("answer", "")).strip()}

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


def compute_metrics(result, pred_col):
    valid = result[result[pred_col].isin([0, 1])].copy()
    parse_success = round(len(valid) / max(len(result), 1), 3)

    if valid.empty:
        return parse_success, "", "", "", "", 0, 0, 0, 0

    y_true = valid["is_hallucinated"].astype(int)
    y_pred = valid[pred_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = round((tp + tn) / len(valid), 3)
    precision = round(tp / max(tp + fp, 1), 3)
    recall = round(tp / max(tp + fn, 1), 3)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-9), 3)

    return parse_success, accuracy, precision, recall, f1, tp, tn, fp, fn


def main():
    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: python -m src.evaluation.run_new_model_medhalt_25row_pilot "
            "\"MODEL_ID\" \"run_name\""
        )

    model_id = sys.argv[1]
    run_name = sys.argv[2]

    pred_col = f"{run_name}_prediction"
    raw_col = f"{run_name}_raw_response"

    out_path = OUT_DIR / f"{run_name}_medhalt_25row_pilot_predictions.csv"
    report_path = REPORT_DIR / f"{run_name}_medhalt_25row_pilot_report.md"

    df = pd.read_csv(INPUT_PATH).head(25).copy()

    print("Loading tokenizer:", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model:", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    outputs = []
    start_time = time.time()

    for position, (idx, row) in enumerate(df.iterrows(), start=1):
        messages = [
            {
                "role": "user",
                "content": "You are a careful healthcare AI safety evaluator.`n`n" + make_prompt(row),
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
            raw_col: response,
            pred_col: prediction,
            "runtime_seconds": elapsed,
        })

        pd.DataFrame(outputs).to_csv(out_path, index=False)

        print(
            f"Row {position}/25 complete | prediction={prediction} "
            f"| response={response!r} | seconds={elapsed}"
        )

    result = pd.DataFrame(outputs)
    result.to_csv(out_path, index=False)

    parse_success, accuracy, precision, recall, f1, tp, tn, fp, fn = compute_metrics(result, pred_col)

    total_runtime = round(time.time() - start_time, 2)
    avg_runtime = round(result["runtime_seconds"].mean(), 2)

    report = f"""# {run_name} Med-HALT 25-Row Pilot

## Model

{model_id}

## Results

| Metric | Value |
|---|---:|
| Rows tested | {len(result)} |
| Parse success | {parse_success} |
| Accuracy | {accuracy} |
| Precision | {precision} |
| Recall | {recall} |
| F1 score | {f1} |
| True positives | {tp} |
| True negatives | {tn} |
| False positives | {fp} |
| False negatives | {fn} |
| Total runtime seconds | {total_runtime} |
| Average runtime per row | {avg_runtime} |

## Interpretation

This is a pilot run. The model should only move to a full 100-row evaluation if it produces parseable outputs and shows useful recall without too many false positives.
"""

    report_path.write_text(report, encoding="utf-8")

    print()
    print(f"{run_name} 25-row pilot complete")
    print("parse_success:", parse_success)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)
    print("total_runtime_seconds:", total_runtime)
    print("average_runtime_per_row:", avg_runtime)
    print("Saved:", out_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()

