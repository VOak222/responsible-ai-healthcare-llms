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
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LABELS = [
    "yes",
    "no",
    "maybe",
]

PROJECT_DIR = Path(__file__).resolve().parents[2]

EVALUATION_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_100row_evaluation_set.csv"
)

FROZEN_TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "controlled_benchmark_test.jsonl"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "pubmedqa_100row_comparison"
)

PREDICTION_FILE = (
    RESULT_DIR
    / "pubmedqa_lora_100row_predictions.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "pubmedqa_lora_100row_summary.csv"
)


# ============================================================
# Locate selected PubMedQA LoRA
# ============================================================

ADAPTER_CANDIDATES = [
    (
        PROJECT_DIR
        / "artifacts"
        / "checkpoints"
        / "qwen2_5_3b_controlled_benchmark"
        / "epoch_1"
    ),
    (
        PROJECT_DIR
        / "artifacts"
        / "adapters"
        / "qwen2_5_3b_controlled_benchmark"
    ),
]


def adapter_has_weights(path):
    return (
        (path / "adapter_model.safetensors").exists()
        or
        (path / "adapter_model.bin").exists()
    )


def find_adapter():
    for path in ADAPTER_CANDIDATES:

        if (
            (path / "adapter_config.json").exists()
            and adapter_has_weights(path)
        ):
            return path

    print()
    print("Could not find the PubMedQA LoRA adapter.")
    print("Checked:")

    for path in ADAPTER_CANDIDATES:
        print("  ", path)

    raise FileNotFoundError(
        "Selected PubMedQA adapter was not found."
    )


ADAPTER_PATH = find_adapter()


# ============================================================
# Load evaluation data
# ============================================================

print()
print("=" * 72)
print("PUBMEDQA 100-ROW EVALUATION - QWEN + PUBMEDQA LORA")
print("=" * 72)

df = pd.read_csv(
    EVALUATION_FILE,
    dtype={"pubmed_id": str},
)

if len(df) != 100:
    raise ValueError(
        f"Expected 100 rows but found {len(df)}."
    )

label_counts = (
    df["ground_truth_label"]
    .value_counts()
    .to_dict()
)

if label_counts.get("yes", 0) != 50:
    raise ValueError(
        "Expected exactly 50 YES rows."
    )

if label_counts.get("no", 0) != 50:
    raise ValueError(
        "Expected exactly 50 NO rows."
    )

print("Dataset verified.")
print(f"Rows: {len(df)}")
print("YES: 50")
print("NO:  50")

print()
print("Selected PubMedQA adapter:")
print(ADAPTER_PATH)


# ============================================================
# Load exact prompts from original frozen test set
# ============================================================

with FROZEN_TEST_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    frozen_records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


prompts = []

for _, row in df.iterrows():

    source_row = int(
        row["source_row_number"]
    )

    frozen_record = (
        frozen_records[
            source_row - 1
        ]
    )

    if str(
        frozen_record["id"]
    ) != str(
        row["pubmed_id"]
    ):

        raise ValueError(
            "PubMed ID mismatch between "
            "evaluation CSV and frozen test."
        )

    prompts.append(
        frozen_record["prompt"]
    )


df["evaluation_prompt"] = prompts


# ============================================================
# GPU check
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected."
    )

print()
print(
    "GPU:",
    torch.cuda.get_device_name(0),
)


# ============================================================
# Tokenizer
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# Base Qwen model
# ============================================================

print()
print(
    "Loading Qwen2.5-3B base model..."
)

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
)

base_model.eval()


# ============================================================
# Attach selected PubMedQA LoRA
# ============================================================

print()
print(
    "Attaching PubMedQA LoRA..."
)

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
)

model.eval()

print()
print(
    "Qwen2.5-3B + PubMedQA LoRA "
    "loaded successfully."
)


# ============================================================
# Candidate label scoring
# ============================================================

@torch.inference_mode()
def score_candidate(
    prompt,
    label,
):

    candidate_text = " " + label

    prompt_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    full_tokens = tokenizer(
        prompt + candidate_text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = (
        full_tokens["input_ids"]
        .to(model.device)
    )

    attention_mask = (
        full_tokens["attention_mask"]
        .to(model.device)
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    prompt_length = (
        prompt_tokens["input_ids"]
        .shape[1]
    )

    candidate_ids = (
        full_tokens["input_ids"][0]
        [prompt_length:]
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    score = 0.0

    for index, token_id in enumerate(
        candidate_ids
    ):

        prediction_position = (
            prompt_length
            + index
            - 1
        )

        token_log_prob = (
            log_probs[
                0,
                prediction_position,
                token_id,
            ]
            .item()
        )

        score += token_log_prob

    if len(candidate_ids) > 0:
        score /= len(candidate_ids)

    return score


def predict_label(prompt):

    scores = {}

    for label in LABELS:

        scores[label] = (
            score_candidate(
                prompt,
                label,
            )
        )

    predicted_label = max(
        scores,
        key=scores.get,
    )

    return (
        predicted_label,
        scores,
    )


# ============================================================
# Run all 100 rows
# ============================================================

print()
print("=" * 72)
print(
    "RUNNING QWEN + PUBMEDQA LORA "
    "ON 100 PUBMEDQA ROWS"
)
print("=" * 72)
print()

results = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="PubMedQA LoRA evaluation",
):

    prediction, scores = (
        predict_label(
            row[
                "evaluation_prompt"
            ]
        )
    )

    correct = (
        prediction
        == row[
            "ground_truth_label"
        ]
    )

    results.append(
        {
            "sample_id":
                int(row["sample_id"]),

            "pubmed_id":
                row["pubmed_id"],

            "ground_truth_label":
                row[
                    "ground_truth_label"
                ],

            "pubmedqa_lora_prediction":
                prediction,

            "correct":
                correct,

            "yes_score":
                scores["yes"],

            "no_score":
                scores["no"],

            "maybe_score":
                scores["maybe"],
        }
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# Calculate metrics
# ============================================================

y_true = (
    result_df[
        "ground_truth_label"
    ]
)

y_pred = (
    result_df[
        "pubmedqa_lora_prediction"
    ]
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[
            "yes",
            "no",
        ],
        zero_division=0,
    )
)

macro_f1 = (
    f1.mean()
)

yes_recall = (
    recall[0]
)

no_recall = (
    recall[1]
)

correct_count = int(
    result_df[
        "correct"
    ].sum()
)

yes_predictions = int(
    (
        result_df[
            "pubmedqa_lora_prediction"
        ]
        == "yes"
    ).sum()
)

no_predictions = int(
    (
        result_df[
            "pubmedqa_lora_prediction"
        ]
        == "no"
    ).sum()
)

maybe_predictions = int(
    (
        result_df[
            "pubmedqa_lora_prediction"
        ]
        == "maybe"
    ).sum()
)


# ============================================================
# Confusion matrix
# ============================================================

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        "yes",
        "no",
        "maybe",
    ],
)

matrix_df = pd.DataFrame(
    matrix,
    index=[
        "True YES",
        "True NO",
        "True MAYBE",
    ],
    columns=[
        "Pred YES",
        "Pred NO",
        "Pred MAYBE",
    ],
)


# ============================================================
# Save outputs
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result_df.to_csv(
    PREDICTION_FILE,
    index=False,
)

summary_df = pd.DataFrame(
    [
        {
            "model":
                "Qwen2.5-3B + PubMedQA LoRA",

            "adapter":
                str(ADAPTER_PATH),

            "rows":
                len(result_df),

            "correct":
                correct_count,

            "accuracy":
                accuracy,

            "macro_f1_yes_no":
                macro_f1,

            "yes_recall":
                yes_recall,

            "no_recall":
                no_recall,

            "predicted_yes":
                yes_predictions,

            "predicted_no":
                no_predictions,

            "predicted_maybe":
                maybe_predictions,
        }
    ]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print clean screenshot-ready output
# ============================================================

print()
print()
print("=" * 72)
print(
    "QWEN2.5-3B + PUBMEDQA LORA RESULTS"
)
print("=" * 72)

print(
    f"Evaluation rows:      "
    f"{len(result_df)}"
)

print(
    f"Correct predictions: "
    f"{correct_count}"
    f"/{len(result_df)}"
)

print(
    f"Accuracy:             "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro-F1:             "
    f"{macro_f1:.3f}"
)

print(
    f"YES recall:           "
    f"{yes_recall * 100:.1f}%"
)

print(
    f"NO recall:            "
    f"{no_recall * 100:.1f}%"
)

print()

print("Prediction distribution:")

print(
    f"YES:                  "
    f"{yes_predictions}"
)

print(
    f"NO:                   "
    f"{no_predictions}"
)

print(
    f"MAYBE:                "
    f"{maybe_predictions}"
)

print()

print("Confusion matrix:")
print(
    matrix_df.to_string()
)

print()
print(
    "Prediction CSV:"
)
print(
    PREDICTION_FILE
)

print()
print(
    "Summary CSV:"
)
print(
    SUMMARY_FILE
)

print()
print(
    "Important: this is "
    "PubMedQA task accuracy, "
    "not clinical accuracy."
)

print("=" * 72)


# ============================================================
# Cleanup
# ============================================================

del model
del base_model

gc.collect()

torch.cuda.empty_cache()