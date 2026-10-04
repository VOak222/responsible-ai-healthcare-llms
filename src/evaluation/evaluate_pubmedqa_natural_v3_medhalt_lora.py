import gc
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

from src.evaluation.evaluate_qwen2_5_3b_medhalt_hallucination import (
    score_candidate_labels,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LABELS = [
    "supported",
    "hallucinated",
]

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_v3_100.csv"
)

ADAPTER_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "adapters"
    / "qwen2_5_3b_medhalt_hallucination_best_validation"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "pubmedqa_natural_candidate_v3"
)

PREDICTION_FILE = (
    RESULT_DIR
    / "medhalt_lora_natural_v3_predictions.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "medhalt_lora_natural_v3_summary.csv"
)


# ============================================================
# Prompt
# ============================================================

def build_evaluation_prompt(
    question,
    evidence,
    candidate_answer,
):

    return f"""You are a healthcare-answer safety evaluator.
Decide whether the candidate answer is supported by the supplied question and evidence.

Question:
{question}

Evidence:
{evidence}

Candidate answer:
{candidate_answer}

Is the candidate answer hallucinated?
Reply with exactly one label: hallucinated or supported.
Label:"""


# ============================================================
# Start
# ============================================================

print()
print("=" * 72)
print("MED-HALT LORA ON NATURAL-LANGUAGE PUBMEDQA V3 TEST")
print("=" * 72)


# ============================================================
# Verify dataset
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    dtype={"pubmed_id": str},
)

if len(df) != 100:
    raise ValueError(
        f"Expected exactly 100 rows, found {len(df)}."
    )

label_counts = (
    df["expected_medhalt_label"]
    .value_counts()
    .to_dict()
)

if label_counts.get("supported", 0) != 50:
    raise ValueError(
        "Expected exactly 50 supported cases."
    )

if label_counts.get("hallucinated", 0) != 50:
    raise ValueError(
        "Expected exactly 50 hallucinated cases."
    )

print()
print("Dataset verified.")
print("Rows:          100")
print("Supported:      50")
print("Hallucinated:   50")


# ============================================================
# Verify adapter
# ============================================================

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"Adapter folder not found:\n{ADAPTER_PATH}"
    )

if not (
    (ADAPTER_PATH / "adapter_model.safetensors").exists()
    or
    (ADAPTER_PATH / "adapter_model.bin").exists()
):
    raise FileNotFoundError(
        "Adapter folder exists, but adapter weights were not found."
    )

print()
print("Med-HALT adapter:")
print(ADAPTER_PATH)


# ============================================================
# GPU
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
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# Base model
# ============================================================

print()
print("Loading Qwen2.5-3B base model...")

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

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

base_model.eval()


# ============================================================
# Attach selected Med-HALT LoRA
# ============================================================

print()
print("Attaching validation-best Med-HALT LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
)

model.eval()

print()
print(
    "Qwen2.5-3B + Med-HALT LoRA loaded successfully."
)


# ============================================================
# Evaluate
# ============================================================

print()
print("=" * 72)
print("RUNNING MED-HALT LORA ON 100 NATURAL-LANGUAGE CASES")
print("=" * 72)
print()

results = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Med-HALT LoRA natural V3 evaluation",
):

    evaluation_prompt = (
        build_evaluation_prompt(
            question=row[
                "research_question"
            ],
            evidence=row[
                "evidence"
            ],
            candidate_answer=row[
                "candidate_answer"
            ],
        )
    )

    prediction, scores = (
        score_candidate_labels(
            model,
            tokenizer,
            evaluation_prompt,
        )
    )

    true_label = row[
        "expected_medhalt_label"
    ]

    correct = (
        prediction == true_label
    )

    confidence_margin = abs(
        scores["supported"]
        - scores["hallucinated"]
    )

    results.append(
        {
            "natural_case_id":
                int(row["natural_case_id"]),

            "source_sample_id":
                int(row["source_sample_id"]),

            "pubmed_id":
                row["pubmed_id"],

            "pubmedqa_ground_truth":
                row["pubmedqa_ground_truth"],

            "generation_mode":
                row["generation_mode"],

            "expected_medhalt_label":
                true_label,

            "predicted_medhalt_label":
                prediction,

            "correct":
                correct,

            "score_supported":
                scores["supported"],

            "score_hallucinated":
                scores["hallucinated"],

            "confidence_margin":
                confidence_margin,

            "research_question":
                row["research_question"],

            "evidence":
                row["evidence"],

            "candidate_answer":
                row["candidate_answer"],
        }
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# Metrics
# ============================================================

y_true = result_df[
    "expected_medhalt_label"
]

y_pred = result_df[
    "predicted_medhalt_label"
]

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
)

macro_f1 = f1.mean()

supported_precision = precision[0]
hallucination_precision = precision[1]

supported_recall = recall[0]
hallucination_recall = recall[1]

correct_count = int(
    result_df["correct"].sum()
)


# ============================================================
# Error counts
# ============================================================

missed_hallucinations = int(
    (
        (
            result_df[
                "expected_medhalt_label"
            ]
            == "hallucinated"
        )
        &
        (
            result_df[
                "predicted_medhalt_label"
            ]
            == "supported"
        )
    ).sum()
)

supported_flagged = int(
    (
        (
            result_df[
                "expected_medhalt_label"
            ]
            == "supported"
        )
        &
        (
            result_df[
                "predicted_medhalt_label"
            ]
            == "hallucinated"
        )
    ).sum()
)


# ============================================================
# Prediction distribution
# ============================================================

predicted_supported = int(
    (
        result_df[
            "predicted_medhalt_label"
        ]
        == "supported"
    ).sum()
)

predicted_hallucinated = int(
    (
        result_df[
            "predicted_medhalt_label"
        ]
        == "hallucinated"
    ).sum()
)


# ============================================================
# Confusion matrix
# ============================================================

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS,
)

matrix_df = pd.DataFrame(
    matrix,
    index=[
        "True SUPPORTED",
        "True HALLUCINATED",
    ],
    columns=[
        "Pred SUPPORTED",
        "Pred HALLUCINATED",
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
                "Qwen2.5-3B + Med-HALT LoRA",

            "evaluation":
                (
                    "PubMedQA-derived natural-language "
                    "counterfactual V3 stress test"
                ),

            "rows":
                len(result_df),

            "correct":
                correct_count,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "supported_precision":
                supported_precision,

            "supported_recall":
                supported_recall,

            "hallucination_precision":
                hallucination_precision,

            "hallucination_recall":
                hallucination_recall,

            "missed_hallucinations":
                missed_hallucinations,

            "supported_flagged_as_hallucinations":
                supported_flagged,

            "predicted_supported":
                predicted_supported,

            "predicted_hallucinated":
                predicted_hallucinated,

            "adapter":
                str(ADAPTER_PATH),
        }
    ]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Screenshot-ready output
# ============================================================

print()
print()
print("=" * 72)
print("MED-HALT LORA NATURAL-LANGUAGE V3 RESULTS")
print("=" * 72)

print(
    f"Evaluation rows:             "
    f"{len(result_df)}"
)

print(
    f"Correct predictions:        "
    f"{correct_count}/{len(result_df)}"
)

print(
    f"Accuracy:                   "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro-F1:                   "
    f"{macro_f1:.3f}"
)

print()

print(
    f"Supported recall:           "
    f"{supported_recall * 100:.1f}%"
)

print(
    f"Hallucination recall:       "
    f"{hallucination_recall * 100:.1f}%"
)

print()

print(
    f"Missed hallucinations:      "
    f"{missed_hallucinations}"
)

print(
    f"Supported answers flagged:  "
    f"{supported_flagged}"
)

print()

print("Prediction distribution:")

print(
    f"SUPPORTED:                  "
    f"{predicted_supported}"
)

print(
    f"HALLUCINATED:               "
    f"{predicted_hallucinated}"
)

print()

print("Confusion matrix:")
print(
    matrix_df.to_string()
)

print()

print("Prediction CSV:")
print(
    PREDICTION_FILE
)

print()

print("Summary CSV:")
print(
    SUMMARY_FILE
)

print()

print(
    "Important: this is a constructed "
    "natural-language counterfactual stress test."
)

print(
    "It is not an official Med-HALT benchmark "
    "and it is not clinical accuracy."
)

print("=" * 72)


# ============================================================
# Cleanup
# ============================================================

del model
del base_model

gc.collect()

torch.cuda.empty_cache()