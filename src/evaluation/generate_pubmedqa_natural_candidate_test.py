import gc
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_medhalt_cross_dataset_100.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_100.csv"
)

TEMP_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_progress.csv"
)


# ============================================================
# Helper
# ============================================================

def opposite_label(label):
    label = label.strip().lower()

    if label == "yes":
        return "no"

    if label == "no":
        return "yes"

    raise ValueError(
        f"Unexpected PubMedQA label: {label}"
    )


# ============================================================
# Load source questions
# ============================================================

print()
print("=" * 72)
print("GENERATING NATURAL PUBMEDQA CANDIDATE ANSWERS")
print("=" * 72)

source_df = pd.read_csv(
    SOURCE_FILE,
    dtype={"pubmed_id": str},
)

# Each PubMedQA question appears twice in the earlier
# cross-dataset test. Keep only one copy per source question.
questions = (
    source_df[
        [
            "source_sample_id",
            "pubmed_id",
            "pubmedqa_ground_truth",
            "research_question",
            "evidence",
        ]
    ]
    .drop_duplicates(
        subset=["source_sample_id"]
    )
    .sort_values(
        "source_sample_id"
    )
    .reset_index(drop=True)
)

if len(questions) != 50:
    raise ValueError(
        f"Expected 50 unique questions, found {len(questions)}."
    )

print()
print("Source questions verified: 50")
print("Each question will produce:")
print("  1 natural supported candidate")
print("  1 natural hallucinated candidate")
print("Total expected cases: 100")


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


# ============================================================
# Load Base Qwen
# ============================================================

print()
print("Loading Qwen2.5-3B-Instruct as candidate generator...")

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

model.eval()

print()
print("Generator loaded successfully.")


# ============================================================
# Generation function
# ============================================================

@torch.inference_mode()
def generate_answer(
    question,
    evidence,
    desired_conclusion,
    mode,
):

    conclusion_word = (
        "Yes"
        if desired_conclusion == "yes"
        else "No"
    )

    if mode == "supported":

        instruction = f"""
You are generating a realistic AI answer for a biomedical research question.

Use only the supplied research evidence.

The correct conclusion for this controlled case is:
{conclusion_word}

Write a natural answer of 2 to 3 sentences.

Requirements:
- Begin the answer exactly with "{conclusion_word}."
- Explain the conclusion naturally using the study findings.
- Do not mention these instructions.
- Do not mention labels, benchmarks, datasets, or evaluation.
- Do not say "according to the supplied evidence".
- Do not invent facts that are not supported by the abstract.
"""

    elif mode == "hallucinated":

        instruction = f"""
You are generating a deliberately incorrect but realistic-looking AI answer
for a controlled hallucination-detection experiment.

For this test case, the answer must incorrectly conclude:
{conclusion_word}

Write a plausible natural answer of 2 to 3 sentences.

Requirements:
- Begin the answer exactly with "{conclusion_word}."
- Make the explanation support that incorrect conclusion.
- Keep the answer medically plausible in wording.
- Do not reveal that the answer is intentionally incorrect.
- Do not mention these instructions.
- Do not mention labels, benchmarks, datasets, or evaluation.
"""

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    user_prompt = f"""
{instruction}

Research question:
{question}

Research evidence:
{evidence}

Candidate answer:
""".strip()

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer


# ============================================================
# Resume support
# ============================================================

existing_rows = []

completed_keys = set()

if TEMP_FILE.exists():

    existing_df = pd.read_csv(
        TEMP_FILE,
        dtype={"pubmed_id": str},
    )

    existing_rows = (
        existing_df
        .to_dict("records")
    )

    for row in existing_rows:

        completed_keys.add(
            (
                int(row["source_sample_id"]),
                row["generation_mode"],
            )
        )

    print()
    print(
        f"Resuming from existing progress: "
        f"{len(existing_rows)} cases already generated."
    )


results = existing_rows.copy()


# ============================================================
# Generate candidates
# ============================================================

print()
print("=" * 72)
print("GENERATING 100 NATURAL-LANGUAGE CANDIDATES")
print("=" * 72)
print()

for _, row in tqdm(
    questions.iterrows(),
    total=len(questions),
    desc="Questions",
):

    source_sample_id = int(
        row["source_sample_id"]
    )

    ground_truth = (
        row["pubmedqa_ground_truth"]
        .strip()
        .lower()
    )

    question = row[
        "research_question"
    ]

    evidence = row[
        "evidence"
    ]

    pubmed_id = row[
        "pubmed_id"
    ]


    # --------------------------------------------------------
    # Supported candidate
    # --------------------------------------------------------

    supported_key = (
        source_sample_id,
        "supported",
    )

    if supported_key not in completed_keys:

        supported_conclusion = (
            ground_truth
        )

        supported_answer = generate_answer(
            question=question,
            evidence=evidence,
            desired_conclusion=supported_conclusion,
            mode="supported",
        )

        expected_prefix = (
            "Yes."
            if supported_conclusion == "yes"
            else "No."
        )

        results.append(
            {
                "source_sample_id":
                    source_sample_id,

                "pubmed_id":
                    pubmed_id,

                "pubmedqa_ground_truth":
                    ground_truth,

                "research_question":
                    question,

                "evidence":
                    evidence,

                "candidate_answer":
                    supported_answer,

                "candidate_conclusion":
                    supported_conclusion,

                "expected_medhalt_label":
                    "supported",

                "generation_mode":
                    "supported",

                "expected_prefix":
                    expected_prefix,

                "format_ok":
                    supported_answer.startswith(
                        expected_prefix
                    ),

                "generator_model":
                    MODEL_ID,
            }
        )

        pd.DataFrame(
            results
        ).to_csv(
            TEMP_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    # --------------------------------------------------------
    # Hallucinated candidate
    # --------------------------------------------------------

    hallucinated_key = (
        source_sample_id,
        "hallucinated",
    )

    if hallucinated_key not in completed_keys:

        wrong_conclusion = opposite_label(
            ground_truth
        )

        hallucinated_answer = generate_answer(
            question=question,
            evidence=evidence,
            desired_conclusion=wrong_conclusion,
            mode="hallucinated",
        )

        expected_prefix = (
            "Yes."
            if wrong_conclusion == "yes"
            else "No."
        )

        results.append(
            {
                "source_sample_id":
                    source_sample_id,

                "pubmed_id":
                    pubmed_id,

                "pubmedqa_ground_truth":
                    ground_truth,

                "research_question":
                    question,

                "evidence":
                    evidence,

                "candidate_answer":
                    hallucinated_answer,

                "candidate_conclusion":
                    wrong_conclusion,

                "expected_medhalt_label":
                    "hallucinated",

                "generation_mode":
                    "hallucinated",

                "expected_prefix":
                    expected_prefix,

                "format_ok":
                    hallucinated_answer.startswith(
                        expected_prefix
                    ),

                "generator_model":
                    MODEL_ID,
            }
        )

        pd.DataFrame(
            results
        ).to_csv(
            TEMP_FILE,
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# Finalize dataset
# ============================================================

final_df = pd.DataFrame(
    results
)

final_df = (
    final_df
    .sort_values(
        [
            "source_sample_id",
            "generation_mode",
        ]
    )
    .reset_index(drop=True)
)

final_df.insert(
    0,
    "natural_case_id",
    range(
        1,
        len(final_df) + 1,
    ),
)


# ============================================================
# Verification
# ============================================================

if len(final_df) != 100:
    raise ValueError(
        f"Expected 100 generated cases, found {len(final_df)}."
    )

label_counts = (
    final_df[
        "expected_medhalt_label"
    ]
    .value_counts()
    .to_dict()
)

if label_counts.get(
    "supported",
    0,
) != 50:

    raise ValueError(
        "Expected 50 supported cases."
    )

if label_counts.get(
    "hallucinated",
    0,
) != 50:

    raise ValueError(
        "Expected 50 hallucinated cases."
    )


format_failures = int(
    (
        final_df[
            "format_ok"
        ]
        == False
    ).sum()
)


# ============================================================
# Save final dataset
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print()
print()
print("=" * 72)
print("NATURAL-LANGUAGE CANDIDATE DATASET CREATED")
print("=" * 72)

print(
    f"Unique PubMedQA questions:  "
    f"{len(questions)}"
)

print(
    f"Total candidate answers:    "
    f"{len(final_df)}"
)

print(
    f"Supported cases:            "
    f"{label_counts['supported']}"
)

print(
    f"Hallucinated cases:         "
    f"{label_counts['hallucinated']}"
)

print(
    f"Prefix/format failures:     "
    f"{format_failures}"
)

print()

print(
    "Output CSV:"
)
print(
    OUTPUT_FILE
)

print()

if format_failures == 0:

    print(
        "All generated candidates begin with "
        "the intended Yes/No conclusion."
    )

else:

    print(
        "WARNING: Some generated answers did not "
        "follow the required Yes/No prefix."
    )

    print(
        "We must review those rows before evaluation."
    )

print()

print(
    "This is a constructed LLM-generated "
    "cross-dataset evaluation set."
)

print(
    "It is not an official PubMedQA or Med-HALT benchmark."
)

print("=" * 72)


# ============================================================
# Cleanup
# ============================================================

del model

gc.collect()

torch.cuda.empty_cache()