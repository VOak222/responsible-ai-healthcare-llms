import gc
from pathlib import Path

import pandas as pd
import torch
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
    / "pubmedqa_natural_candidate_test_v3_100.csv"
)

PROGRESS_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_v3_progress.csv"
)


# ============================================================
# Helpers
# ============================================================

def opposite_label(label):

    label = str(label).strip().lower()

    if label == "yes":
        return "no"

    if label == "no":
        return "yes"

    raise ValueError(
        f"Unexpected label: {label}"
    )


def prefix_for(label):

    if label == "yes":
        return "Yes."

    if label == "no":
        return "No."

    raise ValueError(label)


# ============================================================
# Load same 50 PubMedQA questions
# ============================================================

print()
print("=" * 72)
print("GENERATING NATURAL PUBMEDQA CANDIDATE TEST - VERSION 3")
print("=" * 72)

source_df = pd.read_csv(
    SOURCE_FILE,
    dtype={"pubmed_id": str},
)

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
    .sort_values("source_sample_id")
    .reset_index(drop=True)
)

if len(questions) != 50:
    raise ValueError(
        f"Expected 50 unique questions, found {len(questions)}."
    )

print()
print("Source questions:       50")
print("Expected final cases:   100")
print("Supported:              50")
print("Hallucinated:           50")


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
# Model
# ============================================================

print()
print(
    "Loading Qwen2.5-3B-Instruct "
    "as candidate generator..."
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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

model.eval()

print("Generator loaded successfully.")


# ============================================================
# Generic generation helper
# ============================================================

@torch.inference_mode()
def generate_text(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
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

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# Supported answer generation
# ============================================================

def generate_supported(
    question,
    evidence,
    ground_truth,
):

    conclusion = (
        "Yes"
        if ground_truth == "yes"
        else "No"
    )

    prompt = f"""
Write a concise AI-generated answer to the biomedical research question.

The answer must accurately represent the supplied study evidence.

The correct conclusion is {conclusion}.

STRICT RULES:
- Begin exactly with "{conclusion}."
- Write exactly 2 short complete sentences.
- Maximum 60 words total.
- Give only the central study finding needed to support the conclusion.
- Every sentence must agree with the {conclusion} conclusion.
- Do not introduce findings that point toward the opposite conclusion.
- Do not add speculation.
- Do not use "however", "although", "despite", or other reversal language.
- Do not mention prompts, labels, datasets, benchmarks, or evaluation.

Research question:
{question}

Study evidence:
{evidence}

Candidate answer:
""".strip()

    return generate_text(
        prompt
    )


# ============================================================
# Hallucinated answer generation
#
# IMPORTANT:
# This stage DOES NOT receive the original evidence.
# It only receives the supported candidate and is asked to
# produce a coherent counterfactual opposite.
# ============================================================

def generate_hallucinated(
    question,
    supported_answer,
    wrong_conclusion,
):

    conclusion = (
        "Yes"
        if wrong_conclusion == "yes"
        else "No"
    )

    prompt = f"""
Create a realistic counterfactual AI answer for a controlled
hallucination-detection experiment.

You are given a biomedical research question and a correct candidate answer.

Rewrite the answer so that it consistently supports the EXACT OPPOSITE
conclusion.

The required incorrect conclusion is {conclusion}.

STRICT RULES:
- Begin exactly with "{conclusion}."
- Write exactly 2 short complete sentences.
- Maximum 60 words total.
- Reverse the central conclusion of the correct answer.
- Every sentence must support the incorrect {conclusion} conclusion.
- Do not quote or repeat facts that support the original conclusion.
- Do not correct yourself.
- Do not hedge toward the original answer.
- Do not use "however", "although", "despite", "but", or similar
  contrast language.
- Do not say the evidence actually supports something else.
- Do not mention that the answer is intentionally incorrect.
- Do not mention prompts, labels, datasets, benchmarks, or evaluation.

Research question:
{question}

Correct candidate answer:
{supported_answer}

Counterfactual candidate answer:
""".strip()

    return generate_text(
        prompt
    )


# ============================================================
# Resume support
# ============================================================

results = []

completed = set()

if PROGRESS_FILE.exists():

    progress_df = pd.read_csv(
        PROGRESS_FILE,
        dtype={"pubmed_id": str},
    )

    results = (
        progress_df
        .to_dict("records")
    )

    for row in results:

        completed.add(
            int(
                row["source_sample_id"]
            )
        )

    print()
    print(
        f"Resuming with "
        f"{len(completed)} completed questions."
    )


# ============================================================
# Generate paired cases
# ============================================================

print()
print("=" * 72)
print("GENERATING PAIRED NATURAL-LANGUAGE CASES")
print("=" * 72)
print()

for _, row in tqdm(
    questions.iterrows(),
    total=len(questions),
    desc="Questions",
):

    source_id = int(
        row["source_sample_id"]
    )

    if source_id in completed:
        continue

    pubmed_id = row[
        "pubmed_id"
    ]

    ground_truth = (
        str(
            row[
                "pubmedqa_ground_truth"
            ]
        )
        .strip()
        .lower()
    )

    question = row[
        "research_question"
    ]

    evidence = row[
        "evidence"
    ]


    # --------------------------------------------------------
    # Generate evidence-supported answer
    # --------------------------------------------------------

    supported_answer = (
        generate_supported(
            question=question,
            evidence=evidence,
            ground_truth=ground_truth,
        )
    )


    # --------------------------------------------------------
    # Generate opposite counterfactual WITHOUT evidence
    # --------------------------------------------------------

    wrong_conclusion = (
        opposite_label(
            ground_truth
        )
    )

    hallucinated_answer = (
        generate_hallucinated(
            question=question,
            supported_answer=supported_answer,
            wrong_conclusion=wrong_conclusion,
        )
    )


    # --------------------------------------------------------
    # Save supported row
    # --------------------------------------------------------

    results.append(
        {
            "source_sample_id":
                source_id,

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
                ground_truth,

            "expected_medhalt_label":
                "supported",

            "generation_mode":
                "supported",

            "expected_prefix":
                prefix_for(
                    ground_truth
                ),

            "prefix_ok":
                supported_answer.startswith(
                    prefix_for(
                        ground_truth
                    )
                ),

            "ends_complete":
                supported_answer.rstrip().endswith(
                    (
                        ".",
                        "!",
                        "?",
                    )
                ),

            "generator_model":
                MODEL_ID,
        }
    )


    # --------------------------------------------------------
    # Save hallucinated row
    # --------------------------------------------------------

    results.append(
        {
            "source_sample_id":
                source_id,

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
                "counterfactual_hallucinated",

            "expected_prefix":
                prefix_for(
                    wrong_conclusion
                ),

            "prefix_ok":
                hallucinated_answer.startswith(
                    prefix_for(
                        wrong_conclusion
                    )
                ),

            "ends_complete":
                hallucinated_answer.rstrip().endswith(
                    (
                        ".",
                        "!",
                        "?",
                    )
                ),

            "generator_model":
                MODEL_ID,
        }
    )


    # --------------------------------------------------------
    # Save progress after each paired question
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        PROGRESS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Finalize
# ============================================================

final_df = pd.DataFrame(
    results
)

final_df = (
    final_df
    .sort_values(
        [
            "source_sample_id",
            "expected_medhalt_label",
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
        f"Expected 100 rows, found {len(final_df)}."
    )

counts = (
    final_df[
        "expected_medhalt_label"
    ]
    .value_counts()
    .to_dict()
)

prefix_failures = int(
    (
        final_df[
            "prefix_ok"
        ]
        == False
    ).sum()
)

incomplete_endings = int(
    (
        final_df[
            "ends_complete"
        ]
        == False
    ).sum()
)


# ============================================================
# Save
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Print summary
# ============================================================

print()
print()
print("=" * 72)
print("NATURAL-LANGUAGE CANDIDATE DATASET V3 CREATED")
print("=" * 72)

print(
    f"Unique questions:          "
    f"{len(questions)}"
)

print(
    f"Total candidates:          "
    f"{len(final_df)}"
)

print(
    f"Supported cases:           "
    f"{counts.get('supported', 0)}"
)

print(
    f"Hallucinated cases:        "
    f"{counts.get('hallucinated', 0)}"
)

print(
    f"Prefix failures:           "
    f"{prefix_failures}"
)

print(
    f"Incomplete endings:        "
    f"{incomplete_endings}"
)

print()

print(
    "Output:"
)
print(
    OUTPUT_FILE
)

print()

print(
    "Hallucinated answers were generated from the "
    "supported answer without exposing the original abstract."
)

print()

print(
    "Manual semantic spot-checking is still required "
    "before model evaluation."
)

print("=" * 72)


# ============================================================
# Cleanup
# ============================================================

del model

gc.collect()

torch.cuda.empty_cache()