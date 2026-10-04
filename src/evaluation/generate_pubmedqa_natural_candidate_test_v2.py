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
    / "pubmedqa_natural_candidate_test_v2_100.csv"
)

PROGRESS_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_v2_progress.csv"
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
        f"Unexpected PubMedQA label: {label}"
    )


def expected_prefix(label):

    if label == "yes":
        return "Yes."

    if label == "no":
        return "No."

    raise ValueError(label)


# ============================================================
# Load the same 50 PubMedQA questions
# ============================================================

print()
print("=" * 72)
print("GENERATING NATURAL PUBMEDQA CANDIDATE TEST - VERSION 2")
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
print("Source questions: 50")
print("Expected final cases: 100")
print("  50 supported")
print("  50 hallucinated")


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
# Generation
# ============================================================

@torch.inference_mode()
def generate_answer(
    question,
    evidence,
    desired_conclusion,
    mode,
):

    conclusion = (
        "Yes"
        if desired_conclusion == "yes"
        else "No"
    )

    if mode == "supported":

        instruction = f"""
Create a realistic AI-generated biomedical answer.

The answer must accurately reflect the supplied research evidence.

The correct overall conclusion is:
{conclusion}

Requirements:
1. Begin exactly with "{conclusion}."
2. Write exactly 2 complete sentences.
3. Explain the study finding naturally.
4. Use only information supported by the evidence.
5. Keep the answer internally consistent with the {conclusion} conclusion.
6. Do not mention datasets, labels, benchmarks, prompts, or evaluation.
7. Do not mention that you were instructed to produce a particular conclusion.
8. End with a complete sentence.
""".strip()

    elif mode == "hallucinated":

        instruction = f"""
Create a realistic but deliberately incorrect AI-generated biomedical answer
for a controlled hallucination-detection experiment.

The answer must consistently argue the following incorrect conclusion:
{conclusion}

Requirements:
1. Begin exactly with "{conclusion}."
2. Write exactly 2 complete sentences.
3. Every sentence must support the incorrect {conclusion} conclusion.
4. Do NOT state any study finding that supports the true opposite conclusion.
5. Do NOT correct yourself, hedge back toward the true answer, or acknowledge
   that the opposite conclusion is supported.
6. You may plausibly misinterpret, reverse, or overstate a study finding so the
   answer sounds like a realistic hallucinated AI response.
7. Keep the answer internally consistent from beginning to end.
8. Do not mention datasets, labels, benchmarks, prompts, or evaluation.
9. Do not reveal that the answer is intentionally incorrect.
10. End with a complete sentence.

Important:
If the evidence actually supports the opposite conclusion, you must NOT quote
or describe that correct finding and then contradict yourself. The entire
answer must consistently defend the requested incorrect conclusion.
""".strip()

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
        max_new_tokens=200,
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

completed = set()

if PROGRESS_FILE.exists():

    existing_df = pd.read_csv(
        PROGRESS_FILE,
        dtype={"pubmed_id": str},
    )

    existing_rows = (
        existing_df
        .to_dict("records")
    )

    for row in existing_rows:

        completed.add(
            (
                int(row["source_sample_id"]),
                row["generation_mode"],
            )
        )

    print()
    print(
        f"Resuming: "
        f"{len(existing_rows)} cases already generated."
    )


results = existing_rows.copy()


# ============================================================
# Generate 100 cases
# ============================================================

print()
print("=" * 72)
print("GENERATING VERSION-2 NATURAL CANDIDATES")
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
    # Supported answer
    # --------------------------------------------------------

    key = (
        source_sample_id,
        "supported",
    )

    if key not in completed:

        supported_conclusion = (
            ground_truth
        )

        answer = generate_answer(
            question,
            evidence,
            supported_conclusion,
            "supported",
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
                    answer,

                "candidate_conclusion":
                    supported_conclusion,

                "expected_medhalt_label":
                    "supported",

                "generation_mode":
                    "supported",

                "expected_prefix":
                    expected_prefix(
                        supported_conclusion
                    ),

                "prefix_ok":
                    answer.startswith(
                        expected_prefix(
                            supported_conclusion
                        )
                    ),

                "ends_complete":
                    answer.rstrip().endswith(
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

        pd.DataFrame(
            results
        ).to_csv(
            PROGRESS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    # --------------------------------------------------------
    # Hallucinated answer
    # --------------------------------------------------------

    key = (
        source_sample_id,
        "hallucinated",
    )

    if key not in completed:

        wrong_conclusion = (
            opposite_label(
                ground_truth
            )
        )

        answer = generate_answer(
            question,
            evidence,
            wrong_conclusion,
            "hallucinated",
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
                    answer,

                "candidate_conclusion":
                    wrong_conclusion,

                "expected_medhalt_label":
                    "hallucinated",

                "generation_mode":
                    "hallucinated",

                "expected_prefix":
                    expected_prefix(
                        wrong_conclusion
                    ),

                "prefix_ok":
                    answer.startswith(
                        expected_prefix(
                            wrong_conclusion
                        )
                    ),

                "ends_complete":
                    answer.rstrip().endswith(
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
# Verify
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

if counts.get(
    "supported",
    0,
) != 50:
    raise ValueError(
        "Expected 50 supported cases."
    )

if counts.get(
    "hallucinated",
    0,
) != 50:
    raise ValueError(
        "Expected 50 hallucinated cases."
    )

prefix_failures = int(
    (
        final_df[
            "prefix_ok"
        ]
        == False
    ).sum()
)

incomplete_answers = int(
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
# Summary
# ============================================================

print()
print()
print("=" * 72)
print("NATURAL-LANGUAGE CANDIDATE DATASET V2 CREATED")
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
    f"{counts['supported']}"
)

print(
    f"Hallucinated cases:        "
    f"{counts['hallucinated']}"
)

print(
    f"Prefix failures:           "
    f"{prefix_failures}"
)

print(
    f"Incomplete endings:        "
    f"{incomplete_answers}"
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
    "This dataset still requires manual spot-checking "
    "before model evaluation."
)

print("=" * 72)


# ============================================================
# Cleanup
# ============================================================

del model

gc.collect()

torch.cuda.empty_cache()