from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MAX_LENGTH = 768

CANDIDATE_LABELS = [
    "supported",
    "hallucinated",
]

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_medhalt_cross_dataset_100.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "pubmedqa_medhalt_cross_dataset"
    / "prompt_truncation_audit.csv"
)


print()
print("=" * 72)
print("PUBMEDQA -> MED-HALT PROMPT TRUNCATION AUDIT")
print("=" * 72)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    dtype={"pubmed_id": str},
)

if len(df) != 100:
    raise ValueError(
        f"Expected 100 rows, found {len(df)}."
    )


# ------------------------------------------------------------
# Load same tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

print()
print(
    "Tokenizer truncation side:",
    tokenizer.truncation_side,
)

print(
    "MAX_LENGTH:",
    MAX_LENGTH,
)


# ------------------------------------------------------------
# Match original Med-HALT scorer
# ------------------------------------------------------------

candidate_token_ids = [
    tokenizer(
        f" {label}",
        add_special_tokens=False,
    )["input_ids"]
    for label in CANDIDATE_LABELS
]

longest_candidate = max(
    len(ids)
    for ids in candidate_token_ids
)

prompt_max_length = (
    MAX_LENGTH
    - longest_candidate
)

print(
    "Prompt token budget:",
    prompt_max_length,
)


# ------------------------------------------------------------
# Audit all prompts
# ------------------------------------------------------------

results = []

for _, row in df.iterrows():

    prompt = str(
        row["evaluation_prompt"]
    )

    candidate_answer = str(
        row["candidate_answer"]
    )

    # Full untruncated prompt
    full_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    # Exact truncation logic used by Med-HALT scoring
    truncated_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]

    decoded = tokenizer.decode(
        truncated_ids,
        skip_special_tokens=True,
    )

    full_length = len(
        full_ids
    )

    was_truncated = (
        full_length
        > prompt_max_length
    )

    candidate_header_visible = (
        "Candidate answer:"
        in decoded
    )

    candidate_text_visible = (
        candidate_answer
        in decoded
    )

    judge_question_visible = (
        "Is the candidate answer hallucinated?"
        in decoded
    )

    reply_instruction_visible = (
        "Reply with exactly one label:"
        in decoded
    )

    final_label_marker_visible = (
        "Label:"
        in decoded
    )

    fully_usable = (
        candidate_header_visible
        and candidate_text_visible
        and judge_question_visible
        and reply_instruction_visible
        and final_label_marker_visible
    )

    results.append(
        {
            "cross_case_id":
                int(row["cross_case_id"]),

            "pubmed_id":
                row["pubmed_id"],

            "expected_medhalt_label":
                row[
                    "expected_medhalt_label"
                ],

            "case_type":
                row["case_type"],

            "full_prompt_tokens":
                full_length,

            "prompt_token_budget":
                prompt_max_length,

            "tokens_over_budget":
                max(
                    0,
                    full_length
                    - prompt_max_length,
                ),

            "was_truncated":
                was_truncated,

            "candidate_header_visible":
                candidate_header_visible,

            "candidate_text_visible":
                candidate_text_visible,

            "judge_question_visible":
                judge_question_visible,

            "reply_instruction_visible":
                reply_instruction_visible,

            "final_label_marker_visible":
                final_label_marker_visible,

            "fully_usable":
                fully_usable,
        }
    )


audit_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Calculate totals
# ------------------------------------------------------------

truncated_count = int(
    audit_df[
        "was_truncated"
    ].sum()
)

candidate_missing = int(
    (
        ~audit_df[
            "candidate_text_visible"
        ]
    ).sum()
)

judge_question_missing = int(
    (
        ~audit_df[
            "judge_question_visible"
        ]
    ).sum()
)

label_marker_missing = int(
    (
        ~audit_df[
            "final_label_marker_visible"
        ]
    ).sum()
)

fully_usable = int(
    audit_df[
        "fully_usable"
    ].sum()
)

not_fully_usable = (
    len(audit_df)
    - fully_usable
)


# ------------------------------------------------------------
# Save detailed audit
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# Print screenshot-ready summary
# ------------------------------------------------------------

print()
print("=" * 72)
print("TRUNCATION AUDIT RESULTS")
print("=" * 72)

print(
    f"Total prompts:                    "
    f"{len(audit_df)}"
)

print(
    f"Prompts exceeding token budget:  "
    f"{truncated_count}"
)

print(
    f"Candidate answer missing:        "
    f"{candidate_missing}"
)

print(
    f"Judge question missing:          "
    f"{judge_question_missing}"
)

print(
    f"Final Label marker missing:      "
    f"{label_marker_missing}"
)

print()
print(
    f"Fully usable prompts:            "
    f"{fully_usable}/"
    f"{len(audit_df)}"
)

print(
    f"Not fully usable:                "
    f"{not_fully_usable}/"
    f"{len(audit_df)}"
)

print()

print(
    f"Shortest full prompt:            "
    f"{audit_df['full_prompt_tokens'].min()} tokens"
)

print(
    f"Average full prompt:             "
    f"{audit_df['full_prompt_tokens'].mean():.1f} tokens"
)

print(
    f"Longest full prompt:             "
    f"{audit_df['full_prompt_tokens'].max()} tokens"
)

print()

print(
    "Detailed audit CSV:"
)
print(
    OUTPUT_FILE
)

print()


# ------------------------------------------------------------
# Show affected examples
# ------------------------------------------------------------

affected = audit_df[
    ~audit_df["fully_usable"]
]

if len(affected) > 0:

    print(
        "First affected cases:"
    )

    print(
        affected[
            [
                "cross_case_id",
                "pubmed_id",
                "expected_medhalt_label",
                "full_prompt_tokens",
                "candidate_text_visible",
                "judge_question_visible",
                "final_label_marker_visible",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "All 100 prompts retained the candidate "
        "and judging instruction."
    )


print()
print("=" * 72)