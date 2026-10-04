from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "pubmedqa_natural_candidate_v3"
)

BASE_FILE = (
    RESULT_DIR
    / "base_qwen_natural_v3_predictions.csv"
)

LORA_FILE = (
    RESULT_DIR
    / "medhalt_lora_natural_v3_predictions.csv"
)

OUTPUT_CSV = (
    RESULT_DIR
    / "natural_v3_base_vs_medhalt_lora_comparison.csv"
)

OUTPUT_XLSX = (
    RESULT_DIR
    / "natural_v3_base_vs_medhalt_lora_comparison.xlsx"
)


# ============================================================
# Load
# ============================================================

base = pd.read_csv(
    BASE_FILE,
    dtype={"pubmed_id": str},
)

lora = pd.read_csv(
    LORA_FILE,
    dtype={"pubmed_id": str},
)


if len(base) != 100 or len(lora) != 100:
    raise ValueError(
        "Expected 100 rows in both prediction files."
    )


# ============================================================
# Keep and rename relevant columns
# ============================================================

base_small = base[
    [
        "natural_case_id",
        "base_qwen_prediction",
        "correct",
        "score_supported",
        "score_hallucinated",
        "confidence_margin",
    ]
].copy()

base_small = base_small.rename(
    columns={
        "correct":
            "base_correct",

        "score_supported":
            "base_score_supported",

        "score_hallucinated":
            "base_score_hallucinated",

        "confidence_margin":
            "base_confidence_margin",
    }
)


lora_small = lora[
    [
        "natural_case_id",
        "predicted_medhalt_label",
        "correct",
        "score_supported",
        "score_hallucinated",
        "confidence_margin",
    ]
].copy()

lora_small = lora_small.rename(
    columns={
        "predicted_medhalt_label":
            "medhalt_lora_prediction",

        "correct":
            "lora_correct",

        "score_supported":
            "lora_score_supported",

        "score_hallucinated":
            "lora_score_hallucinated",

        "confidence_margin":
            "lora_confidence_margin",
    }
)


# ============================================================
# Use descriptive columns from Base file
# ============================================================

details = base[
    [
        "natural_case_id",
        "source_sample_id",
        "pubmed_id",
        "pubmedqa_ground_truth",
        "generation_mode",
        "expected_medhalt_label",
        "research_question",
        "evidence",
        "candidate_answer",
    ]
].copy()


# ============================================================
# Merge
# ============================================================

comparison = (
    details
    .merge(
        base_small,
        on="natural_case_id",
        how="inner",
    )
    .merge(
        lora_small,
        on="natural_case_id",
        how="inner",
    )
)


if len(comparison) != 100:
    raise ValueError(
        f"Expected 100 merged rows, found {len(comparison)}."
    )


# ============================================================
# Improvement categories
# ============================================================

def change_category(row):

    base_correct = bool(
        row["base_correct"]
    )

    lora_correct = bool(
        row["lora_correct"]
    )

    if (
        not base_correct
        and lora_correct
    ):
        return "Fixed by LoRA"

    if (
        base_correct
        and not lora_correct
    ):
        return "Lost after LoRA"

    if (
        base_correct
        and lora_correct
    ):
        return "Both correct"

    return "Both wrong"


comparison[
    "comparison_outcome"
] = comparison.apply(
    change_category,
    axis=1,
)


# ============================================================
# Summary counts
# ============================================================

base_correct_count = int(
    comparison[
        "base_correct"
    ].sum()
)

lora_correct_count = int(
    comparison[
        "lora_correct"
    ].sum()
)

fixed_by_lora = int(
    (
        comparison[
            "comparison_outcome"
        ]
        == "Fixed by LoRA"
    ).sum()
)

lost_after_lora = int(
    (
        comparison[
            "comparison_outcome"
        ]
        == "Lost after LoRA"
    ).sum()
)

both_correct = int(
    (
        comparison[
            "comparison_outcome"
        ]
        == "Both correct"
    ).sum()
)

both_wrong = int(
    (
        comparison[
            "comparison_outcome"
        ]
        == "Both wrong"
    ).sum()
)


# ============================================================
# Specific LoRA errors
# ============================================================

lora_missed_hallucinations = comparison[
    (
        comparison[
            "expected_medhalt_label"
        ]
        == "hallucinated"
    )
    &
    (
        comparison[
            "medhalt_lora_prediction"
        ]
        == "supported"
    )
].copy()

lora_false_flags = comparison[
    (
        comparison[
            "expected_medhalt_label"
        ]
        == "supported"
    )
    &
    (
        comparison[
            "medhalt_lora_prediction"
        ]
        == "hallucinated"
    )
].copy()


# ============================================================
# Summary dataframe
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Metric":
                "Base Qwen accuracy",

            "Value":
                f"{base_correct_count}/100 ({base_correct_count}%)",
        },
        {
            "Metric":
                "Med-HALT LoRA accuracy",

            "Value":
                f"{lora_correct_count}/100 ({lora_correct_count}%)",
        },
        {
            "Metric":
                "Accuracy improvement",

            "Value":
                f"+{lora_correct_count - base_correct_count} percentage points",
        },
        {
            "Metric":
                "Base errors fixed by LoRA",

            "Value":
                fixed_by_lora,
        },
        {
            "Metric":
                "Base-correct cases lost after LoRA",

            "Value":
                lost_after_lora,
        },
        {
            "Metric":
                "Both models correct",

            "Value":
                both_correct,
        },
        {
            "Metric":
                "Both models wrong",

            "Value":
                both_wrong,
        },
        {
            "Metric":
                "LoRA missed hallucinations",

            "Value":
                len(
                    lora_missed_hallucinations
                ),
        },
        {
            "Metric":
                "LoRA supported answers falsely flagged",

            "Value":
                len(
                    lora_false_flags
                ),
        },
    ]
)


# ============================================================
# Save CSV
# ============================================================

comparison.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Save Excel
# ============================================================

with pd.ExcelWriter(
    OUTPUT_XLSX,
    engine="openpyxl",
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    comparison.to_excel(
        writer,
        sheet_name="All_100_Cases",
        index=False,
    )

    comparison[
        comparison[
            "comparison_outcome"
        ]
        == "Fixed by LoRA"
    ].to_excel(
        writer,
        sheet_name="Fixed_by_LoRA",
        index=False,
    )

    comparison[
        comparison[
            "comparison_outcome"
        ]
        == "Lost after LoRA"
    ].to_excel(
        writer,
        sheet_name="Lost_after_LoRA",
        index=False,
    )

    lora_missed_hallucinations.to_excel(
        writer,
        sheet_name="Missed_Hallucinations",
        index=False,
    )

    lora_false_flags.to_excel(
        writer,
        sheet_name="False_Hallucination_Flags",
        index=False,
    )

    workbook = writer.book

    for worksheet in workbook.worksheets:

        worksheet.freeze_panes = "A2"

        for cell in worksheet[1]:
            cell.font = cell.font.copy(
                bold=True
            )

        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = (
                    cell.alignment.copy(
                        vertical="top",
                        wrap_text=True,
                    )
                )

        for column in worksheet.columns:

            letter = column[0].column_letter

            max_length = 0

            for cell in column[:50]:

                if cell.value is not None:

                    max_length = max(
                        max_length,
                        len(str(cell.value)),
                    )

            worksheet.column_dimensions[
                letter
            ].width = min(
                max(max_length + 2, 12),
                65,
            )


# ============================================================
# Screenshot-ready output
# ============================================================

print()
print("=" * 72)
print("NATURAL-LANGUAGE V3 BASE VS MED-HALT LORA COMPARISON")
print("=" * 72)

print()
print(
    f"Base Qwen correct:                "
    f"{base_correct_count}/100"
)

print(
    f"Med-HALT LoRA correct:            "
    f"{lora_correct_count}/100"
)

print(
    f"Accuracy improvement:             "
    f"+{lora_correct_count - base_correct_count} percentage points"
)

print()

print(
    f"Base errors fixed by LoRA:        "
    f"{fixed_by_lora}"
)

print(
    f"Base-correct cases lost by LoRA:  "
    f"{lost_after_lora}"
)

print(
    f"Both models correct:              "
    f"{both_correct}"
)

print(
    f"Both models wrong:                "
    f"{both_wrong}"
)

print()

print(
    f"LoRA missed hallucinations:       "
    f"{len(lora_missed_hallucinations)}"
)

print(
    f"LoRA false hallucination flags:   "
    f"{len(lora_false_flags)}"
)

print()
print("CSV:")
print(OUTPUT_CSV)

print()
print("Excel:")
print(OUTPUT_XLSX)

print()
print(
    "This comparison uses the exact same "
    "100 natural-language V3 cases for both models."
)

print("=" * 72)