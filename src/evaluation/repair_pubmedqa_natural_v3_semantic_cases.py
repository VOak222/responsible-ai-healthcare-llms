import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_v3_100.csv"
)

BACKUP_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "pubmedqa_natural_candidate_test_v3_before_semantic_repair.csv"
)


# ============================================================
# Backup original V3
# ============================================================

if not BACKUP_FILE.exists():

    shutil.copy2(
        DATA_FILE,
        BACKUP_FILE,
    )

    print()
    print("Backup created:")
    print(BACKUP_FILE)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    dtype={"pubmed_id": str},
)


# ============================================================
# Human-audited counterfactual repairs
#
# These four generated hallucinated candidates were internally
# inconsistent or did not cleanly support the opposite stance.
# ============================================================

repairs = {

    43: (
        "Yes. Indigenous Australians show evidence of premature ageing, "
        "with age-related mortality patterns occurring substantially earlier "
        "than in non-Indigenous Australians. This indicates that they reach "
        "comparable age-related health risks at younger ages."
    ),

    65: (
        "No. Prognosis in polymyalgia rheumatica cannot be reliably predicted "
        "at disease onset. Baseline plasma viscosity and initial prednisolone "
        "dose do not independently predict the duration of steroid therapy."
    ),

    71: (
        "Yes. Differences in carotid endarterectomy utilisation are explained "
        "by population need. Districts with greater stroke burden have higher "
        "utilisation rates regardless of service location or socioeconomic factors."
    ),

    87: (
        "No. 99mTc-WBC scintigraphy and SBFT do not reliably identify or exclude "
        "inflammatory bowel disease in children. Their results show poor agreement "
        "with the eventual clinical diagnosis."
    ),
}


# ============================================================
# Add audit column if needed
# ============================================================

if "semantic_audit_status" not in df.columns:

    df[
        "semantic_audit_status"
    ] = "generated_not_manually_repaired"


# ============================================================
# Apply only the four repairs
# ============================================================

for case_id, new_answer in repairs.items():

    mask = (
        df["natural_case_id"]
        == case_id
    )

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for case {case_id}."
        )

    expected_label = (
        df.loc[
            mask,
            "expected_medhalt_label",
        ]
        .iloc[0]
    )

    if expected_label != "hallucinated":
        raise ValueError(
            f"Case {case_id} is not a hallucinated case."
        )

    print()
    print("=" * 72)
    print(f"CASE {case_id}")
    print("=" * 72)

    print()
    print("OLD:")
    print(
        df.loc[
            mask,
            "candidate_answer",
        ].iloc[0]
    )

    print()
    print("NEW:")
    print(new_answer)

    df.loc[
        mask,
        "candidate_answer",
    ] = new_answer

    df.loc[
        mask,
        "semantic_audit_status",
    ] = "human_audited_counterfactual_repair"


# ============================================================
# Verify structure
# ============================================================

if len(df) != 100:
    raise ValueError(
        f"Expected 100 rows, found {len(df)}."
    )

supported = int(
    (
        df["expected_medhalt_label"]
        == "supported"
    ).sum()
)

hallucinated = int(
    (
        df["expected_medhalt_label"]
        == "hallucinated"
    ).sum()
)

if supported != 50 or hallucinated != 50:
    raise ValueError(
        "Expected 50 supported and 50 hallucinated rows."
    )


# ============================================================
# Save repaired final V3 dataset
# ============================================================

df.to_csv(
    DATA_FILE,
    index=False,
    encoding="utf-8-sig",
)


print()
print()
print("=" * 72)
print("V3 SEMANTIC REPAIR COMPLETE")
print("=" * 72)

print(
    f"Rows:                       {len(df)}"
)

print(
    f"Supported:                  {supported}"
)

print(
    f"Hallucinated:               {hallucinated}"
)

print(
    "Human-audited repairs:      4"
)

print()

print("Repaired case IDs:")
print("43, 65, 71, 87")

print()

print("Updated dataset:")
print(DATA_FILE)

print()

print("Original V3 backup:")
print(BACKUP_FILE)

print("=" * 72)