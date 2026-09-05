from pathlib import Path
import importlib.util
import platform
import shutil
import sys
import pandas as pd

OUT_DIR = Path("results/comparison")
REPORT_DIR = Path("reports/future_work")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "week5_model_dataset_expansion_feasibility.csv"
REPORT_PATH = REPORT_DIR / "week5_model_dataset_expansion_plan.md"


def package_available(name):
    return importlib.util.find_spec(name) is not None


def get_torch_info():
    if not package_available("torch"):
        return {
            "torch_available": False,
            "cuda_available": False,
            "gpu_name": "not available",
        }

    import torch

    gpu_name = "not available"
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def get_system_info():
    disk = shutil.disk_usage(".")
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "free_disk_gb": round(disk.free / (1024 ** 3), 2),
    }


def main():
    torch_info = get_torch_info()
    system_info = get_system_info()

    packages = {
        "torch": package_available("torch"),
        "transformers": package_available("transformers"),
        "accelerate": package_available("accelerate"),
        "datasets": package_available("datasets"),
        "pandas": package_available("pandas"),
        "sentence_transformers": package_available("sentence_transformers"),
    }

    rows = [
        {
            "candidate": "Qwen/Qwen2.5-3B-Instruct",
            "type": "model",
            "purpose": "Stronger local LLM comparison after Qwen 0.5B and Qwen 1.5B",
            "status": "planned",
            "readiness_check": "Use first if local memory and disk are sufficient",
            "risk": "May be slower or heavier than Qwen 1.5B",
            "next_action": "Run a 5-row smoke test before full evaluation",
        },
        {
            "candidate": "microsoft/Phi-3.5-mini-instruct",
            "type": "model",
            "purpose": "Alternative small open model for comparison",
            "status": "backup candidate",
            "readiness_check": "Use if Qwen2.5-3B is too slow or unstable locally",
            "risk": "Different chat format and output behavior may need parsing adjustment",
            "next_action": "Keep as second model option after Qwen2.5-3B test",
        },
        {
            "candidate": "MedMCQA",
            "type": "dataset",
            "purpose": "Additional medical multiple-choice reasoning benchmark",
            "status": "planned",
            "readiness_check": "Use a controlled sample first instead of the full dataset",
            "risk": "Needs clean answer-choice formatting and evaluation parser",
            "next_action": "Create a small sample loader and validate columns",
        },
        {
            "candidate": "Final governance/reporting layer",
            "type": "project feature",
            "purpose": "Combine model result, trust score, risk reason, action, and review decision",
            "status": "planned",
            "readiness_check": "Build after Qwen2.5-3B and MedMCQA are tested",
            "risk": "Should stay clear and not become overcomplicated",
            "next_action": "Extend project-level view after new model/dataset results exist",
        },
    ]

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_CSV, index=False)

    package_lines = "\n".join(
        f"| {name} | {'available' if available else 'missing'} |"
        for name, available in packages.items()
    )

    report = f"""# Week 5 Future Work: Model and Dataset Expansion Plan

## Purpose

This note prepares the next phase of the project without changing this week's report or PPT.

The goal is to make the project stronger by adding one larger local model, one additional medical reasoning dataset, and a final governance/reporting layer.

## Environment Check

| Check | Value |
|---|---|
| Python version | {system_info["python_version"]} |
| Platform | {system_info["platform"]} |
| Free disk space | {system_info["free_disk_gb"]} GB |
| PyTorch available | {torch_info["torch_available"]} |
| CUDA available | {torch_info["cuda_available"]} |
| GPU | {torch_info["gpu_name"]} |

## Package Check

| Package | Status |
|---|---|
{package_lines}

## Planned Additions

| Addition | Why It Matters |
|---|---|
| Qwen2.5-3B-Instruct | Adds a stronger local model comparison beyond Qwen 0.5B and Qwen 1.5B. |
| Phi-3.5-mini-instruct | Gives a backup small-model option if Qwen2.5-3B is too heavy locally. |
| MedMCQA | Adds a broader medical reasoning dataset for multiple-choice healthcare evaluation. |
| Governance/reporting layer | Makes the final project look like a professional evaluation framework, not only model testing. |

## Implementation Approach

The next step should be a small smoke test, not a full run.

First, test Qwen2.5-3B on 5 Med-HALT rows. If it runs properly, expand to 25 rows, then 100 rows.

After that, add a small MedMCQA sample and reuse the existing trustworthiness and routing structure.

## Current Decision

These additions should be treated as Week 5 future work. They should not be added to this week's PPT or report yet, but the groundwork is now prepared.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Week 5 model and dataset expansion feasibility audit complete")
    print()
    print("Environment:")
    for key, value in {**system_info, **torch_info}.items():
        print(f"{key}: {value}")

    print()
    print("Packages:")
    for name, available in packages.items():
        print(f"{name}: {'available' if available else 'missing'}")

    print()
    print(audit.to_string(index=False))
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
