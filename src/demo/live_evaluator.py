"""Live local evaluation using the selected validated Qwen 2.5 3B LoRA adapters."""

from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MEDHALT_ADAPTER = (
    PROJECT_DIR
    / "artifacts"
    / "adapters"
    / "qwen2_5_3b_medhalt_hallucination_best_validation"
)

PUBMEDQA_ADAPTER = (
    PROJECT_DIR
    / "artifacts"
    / "checkpoints"
    / "qwen2_5_3b_controlled_benchmark"
    / "epoch_1"
)

MANUAL_LOG_PATH = (
    PROJECT_DIR
    / "results"
    / "manual_evaluation"
    / "manual_case_log.csv"
)

MAX_LENGTH = 768
MAX_EVIDENCE_CHARS = 2600
LOW_MARGIN_THRESHOLD = 0.15


def build_pubmedqa_prompt(question, evidence):
    """Use the PubMedQA adapter for its original question-level task."""

    return f"""You are a biomedical evidence assistant.
Use only the supplied PubMed abstract evidence.
Answer with one decision: yes, no, or maybe.

Research question:
{question}

PubMed abstract evidence:
{evidence}

Decision:"""


def build_medhalt_prompt(question, evidence, candidate_answer):
    """Use the exact Med-HALT-style task format."""

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


def infer_candidate_conclusion(candidate_answer):
    """Conservatively infer an explicit yes/no/maybe stance from the answer."""

    normalized = " ".join(candidate_answer.strip().lower().split())

    if not normalized:
        return "uncertain"

    explicit_patterns = {
        "yes": [
            r"^yes\b",
            r"^answer\s*:\s*yes\b",
            r"^the answer is yes\b",
            r"^the conclusion is yes\b",
        ],
        "no": [
            r"^no\b",
            r"^answer\s*:\s*no\b",
            r"^the answer is no\b",
            r"^the conclusion is no\b",
        ],
        "maybe": [
            r"^maybe\b",
            r"^answer\s*:\s*maybe\b",
            r"^the answer is maybe\b",
            r"^the conclusion is maybe\b",
            r"^(it is|it's) (unclear|uncertain|inconclusive)\b",
            r"^(unclear|uncertain|inconclusive)\b",
        ],
    }

    for label, patterns in explicit_patterns.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label

    uncertainty_phrases = [
        "insufficient evidence",
        "not enough evidence",
        "cannot determine",
        "can't determine",
        "cannot be determined",
        "evidence is mixed",
        "evidence remains mixed",
    ]

    if any(phrase in normalized for phrase in uncertainty_phrases):
        return "maybe"

    return "uncertain"


def assess_candidate_evidence_consistency(
    evidence_conclusion,
    candidate_conclusion,
):
    """Compare an explicit candidate stance with the PubMedQA conclusion."""

    if (
        evidence_conclusion in {"yes", "no"}
        and candidate_conclusion in {"yes", "no"}
    ):
        if evidence_conclusion == candidate_conclusion:
            return (
                "consistent",
                "The candidate's explicit yes/no conclusion matches the "
                "PubMedQA evidence conclusion.",
            )

        return (
            "contradiction",
            "The candidate's explicit yes/no conclusion conflicts with the "
            "PubMedQA evidence conclusion.",
        )

    if evidence_conclusion == "maybe":
        return (
            "uncertain",
            "The PubMedQA evidence conclusion is inconclusive, so a firm "
            "candidate-evidence comparison is not appropriate.",
        )

    if candidate_conclusion == "maybe":
        return (
            "uncertain",
            "The candidate answer itself is inconclusive, so it cannot be "
            "cleanly matched to a yes/no evidence conclusion.",
        )

    return (
        "uncertain",
        "The candidate answer does not state a clear yes/no/maybe conclusion "
        "that can be compared conservatively.",
    )


def clip_evidence(evidence):
    """Keep live inputs inside the model context window."""

    evidence = evidence.strip()

    if len(evidence) <= MAX_EVIDENCE_CHARS:
        return evidence

    shortened = evidence[:MAX_EVIDENCE_CHARS].rsplit(" ", 1)[0]

    return shortened + " [truncated for model context]"


class DualAdapterEvaluator:
    """Loads Qwen once and switches between Med-HALT and PubMedQA LoRAs."""

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Live evaluation requires the local NVIDIA GPU, but CUDA "
                "was not available."
            )

        for adapter_path in [MEDHALT_ADAPTER, PUBMEDQA_ADAPTER]:
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"Adapter directory was not found: {adapter_path}"
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(MEDHALT_ADAPTER),
            adapter_name="medhalt",
        )

        model.load_adapter(
            str(PUBMEDQA_ADAPTER),
            adapter_name="pubmedqa",
        )

        model.eval()
        self.model = model

    @torch.inference_mode()
    def score_labels(self, prompt, labels, adapter_name):
        """Score fixed labels using the same likelihood method as evaluation."""

        self.load()
        self.model.set_adapter(adapter_name)

        candidate_token_ids = [
            self.tokenizer(
                f" {label}",
                add_special_tokens=False,
            )["input_ids"]
            for label in labels
        ]

        longest_candidate = max(
            len(token_ids)
            for token_ids in candidate_token_ids
        )

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH - longest_candidate,
        )["input_ids"]

        sequences = [
            prompt_ids + label_ids
            for label_ids in candidate_token_ids
        ]

        max_sequence_length = max(
            len(sequence)
            for sequence in sequences
        )

        input_ids = torch.full(
            (len(sequences), max_sequence_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros_like(input_ids)

        for row_index, sequence in enumerate(sequences):
            input_ids[row_index, :len(sequence)] = torch.tensor(sequence)
            attention_mask[row_index, :len(sequence)] = 1

        device = next(self.model.parameters()).device

        outputs = self.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )

        scores = {}

        for row_index, (label, label_token_ids) in enumerate(
            zip(labels, candidate_token_ids)
        ):
            token_log_probabilities = []

            for token_offset, token_id in enumerate(label_token_ids):
                position = len(prompt_ids) - 1 + token_offset

                token_logits = outputs.logits[
                    row_index,
                    position,
                ]

                log_probability = torch.log_softmax(
                    token_logits,
                    dim=-1,
                )[token_id]

                token_log_probabilities.append(
                    log_probability.item()
                )

            scores[label] = (
                sum(token_log_probabilities)
                / len(token_log_probabilities)
            )

        return scores

    def evaluate(self, question, candidate_answer, evidence):
        evidence = clip_evidence(evidence)

        medhalt_prompt = build_medhalt_prompt(
            question,
            evidence,
            candidate_answer,
        )

        pubmedqa_prompt = build_pubmedqa_prompt(
            question,
            evidence,
        )

        medhalt_scores = self.score_labels(
            medhalt_prompt,
            ["supported", "hallucinated"],
            "medhalt",
        )

        pubmedqa_scores = self.score_labels(
            pubmedqa_prompt,
            ["yes", "no", "maybe"],
            "pubmedqa",
        )

        medhalt_label = max(
            medhalt_scores,
            key=medhalt_scores.get,
        )

        pubmedqa_label = max(
            pubmedqa_scores,
            key=pubmedqa_scores.get,
        )

        medhalt_margin = abs(
            medhalt_scores["supported"]
            - medhalt_scores["hallucinated"]
        )

        pubmedqa_ordered_scores = sorted(
            pubmedqa_scores.values(),
            reverse=True,
        )

        pubmedqa_margin = (
            pubmedqa_ordered_scores[0]
            - pubmedqa_ordered_scores[1]
        )

        candidate_conclusion = infer_candidate_conclusion(
            candidate_answer
        )

        consistency_label, consistency_explanation = (
            assess_candidate_evidence_consistency(
                pubmedqa_label,
                candidate_conclusion,
            )
        )

        routing_action, routing_reasons = self.route_case(
            medhalt_label,
            pubmedqa_label,
            medhalt_margin,
            pubmedqa_margin,
            candidate_answer,
            consistency_label=consistency_label,
            candidate_conclusion=candidate_conclusion,
        )

        return {
            "medhalt": {
                "label": medhalt_label,
                "scores": {
                    key: round(value, 3)
                    for key, value in medhalt_scores.items()
                },
                "confidence_margin": round(medhalt_margin, 3),
            },
            "pubmedqa": {
                "label": pubmedqa_label,
                "scores": {
                    key: round(value, 3)
                    for key, value in pubmedqa_scores.items()
                },
                "confidence_margin": round(pubmedqa_margin, 3),
            },
            "candidate_conclusion": {
                "label": candidate_conclusion,
                "method": "conservative explicit-stance parser",
            },
            "consistency": {
                "label": consistency_label,
                "explanation": consistency_explanation,
            },
            "routing_action": routing_action,
            "routing_reasons": routing_reasons,
            "evidence_used": evidence,
        }

    @staticmethod
    def route_case(
        medhalt_label,
        pubmedqa_label,
        medhalt_margin,
        pubmedqa_margin,
        candidate_answer,
        consistency_label=None,
        candidate_conclusion=None,
    ):
        reasons = []

        answer_lower = candidate_answer.lower()

        safe_refusal_terms = [
            "cannot",
            "can't",
            "unable",
            "insufficient information",
            "consult",
        ]

        if medhalt_label == "hallucinated":
            reasons.append(
                "The Med-HALT safety adapter flagged the candidate answer "
                "as hallucinated."
            )

        if pubmedqa_label == "maybe":
            reasons.append(
                "The PubMedQA evidence conclusion was inconclusive "
                "(maybe), so reviewer interpretation is needed."
            )

        if consistency_label == "contradiction":
            reasons.append(
                "The candidate's explicit conclusion contradicts the "
                "PubMedQA evidence conclusion."
            )
        elif (
            consistency_label == "uncertain"
            and pubmedqa_label != "maybe"
        ):
            reasons.append(
                "Candidate-evidence consistency could not be determined "
                "confidently from an explicit yes/no conclusion."
            )

        if (
            medhalt_margin < LOW_MARGIN_THRESHOLD
            or pubmedqa_margin < LOW_MARGIN_THRESHOLD
        ):
            reasons.append(
                "At least one model signal had a low label-score margin."
            )

        if any(term in answer_lower for term in safe_refusal_terms):
            reasons.append(
                "The response may be a cautious or safe-refusal variant "
                "and needs reviewer interpretation."
            )

        if (
            medhalt_label == "hallucinated"
            or consistency_label == "contradiction"
        ):
            return "Mandatory human review", reasons

        if reasons:
            return "Human review required", reasons

        return (
            "Lower-priority human review",
            [
                "The Med-HALT adapter did not flag hallucination, and no "
                "additional review trigger fired. This research prototype "
                "never grants automatic clinical approval."
            ],
        )


def append_manual_case_log(payload):
    """Save manual exploratory cases separately from frozen benchmark results."""

    MANUAL_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "question": payload["question"],
        "candidate_answer": payload["candidate_answer"],
        "evidence_mode": payload["evidence_mode"],
        "source_record_ids": ";".join(
            payload.get("source_record_ids", [])
        ),
        "medhalt_label": payload["evaluation"]["medhalt"]["label"],
        "pubmedqa_label": payload["evaluation"]["pubmedqa"]["label"],
        "routing_action": payload["evaluation"]["routing_action"],
    }

    write_header = not MANUAL_LOG_PATH.exists()

    with open(
        MANUAL_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)