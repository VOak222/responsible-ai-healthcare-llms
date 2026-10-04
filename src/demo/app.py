"""FastAPI frontend for the local Responsible AI healthcare LLM research prototype."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.demo.evidence_retrieval import (
    EvidenceRetriever,
    MIN_RELEVANCE_SCORE,
)
from src.demo.live_evaluator import (
    DualAdapterEvaluator,
    append_manual_case_log,
)


BENCHMARK_DECISION_SIMILARITY = 0.90
STRONG_MODEL_MARGIN = 0.75


app = FastAPI(
    title="Responsible AI Healthcare LLM Manual Evaluation",
    version="2.1.0",
)


class RetrievalRequest(BaseModel):
    question: str = Field(
        min_length=8,
        max_length=2000,
    )


class EvaluationRequest(BaseModel):
    question: str = Field(
        min_length=8,
        max_length=2000,
    )
    candidate_answer: str = Field(
        min_length=3,
        max_length=4000,
    )
    evidence_override: str = Field(
        default="",
        max_length=7000,
    )


@lru_cache(maxsize=1)
def get_retriever():
    return EvidenceRetriever()


@lru_cache(maxsize=1)
def get_evaluator():
    return DualAdapterEvaluator()


def clean_text(text: str) -> str:
    return " ".join(
        text.strip().split()
    )


def manual_disposition(
    evaluation: dict,
) -> dict:
    """
    Translate evaluator routing into the
    manual-evidence review vocabulary.
    """

    action = evaluation[
        "routing_action"
    ]

    if action == "Lower-priority human review":
        label = (
            "Routine human review — "
            "no escalation"
        )
        kind = "routine"

    elif action == "Mandatory human review":
        label = action
        kind = "mandatory"

    else:
        label = "Human review required"
        kind = "review"

    return {
        "label": label,
        "kind": kind,
        "reasons": evaluation[
            "routing_reasons"
        ],
        "note": (
            "This route uses manually supplied evidence. "
            "The evidence remains unverified for prototype "
            "testing, and the result is not clinical approval."
        ),
    }


def automatic_disposition(
    evaluation: dict,
    sources: list[dict],
) -> dict:
    """
    Create a controlled-corpus disposition for
    automatically retrieved evidence.
    """

    best_similarity = float(
        sources[0][
            "similarity_score"
        ]
    )

    route = evaluation[
        "routing_action"
    ]

    medhalt = evaluation[
        "medhalt"
    ]

    pubmedqa = evaluation[
        "pubmedqa"
    ]

    consistency = evaluation[
        "consistency"
    ]

    candidate = evaluation[
        "candidate_conclusion"
    ]

    # ------------------------------------------------------
    # Evidence between 0.70 and 0.89 is relevant enough to
    # display, but not strong enough for benchmark pass/fail.
    # ------------------------------------------------------

    if (
        best_similarity
        < BENCHMARK_DECISION_SIMILARITY
    ):

        if (
            route
            == "Mandatory human review"
        ):
            return {
                "label": (
                    "Mandatory human review"
                ),
                "kind": "mandatory",
                "reasons": (
                    evaluation[
                        "routing_reasons"
                    ]
                    + [
                        (
                            "The best evidence match "
                            f"({best_similarity:.3f}) "
                            "is below the "
                            f"{BENCHMARK_DECISION_SIMILARITY:.2f} "
                            "controlled-corpus decision threshold."
                        )
                    ]
                ),
                "note": (
                    "The evidence was relevant enough "
                    "to display, but not strong enough "
                    "for an automatic benchmark disposition."
                ),
            }

        return {
            "label": (
                "Human review required"
            ),
            "kind": "review",
            "reasons": [
                (
                    "The best evidence match "
                    f"({best_similarity:.3f}) "
                    "is below the "
                    f"{BENCHMARK_DECISION_SIMILARITY:.2f} "
                    "controlled-corpus decision threshold."
                ),
                (
                    "The case is not assigned a "
                    "benchmark pass/fail because "
                    "evidence similarity is only moderate."
                ),
            ],
            "note": (
                "This is a retrieval-quality safeguard, "
                "not a statement that the candidate answer "
                "is medically incorrect."
            ),
        }

    # ------------------------------------------------------
    # High-quality evidence match: decide whether a
    # conflicting signal is strong enough for benchmark fail.
    # ------------------------------------------------------

    medhalt_margin = float(
        medhalt[
            "confidence_margin"
        ]
    )

    # An explicit candidate/evidence contradiction is enough
    # for a controlled-corpus benchmark fail.
    if (
        consistency[
            "label"
        ]
        == "contradiction"
    ):
        return {
            "label": (
                "Benchmark fail — "
                "evidence conflict"
            ),
            "kind": "fail",
            "reasons": [
                (
                    "The candidate conclusion contradicts "
                    "the PubMedQA evidence conclusion."
                )
            ],
            "note": (
                "This fail applies only to the controlled "
                "local PubMedQA evidence workflow. "
                "It is not a clinical rejection or diagnosis."
            ),
        }

    # A Med-HALT hallucination signal becomes benchmark fail
    # only when the model separation is also strong.
    if (
        medhalt[
            "label"
        ]
        == "hallucinated"
        and medhalt_margin
        >= STRONG_MODEL_MARGIN
    ):
        return {
            "label": (
                "Benchmark fail — "
                "evidence conflict"
            ),
            "kind": "fail",
            "reasons": [
                (
                    "The Med-HALT adapter strongly flagged "
                    "the candidate answer as hallucinated."
                )
            ],
            "note": (
                "This fail applies only to the controlled "
                "local PubMedQA evidence workflow. "
                "It is not a clinical rejection or diagnosis."
            ),
        }

    # A low/moderate Med-HALT hallucination signal means the
    # task-specific signals disagree. It requires review rather
    # than being treated as an automatic benchmark failure.
    if (
        medhalt[
            "label"
        ]
        == "hallucinated"
    ):
        return {
            "label": (
                "Human review required"
            ),
            "kind": "review",
            "reasons": [
                (
                    "PubMedQA and the explicit candidate "
                    "conclusion are consistent, but Med-HALT "
                    "produced a hallucination signal with only "
                    "low or moderate label separation."
                )
            ],
            "note": (
                "The conflicting Med-HALT signal was not "
                "strong enough for a controlled benchmark fail, "
                "so the case is conservatively routed for review."
            ),
        }

    # ------------------------------------------------------
    # High-quality evidence and all major signals align.
    # ------------------------------------------------------

    strong_signals = (
        route
        == "Lower-priority human review"

        and medhalt[
            "label"
        ]
        == "supported"

        and pubmedqa[
            "label"
        ]
        in {
            "yes",
            "no",
        }

        and candidate[
            "label"
        ]
        in {
            "yes",
            "no",
        }

        and consistency[
            "label"
        ]
        == "consistent"

        and float(
            medhalt[
                "confidence_margin"
            ]
        )
        >= STRONG_MODEL_MARGIN

        and float(
            pubmedqa[
                "confidence_margin"
            ]
        )
        >= STRONG_MODEL_MARGIN
    )

    if strong_signals:
        return {
            "label": (
                "Benchmark pass — "
                "evidence aligned"
            ),
            "kind": "pass",
            "reasons": [
                (
                    "The automatically retrieved evidence "
                    "has a very high question match "
                    f"({best_similarity:.3f})."
                ),
                (
                    "Med-HALT marked the candidate as "
                    "supported, and the candidate conclusion "
                    "is consistent with the PubMedQA "
                    "evidence conclusion."
                ),
                (
                    "Both task-specific models have "
                    "strong label separation for this case."
                ),
            ],
            "note": (
                "This pass applies only to the controlled "
                "local PubMedQA evidence workflow. "
                "It is not clinical approval or a probability "
                "of medical correctness."
            ),
        }

    # ------------------------------------------------------
    # Strong retrieval but some other uncertainty remains.
    # ------------------------------------------------------

    return {
        "label": (
            "Human review required"
        ),
        "kind": "review",
        "reasons": evaluation[
            "routing_reasons"
        ],
        "note": (
            "The evidence match is high, but one or more "
            "task-specific signals did not satisfy the "
            "conservative benchmark-pass rules."
        ),
    }


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return HTML_PAGE


@app.get(
    "/api/health"
)
def health():
    return {
        "status": "ready",

        "models": [
            (
                "Qwen 2.5 3B "
                "+ Med-HALT LoRA"
            ),
            (
                "Qwen 2.5 3B "
                "+ PubMedQA LoRA"
            ),
        ],

        "evidence_corpus": (
            "Controlled PubMedQA training "
            "and validation abstracts only"
        ),

        "retrieval_policy": (
            "Question-only semantic retrieval "
            "with a relevance guard"
        ),

        "minimum_relevance_score": (
            MIN_RELEVANCE_SCORE
        ),

        "benchmark_decision_similarity": (
            BENCHMARK_DECISION_SIMILARITY
        ),

        "boundary": (
            "Research prototype; benchmark "
            "dispositions are not clinical "
            "approval or rejection."
        ),
    }


@app.post(
    "/api/retrieve"
)
def retrieve(
    request: RetrievalRequest,
):

    question = clean_text(
        request.question
    )

    try:
        sources = (
            get_retriever()
            .retrieve(
                question
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error

    if sources:
        notice = (
            "Evidence was retrieved using the "
            "research question only. "
            "The candidate answer did not "
            "influence retrieval."
        )

        status = (
            "matched"
        )

    else:
        notice = (
            "No sufficiently relevant evidence "
            "was found in the local research corpus. "
            "Unrelated nearest-neighbor results "
            "were rejected."
        )

        status = (
            "no_relevant_evidence"
        )

    return {
        "sources": sources,

        "source_label": (
            "Local PubMedQA "
            "research corpus"
        ),

        "retrieval_status": (
            status
        ),

        "notice": notice,
    }


@app.post(
    "/api/evaluate"
)
def evaluate(
    request: EvaluationRequest,
):

    question = clean_text(
        request.question
    )

    candidate_answer = clean_text(
        request.candidate_answer
    )

    evidence_override = (
        request
        .evidence_override
        .strip()
    )

    # ======================================================
    # MANUAL EVIDENCE MODE
    # ======================================================

    if evidence_override:

        evaluation_mode = (
            "manual"
        )

        evidence_mode = (
            "User-supplied evidence "
            "(unverified for prototype testing)"
        )

        evidence = (
            evidence_override
        )

        sources = []

    # ======================================================
    # AUTOMATIC RETRIEVAL MODE
    # ======================================================

    else:

        evaluation_mode = (
            "automatic"
        )

        try:
            sources = (
                get_retriever()
                .retrieve(
                    question
                )
            )

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=str(
                    error
                ),
            ) from error

        # No evidence survives the 0.70 relevance guard.
        if not sources:

            return {
                "sources": [],

                "evidence_mode": (
                    "No sufficiently relevant local "
                    "research evidence was found."
                ),

                "evaluation_mode": (
                    evaluation_mode
                ),

                "evaluation": None,

                "disposition": {
                    "label": (
                        "Insufficient local evidence — "
                        "human review required"
                    ),

                    "kind": (
                        "review"
                    ),

                    "reasons": [
                        (
                            "No sufficiently relevant "
                            "local evidence was found "
                            "for this question."
                        ),
                        (
                            "Automatic model evaluation "
                            "was not run against "
                            "unrelated evidence."
                        ),
                        (
                            "Provide verified evidence "
                            "manually for controlled "
                            "testing or send the case "
                            "for qualified human review."
                        ),
                    ],

                    "note": (
                        "Insufficient local evidence "
                        "does not mean the candidate "
                        "answer is incorrect."
                    ),
                },
            }

        evidence_mode = (
            "Automatically retrieved local "
            "research evidence using the "
            "question only"
        )

        evidence = (
            sources[0][
                "evidence"
            ]
        )

    # ======================================================
    # TASK-SPECIFIC MODEL EVALUATION
    # ======================================================

    try:
        evaluation = (
            get_evaluator()
            .evaluate(
                question,
                candidate_answer,
                evidence,
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error

    # ======================================================
    # FINAL DISPOSITION
    # ======================================================

    if (
        evaluation_mode
        == "manual"
    ):

        disposition = (
            manual_disposition(
                evaluation
            )
        )

    else:

        disposition = (
            automatic_disposition(
                evaluation,
                sources,
            )
        )

    response = {
        "sources": sources,
        "evidence_mode": (
            evidence_mode
        ),
        "evaluation_mode": (
            evaluation_mode
        ),
        "evaluation": (
            evaluation
        ),
        "disposition": (
            disposition
        ),
    }

    append_manual_case_log(
        {
            "question": (
                question
            ),

            "candidate_answer": (
                candidate_answer
            ),

            "evidence_mode": (
                evidence_mode
            ),

            "source_record_ids": [
                source[
                    "record_id"
                ]
                for source in sources
            ],

            "evaluation": (
                evaluation
            ),
        }
    )

    return response


HTML_PAGE = r"""
<!doctype html>
<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
Responsible AI Healthcare LLM Evaluation
</title>

<style>

:root {
    --navy: #0f172a;
    --blue: #2563eb;
    --green: #15803d;
    --amber: #a16207;
    --red: #b91c1c;
    --bg: #f7f8fb;
    --panel: #ffffff;
    --line: #dbe2ea;
    --muted: #64748b;
}

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    color: #172033;

    background:
        var(--bg);

    line-height:
        1.45;
}

header {

    background:
        var(--navy);

    color:
        white;

    padding:
        27px 34px;
}

header h1 {

    margin:
        0 0 6px;

    font-size:
        27px;
}

header p {

    margin:
        0;

    color:
        #cbd5e1;

    max-width:
        950px;
}

main {

    max-width:
        1280px;

    margin:
        auto;

    padding:
        24px 34px 42px;
}

.notice {

    background:
        #fff7ed;

    border-left:
        4px solid #ea580c;

    border-radius:
        6px;

    padding:
        13px 15px;

    margin-bottom:
        18px;

    font-size:
        14px;
}

.panel {

    background:
        var(--panel);

    border:
        1px solid var(--line);

    border-radius:
        10px;

    padding:
        18px;

    margin-bottom:
        18px;
}

h2 {

    margin:
        0 0 10px;

    font-size:
        19px;
}

h3 {

    margin:
        0 0 7px;

    font-size:
        15px;
}

label {

    display:
        block;

    font-weight:
        700;

    font-size:
        14px;

    margin:
        13px 0 6px;
}

textarea {

    width:
        100%;

    min-height:
        106px;

    resize:
        vertical;

    padding:
        10px;

    border:
        1px solid #b8c3d2;

    border-radius:
        6px;

    font:
        14px Arial,
        sans-serif;
}

details {
    margin-top:
        14px;
}

summary {

    cursor:
        pointer;

    font-weight:
        700;

    color:
        #334155;
}

.actions {

    display:
        flex;

    gap:
        10px;

    flex-wrap:
        wrap;

    margin-top:
        16px;
}

button {

    padding:
        10px 15px;

    border:
        0;

    border-radius:
        6px;

    background:
        var(--blue);

    color:
        white;

    font-weight:
        700;

    cursor:
        pointer;
}

button.secondary {
    background:
        #475569;
}

button:disabled {

    opacity:
        0.6;

    cursor:
        wait;
}

.grid {

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        18px;
}

.source {

    border:
        1px solid var(--line);

    background:
        #f8fafc;

    border-radius:
        8px;

    padding:
        12px;

    margin-top:
        10px;
}

.source-meta,
.small {

    color:
        var(--muted);

    font-size:
        13px;
}

.source p {

    margin:
        8px 0 0;

    white-space:
        pre-wrap;
}

.no-evidence {

    border:
        1px dashed #d97706;

    background:
        #fffbeb;

    border-radius:
        8px;

    padding:
        13px;

    color:
        #78350f;

    margin-top:
        10px;
}

.cards {

    display:
        grid;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    gap:
        12px;
}

.card {

    border:
        1px solid var(--line);

    border-radius:
        8px;

    padding:
        13px;
}

.label {

    color:
        var(--muted);

    font-size:
        12px;

    font-weight:
        700;

    letter-spacing:
        0.04em;

    text-transform:
        uppercase;
}

.value {

    margin-top:
        4px;

    font-size:
        21px;

    font-weight:
        700;
}

.score {

    margin-top:
        8px;

    color:
        #475569;

    font-size:
        13px;
}

.preference-title {

    margin-top:
        10px;

    margin-bottom:
        6px;

    color:
        #475569;

    font-size:
        12px;

    font-weight:
        700;
}

.preference-row {

    display:
        grid;

    grid-template-columns:
        minmax(
            78px,
            auto
        )
        1fr
        auto;

    gap:
        8px;

    align-items:
        center;

    margin-top:
        6px;

    color:
        #475569;

    font-size:
        12px;
}

.preference-track {

    height:
        8px;

    background:
        #e2e8f0;

    border-radius:
        999px;

    overflow:
        hidden;
}

.preference-fill {

    height:
        100%;

    background:
        var(--blue);

    border-radius:
        999px;
}

.separation {

    margin-top:
        10px;

    color:
        #334155;

    font-size:
        12px;

    font-weight:
        700;
}


/* =========================================================
   FINAL DISPOSITION COLORS
   ========================================================= */

.route {

    border-left:
        5px solid var(--amber);

    background:
        #fffbeb;
}


/* Controlled benchmark pass */

.route.pass {

    border-left-color:
        var(--green);

    background:
        #f0fdf4;
}


/* Controlled benchmark fail */

.route.fail,
.route.mandatory {

    border-left-color:
        var(--red);

    background:
        #fef2f2;
}


/* Manual review / uncertainty */

.route.review,
.route.routine {

    border-left-color:
        var(--amber);

    background:
        #fffbeb;
}


.pill {

    display:
        inline-block;

    padding:
        4px 9px;

    border-radius:
        999px;

    color:
        white;

    background:
        var(--amber);

    font-size:
        13px;

    font-weight:
        700;
}

.pill.pass {
    background:
        var(--green);
}

.pill.fail,
.pill.mandatory {
    background:
        var(--red);
}

.pill.review,
.pill.routine {
    background:
        var(--amber);
}


ul {

    padding-left:
        20px;

    margin-bottom:
        0;
}


.hidden {
    display:
        none;
}


footer {

    color:
        var(--muted);

    font-size:
        13px;

    margin-top:
        20px;
}


@media (
    max-width:
        800px
) {

    header,
    main {

        padding-left:
            18px;

        padding-right:
            18px;
    }

    .grid,
    .cards {

        grid-template-columns:
            1fr;
    }
}

</style>

</head>


<body>


<header>

<h1>
Responsible AI Healthcare LLM Evaluation
</h1>

<p>
Local evidence retrieval, task-specific hallucination
assessment, and review-oriented routing for
research-stage healthcare-AI answers.
</p>

</header>


<main>


<div class="notice">

<strong>
Research prototype:
</strong>

Do not enter patient-identifiable information.

Benchmark pass/fail applies only to the controlled
local evidence workflow.

It does not diagnose, prescribe, or grant
clinical approval.

</div>


<section class="panel">


<h2>
1. Question and AI Draft Answer
</h2>


<label for="question">
Question
</label>


<textarea
    id="question"
    placeholder="Enter the healthcare research question."
></textarea>


<label for="answer">
AI-generated candidate answer
</label>


<textarea
    id="answer"
    placeholder="Paste the draft answer that should be checked against evidence."
></textarea>


<details>

<summary>
Optional controlled testing:
paste evidence manually
</summary>


<p class="small">

Use a research abstract or
manager-provided source excerpt.

AI-generated text is accepted only
for informal testing and is marked
unverified.

</p>


<textarea
    id="override"
    placeholder="Optional evidence override. Leave blank to retrieve local PubMedQA research abstracts automatically."
></textarea>


</details>


<div class="actions">


<button
    class="secondary"
    id="retrieveButton"
    onclick="retrieveEvidence()"
>
Retrieve evidence
</button>


<button
    id="evaluateButton"
    onclick="evaluateCase()"
>
Evaluate answer
</button>


</div>


<p
    id="status"
    class="small"
></p>


</section>


<div
    id="results"
    class="hidden"
>


<div class="grid">


<section class="panel">


<h2>
2. Evidence Used
</h2>


<p
    id="evidenceMode"
    class="small"
></p>


<div id="sources"></div>


</section>


<section>


<div
    id="evaluationPanel"
    class="panel hidden"
>


<h2>
3. Task-Specific Model Signals
</h2>


<div class="cards">


<div class="card">

<div class="label">
Med-HALT hallucination check
</div>

<div
    id="medhaltLabel"
    class="value"
></div>

<div
    id="medhaltScores"
    class="score"
></div>

</div>


<div class="card">

<div class="label">
PubMedQA evidence conclusion
</div>

<div
    id="pubmedLabel"
    class="value"
></div>

<div
    id="pubmedScores"
    class="score"
></div>

<div class="small">

Answers the original research question
from the supplied evidence.

It does not directly judge the
candidate answer.

</div>

</div>


<div class="card">

<div class="label">
Candidate-evidence consistency
</div>

<div
    id="consistencyLabel"
    class="value"
></div>

<div
    id="consistencyDetails"
    class="score"
></div>

</div>


</div>


<p class="small">

Relative model preference shows how strongly
each model ranks its available labels for this case.

These percentages are not probabilities
of medical correctness.

Med-HALT and PubMedQA remain separate
task-specific signals, and consistency is
a rule-based comparison.

</p>


</div>


<div
    id="routePanel"
    class="panel route hidden"
>


<h2>
4. Final Disposition / Review Routing
</h2>


<span
    id="routeAction"
    class="pill"
></span>


<ul
    id="routeReasons"
></ul>


<p
    id="routeNote"
    class="small"
></p>


</div>


</section>


</div>

</div>


<footer>

Automatic retrieval uses only the local controlled
PubMedQA training and validation corpus.

Retrieval is question-only, applies a 0.70
relevance guard, and uses 0.90 as the conservative
benchmark-disposition similarity threshold.

Frozen benchmark test sets are excluded from
the retrieval index.

</footer>


</main>


<script>


function getValue(
    id
) {

    return document
        .getElementById(
            id
        )
        .value
        .trim();
}


function setStatus(
    message
) {

    document
        .getElementById(
            "status"
        )
        .textContent =
            message;
}


function setBusy(
    isBusy,
    message
) {

    document
        .getElementById(
            "retrieveButton"
        )
        .disabled =
            isBusy;

    document
        .getElementById(
            "evaluateButton"
        )
        .disabled =
            isBusy;

    if (
        message
    ) {
        setStatus(
            message
        );
    }
}


function escapeHtml(
    text
) {

    return String(
        text
    )

    .replaceAll(
        "&",
        "&amp;"
    )

    .replaceAll(
        "<",
        "&lt;"
    )

    .replaceAll(
        ">",
        "&gt;"
    )

    .replaceAll(
        '"',
        "&quot;"
    )

    .replaceAll(
        "'",
        "&#039;"
    );
}


function requireQuestion() {

    if (
        getValue(
            "question"
        ).length
        < 8
    ) {

        setStatus(
            "Enter a specific question first."
        );

        return false;
    }

    return true;
}


function requireInputs() {

    if (

        getValue(
            "question"
        ).length
        < 8

        ||

        getValue(
            "answer"
        ).length
        < 3

    ) {

        setStatus(
            "Enter a specific question and a candidate answer first."
        );

        return false;
    }

    return true;
}


function normalizedPreferences(
    scores
) {

    const entries =
        Object.entries(
            scores
        );

    const maxScore =
        Math.max(
            ...entries.map(
                (
                    [
                        ,
                        score
                    ]
                ) =>
                    Number(
                        score
                    )
            )
        );

    const exponentials =
        entries.map(
            (
                [
                    label,
                    score
                ]
            ) => [

                label,

                Math.exp(
                    Number(
                        score
                    )
                    -
                    maxScore
                )

            ]
        );

    const total =
        exponentials.reduce(
            (
                sum,
                [
                    ,
                    value
                ]
            ) =>
                sum
                +
                value,
            0
        );

    return Object.fromEntries(

        exponentials.map(
            (
                [
                    label,
                    value
                ]
            ) => [

                label,

                value
                /
                total

            ]
        )

    );
}


function formatPreference(
    value
) {

    const percentage =
        value
        *
        100;

    if (

        percentage
        > 0

        &&

        percentage
        < 0.1

    ) {

        return "<0.1%";
    }

    return (
        `${percentage.toFixed(1)}%`
    );
}


function separationLabel(
    margin
) {

    const value =
        Number(
            margin
        );

    if (
        value
        < 0.15
    ) {

        return "Low";
    }

    if (
        value
        < 0.75
    ) {

        return "Moderate";
    }

    return "Strong";
}


function displayLabel(
    label
) {

    return String(
        label
    )

    .replaceAll(
        "_",
        " "
    )

    .replace(
        /\b\w/g,
        character =>
            character
            .toUpperCase()
    );
}


function preferenceHtml(
    scores,
    margin
) {

    const entries =
        Object.entries(
            normalizedPreferences(
                scores
            )
        )

        .sort(
            (
                a,
                b
            ) =>
                b[1]
                -
                a[1]
        );

    const rows =
        entries.map(
            (
                [
                    label,
                    value
                ]
            ) => {

                const width =
                    Math.max(
                        value
                        *
                        100,
                        0.5
                    );

                return `
                    <div class="preference-row">

                        <span>
                            ${escapeHtml(
                                displayLabel(
                                    label
                                )
                            )}
                        </span>

                        <div class="preference-track">

                            <div
                                class="preference-fill"
                                style="width:${width.toFixed(1)}%"
                            ></div>

                        </div>

                        <strong>

                            ${escapeHtml(
                                formatPreference(
                                    value
                                )
                            )}

                        </strong>

                    </div>
                `;
            }
        )

        .join(
            ""
        );

    return `

        <div class="preference-title">
            Relative model preference
        </div>

        ${rows}

        <div class="separation">

            Model separation:
            ${escapeHtml(
                separationLabel(
                    margin
                )
            )}

        </div>

    `;
}


function sourceHtml(
    source
) {

    return `

        <div class="source">

            <div class="source-meta">

                ${escapeHtml(
                    source.source_label
                )}

                · record

                ${escapeHtml(
                    source.record_id
                )}

                ·

                ${escapeHtml(
                    source.split
                )}

                split

                · similarity

                ${escapeHtml(
                    source.similarity_score
                )}

            </div>


            <h3>
                Related question
            </h3>


            <div>

                ${escapeHtml(
                    source.research_question
                )}

            </div>


            <h3
                style="margin-top:10px"
            >

                Retrieved abstract

            </h3>


            <p>

                ${escapeHtml(
                    source.evidence
                )}

            </p>

        </div>

    `;
}


function showSources(
    sources,
    mode
) {

    document
        .getElementById(
            "results"
        )
        .classList
        .remove(
            "hidden"
        );

    document
        .getElementById(
            "evidenceMode"
        )
        .textContent =
            mode;

    const container =
        document
        .getElementById(
            "sources"
        );


    if (
        sources.length
    ) {

        container.innerHTML =

            sources
            .map(
                sourceHtml
            )
            .join(
                ""
            );

    }


    else if (

        String(
            mode
        )
        .startsWith(
            "User-supplied evidence"
        )

    ) {

        container.innerHTML = `

            <div class="source">

                <strong>

                    User-supplied evidence
                    used for this evaluation.

                </strong>


                <div
                    class="small"
                    style="margin-top:6px"
                >

                    This evidence was supplied
                    manually and is unverified
                    for prototype testing.

                    No automatic evidence retrieval
                    was used for this case.

                </div>

            </div>

        `;

    }


    else {

        container.innerHTML = `

            <div class="no-evidence">

                <strong>

                    No sufficiently relevant
                    local evidence found.

                </strong>


                <div
                    style="margin-top:6px"
                >

                    Weak nearest-neighbor matches
                    were rejected instead of being
                    used as unrelated evidence.

                    Controlled evidence may be
                    supplied manually.

                </div>

            </div>

        `;

    }
}


function hideDecisionPanels() {

    document
        .getElementById(
            "evaluationPanel"
        )
        .classList
        .add(
            "hidden"
        );

    document
        .getElementById(
            "routePanel"
        )
        .classList
        .add(
            "hidden"
        );
}


function showDisposition(
    disposition
) {

    const panel =
        document
        .getElementById(
            "routePanel"
        );

    const pill =
        document
        .getElementById(
            "routeAction"
        );


    const kinds = [

        "pass",
        "fail",
        "mandatory",
        "review",
        "routine",

    ];


    panel
        .classList
        .remove(
            "hidden"
        );


    for (
        const kind
        of kinds
    ) {

        panel
            .classList
            .remove(
                kind
            );

        pill
            .classList
            .remove(
                kind
            );
    }


    panel
        .classList
        .add(
            disposition.kind
        );


    pill
        .classList
        .add(
            disposition.kind
        );


    pill.textContent =
        disposition.label;


    document
        .getElementById(
            "routeReasons"
        )
        .innerHTML =

            disposition
            .reasons

            .map(
                reason =>
                    `<li>${escapeHtml(
                        reason
                    )}</li>`
            )

            .join(
                ""
            );


    document
        .getElementById(
            "routeNote"
        )
        .textContent =
            disposition.note
            ||
            "";
}


function showEvaluation(
    evaluation
) {

    document
        .getElementById(
            "evaluationPanel"
        )
        .classList
        .remove(
            "hidden"
        );


    document
        .getElementById(
            "medhaltLabel"
        )
        .textContent =
            evaluation
            .medhalt
            .label;


    document
        .getElementById(
            "pubmedLabel"
        )
        .textContent =
            evaluation
            .pubmedqa
            .label;


    document
        .getElementById(
            "medhaltScores"
        )
        .innerHTML =

            preferenceHtml(

                evaluation
                .medhalt
                .scores,

                evaluation
                .medhalt
                .confidence_margin

            );


    document
        .getElementById(
            "pubmedScores"
        )
        .innerHTML =

            preferenceHtml(

                evaluation
                .pubmedqa
                .scores,

                evaluation
                .pubmedqa
                .confidence_margin

            );


    document
        .getElementById(
            "consistencyLabel"
        )
        .textContent =
            evaluation
            .consistency
            .label;


    document
        .getElementById(
            "consistencyDetails"
        )
        .textContent =

            "Candidate conclusion: "

            +

            evaluation
            .candidate_conclusion
            .label

            +

            ". "

            +

            evaluation
            .consistency
            .explanation;
}


async function retrieveEvidence() {

    if (
        !requireQuestion()
    ) {

        return;
    }


    hideDecisionPanels();


    setBusy(

        true,

        "Searching the local PubMedQA research corpus using the question only..."

    );


    try {

        const response =
            await fetch(

                "/api/retrieve",

                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {
                                question:
                                    getValue(
                                        "question"
                                    )
                            }
                        )
                }

            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(

                data.detail
                ||
                "Retrieval failed."

            );
        }


        showSources(

            data.sources,

            `${data.source_label}. ${data.notice}`

        );


        setStatus(

            data.sources.length

            ?

            `Retrieved ${data.sources.length} sufficiently relevant local evidence passage(s).`

            :

            "No sufficiently relevant local evidence was found."

        );

    }


    catch (
        error
    ) {

        setStatus(
            error.message
        );
    }


    finally {

        setBusy(
            false
        );
    }
}


async function evaluateCase() {

    if (
        !requireInputs()
    ) {

        return;
    }


    hideDecisionPanels();


    setBusy(

        true,

        "Loading local models if needed and evaluating the answer..."

    );


    try {

        const response =
            await fetch(

                "/api/evaluate",

                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {

                                question:
                                    getValue(
                                        "question"
                                    ),

                                candidate_answer:
                                    getValue(
                                        "answer"
                                    ),

                                evidence_override:
                                    getValue(
                                        "override"
                                    )

                            }
                        )
                }

            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(

                data.detail
                ||
                "Evaluation failed."

            );
        }


        showSources(

            data.sources
            ||
            [],

            data.evidence_mode
            ||
            "No retrieved evidence"

        );


        if (
            data.evaluation
        ) {

            showEvaluation(
                data.evaluation
            );
        }


        showDisposition(
            data.disposition
        );


        setStatus(

            data.evaluation

            ?

            "Case evaluated. Manual demo cases remain separate from benchmark result files."

            :

            "Automatic evaluation was stopped because sufficiently relevant evidence was unavailable."

        );

    }


    catch (
        error
    ) {

        setStatus(
            error.message
        );
    }


    finally {

        setBusy(
            false
        );
    }
}


</script>


</body>

</html>
"""


if __name__ == "__main__":

    uvicorn.run(

        app,

        host=
            "127.0.0.1",

        port=
            8000,

    )