"""FastAPI frontend for local evidence retrieval and manual answer evaluation."""

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

from src.demo.evidence_retrieval import EvidenceRetriever
from src.demo.live_evaluator import (
    DualAdapterEvaluator,
    append_manual_case_log,
)


app = FastAPI(
    title="Responsible AI Healthcare LLM Manual Evaluation",
    version="2.0.0",
)


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=8, max_length=2000)
    candidate_answer: str = Field(min_length=3, max_length=4000)


class EvaluationRequest(RetrievalRequest):
    evidence_override: str = Field(default="", max_length=7000)


@lru_cache(maxsize=1)
def get_retriever():
    return EvidenceRetriever()


@lru_cache(maxsize=1)
def get_evaluator():
    return DualAdapterEvaluator()


def clean_text(text):
    return " ".join(text.strip().split())


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api/health")
def health():
    return {
        "status": "ready",
        "models": [
            "Qwen 2.5 3B + Med-HALT LoRA",
            "Qwen 2.5 3B + PubMedQA LoRA",
        ],
        "evidence_corpus": (
            "Controlled PubMedQA training and validation abstracts only"
        ),
        "boundary": (
            "Research prototype; no automatic clinical approval."
        ),
    }


@app.post("/api/retrieve")
def retrieve(request: RetrievalRequest):
    question = clean_text(request.question)
    candidate_answer = clean_text(request.candidate_answer)

    try:
        sources = get_retriever().retrieve(
            question,
            candidate_answer,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return {
        "sources": sources,
        "source_label": "Local PubMedQA research corpus",
        "notice": (
            "These are locally retrieved research abstracts, not hospital "
            "guidelines or clinical validation."
        ),
    }


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    question = clean_text(request.question)
    candidate_answer = clean_text(request.candidate_answer)
    evidence_override = request.evidence_override.strip()

    if evidence_override:
        evidence_mode = (
            "User-supplied evidence "
            "(unverified for prototype testing)"
        )
        evidence = evidence_override
        sources = []
    else:
        try:
            sources = get_retriever().retrieve(
                question,
                candidate_answer,
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        if not sources:
            return {
                "sources": [],
                "evaluation": None,
                "routing_action": "Mandatory human review",
                "routing_reasons": [
                    "No local research evidence could be retrieved "
                    "for this case."
                ],
            }

        evidence_mode = (
            "Automatically retrieved local research evidence"
        )

        # The LoRA adapters were trained on one evidence passage per case.
        # Use the closest passage as evaluator input and show all top results.
        evidence = sources[0]["evidence"]

    try:
        evaluation = get_evaluator().evaluate(
            question,
            candidate_answer,
            evidence,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    response = {
        "sources": sources,
        "evidence_mode": evidence_mode,
        "evaluation": evaluation,
    }

    append_manual_case_log(
        {
            "question": question,
            "candidate_answer": candidate_answer,
            "evidence_mode": evidence_mode,
            "source_record_ids": [
                source["record_id"]
                for source in sources
            ],
            "evaluation": evaluation,
        }
    )

    return response


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Responsible AI Healthcare LLM Manual Evaluation</title>

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
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      background: var(--bg);
      line-height: 1.45;
    }

    header {
      background: var(--navy);
      color: white;
      padding: 27px 34px;
    }

    header h1 {
      margin: 0 0 6px;
      font-size: 27px;
    }

    header p {
      margin: 0;
      color: #cbd5e1;
      max-width: 950px;
    }

    main {
      max-width: 1280px;
      margin: auto;
      padding: 24px 34px 42px;
    }

    .notice {
      background: #fff7ed;
      border-left: 4px solid #ea580c;
      border-radius: 6px;
      padding: 13px 15px;
      margin-bottom: 18px;
      font-size: 14px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
      margin-bottom: 18px;
    }

    h2 {
      margin: 0 0 10px;
      font-size: 19px;
    }

    h3 {
      margin: 0 0 7px;
      font-size: 15px;
    }

    label {
      display: block;
      font-weight: 700;
      font-size: 14px;
      margin: 13px 0 6px;
    }

    textarea {
      width: 100%;
      min-height: 106px;
      resize: vertical;
      padding: 10px;
      border: 1px solid #b8c3d2;
      border-radius: 6px;
      font: 14px Arial, sans-serif;
    }

    details {
      margin-top: 14px;
    }

    summary {
      cursor: pointer;
      font-weight: 700;
      color: #334155;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }

    button {
      padding: 10px 15px;
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }

    button.secondary {
      background: #475569;
    }

    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .source {
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 8px;
      padding: 12px;
      margin-top: 10px;
    }

    .source-meta,
    .small {
      color: var(--muted);
      font-size: 13px;
    }

    .source p {
      margin: 8px 0 0;
      white-space: pre-wrap;
    }

    .cards {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .value {
      margin-top: 4px;
      font-size: 21px;
      font-weight: 700;
    }

    .score {
      margin-top: 8px;
      color: #475569;
      font-size: 13px;
    }

    .route {
      border-left: 5px solid var(--amber);
      background: #fffbeb;
    }

    .route.high {
      border-left-color: var(--red);
      background: #fef2f2;
    }

    .pill {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      color: white;
      background: var(--amber);
      font-size: 13px;
      font-weight: 700;
    }

    .pill.high {
      background: var(--red);
    }

    ul {
      padding-left: 20px;
      margin-bottom: 0;
    }

    .hidden {
      display: none;
    }

    footer {
      color: var(--muted);
      font-size: 13px;
      margin-top: 20px;
    }

    @media (max-width: 800px) {
      header,
      main {
        padding-left: 18px;
        padding-right: 18px;
      }

      .grid,
      .cards {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>

<body>
  <header>
    <h1>Responsible AI Healthcare LLM Manual Evaluation</h1>
    <p>
      Local evidence retrieval, task-specific hallucination assessment,
      and human-review routing for research-stage healthcare-AI answers.
    </p>
  </header>

  <main>
    <div class="notice">
      <strong>Research prototype:</strong>
      Do not enter patient-identifiable information. This tool does not
      diagnose, prescribe, or approve a clinical decision. Every result
      requires qualified human review.
    </div>

    <section class="panel">
      <h2>1. Question and AI Draft Answer</h2>

      <label for="question">Question</label>
      <textarea
        id="question"
        placeholder="Example: Does the supplied research evidence support the proposed intervention for this condition?"
      ></textarea>

      <label for="answer">AI-generated candidate answer</label>
      <textarea
        id="answer"
        placeholder="Paste the draft answer that should be checked against evidence."
      ></textarea>

      <details>
        <summary>
          Optional controlled testing: paste evidence manually
        </summary>

        <p class="small">
          Use a research abstract or manager-provided source excerpt.
          AI-generated text is accepted only for informal testing and
          is marked unverified.
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

      <p id="status" class="small"></p>
    </section>

    <div id="results" class="hidden">
      <div class="grid">
        <section class="panel">
          <h2>2. Evidence Used</h2>
          <p id="evidenceMode" class="small"></p>
          <div id="sources"></div>
        </section>

        <section>
          <div class="panel">
            <h2>3. Task-Specific Model Signals</h2>

            <div class="cards">
              <div class="card">
                <div class="label">
                  Med-HALT hallucination check
                </div>

                <div id="medhaltLabel" class="value"></div>

                <div id="medhaltScores" class="score"></div>
              </div>

              <div class="card">
                <div class="label">
                  PubMedQA evidence decision
                </div>

                <div id="pubmedLabel" class="value"></div>

                <div id="pubmedScores" class="score"></div>
              </div>
            </div>

            <p class="small">
              Scores are comparative label log-scores, not calibrated
              probabilities. The two signals are shown separately and
              are not combined into a clinical-performance score.
            </p>
          </div>

          <div id="routePanel" class="panel route">
            <h2>4. Human-Review Routing</h2>

            <span id="routeAction" class="pill"></span>

            <ul id="routeReasons"></ul>
          </div>
        </section>
      </div>
    </div>

    <footer>
      Automatic retrieval uses only the local controlled PubMedQA
      training and validation corpus. Frozen benchmark test sets are
      excluded from this demo retrieval index.
    </footer>
  </main>

  <script>
    function getValue(id) {
      return document.getElementById(id).value.trim();
    }

    function setStatus(message) {
      document.getElementById("status").textContent = message;
    }

    function setBusy(isBusy, message) {
      document.getElementById("retrieveButton").disabled = isBusy;
      document.getElementById("evaluateButton").disabled = isBusy;

      if (message) {
        setStatus(message);
      }
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function requireInputs() {
      const question = getValue("question");
      const answer = getValue("answer");

      if (question.length < 8 || answer.length < 3) {
        setStatus(
          "Enter a specific question and a candidate answer first."
        );

        return false;
      }

      return true;
    }

    function sourceHtml(source) {
      return `
        <div class="source">
          <div class="source-meta">
            ${escapeHtml(source.source_label)}
            · record ${escapeHtml(source.record_id)}
            · ${escapeHtml(source.split)} split
            · similarity ${escapeHtml(source.similarity_score)}
          </div>

          <h3>Related question</h3>
          <div>${escapeHtml(source.research_question)}</div>

          <h3 style="margin-top: 10px">Retrieved abstract</h3>
          <p>${escapeHtml(source.evidence)}</p>
        </div>
      `;
    }

    function showSources(sources, mode) {
      document
        .getElementById("results")
        .classList
        .remove("hidden");

      document.getElementById("evidenceMode").textContent = mode;

      const sourceContainer = document.getElementById("sources");

      if (sources.length) {
        sourceContainer.innerHTML = sources
          .map(sourceHtml)
          .join("");
      } else {
        sourceContainer.innerHTML = `
          <p class="small">
            No automatic sources were retrieved. A human reviewer must
            verify any manually supplied evidence.
          </p>
        `;
      }
    }

    async function retrieveEvidence() {
      if (!requireInputs()) {
        return;
      }

      setBusy(
        true,
        "Searching the local PubMedQA research corpus..."
      );

      try {
        const response = await fetch("/api/retrieve", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: getValue("question"),
            candidate_answer: getValue("answer"),
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Retrieval failed.");
        }

        showSources(
          data.sources,
          `${data.source_label}. ${data.notice}`
        );

        setStatus(
          `Retrieved ${data.sources.length} local evidence passages.`
        );
      } catch (error) {
        setStatus(error.message);
      } finally {
        setBusy(false);
      }
    }

    async function evaluateCase() {
      if (!requireInputs()) {
        return;
      }

      setBusy(
        true,
        "Loading local models if needed and evaluating the answer..."
      );

      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: getValue("question"),
            candidate_answer: getValue("answer"),
            evidence_override: getValue("override"),
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Evaluation failed.");
        }

        showSources(
          data.sources || [],
          data.evidence_mode || "No retrieved evidence"
        );

        const evaluation = data.evaluation;

        if (evaluation) {
          document.getElementById("medhaltLabel").textContent =
            evaluation.medhalt.label;

          document.getElementById("pubmedLabel").textContent =
            evaluation.pubmedqa.label;

          document.getElementById("medhaltScores").textContent =
            `Supported: ${evaluation.medhalt.scores.supported} | ` +
            `Hallucinated: ${evaluation.medhalt.scores.hallucinated} | ` +
            `Margin: ${evaluation.medhalt.confidence_margin}`;

          document.getElementById("pubmedScores").textContent =
            `Yes: ${evaluation.pubmedqa.scores.yes} | ` +
            `No: ${evaluation.pubmedqa.scores.no} | ` +
            `Maybe: ${evaluation.pubmedqa.scores.maybe} | ` +
            `Margin: ${evaluation.pubmedqa.confidence_margin}`;
        }

        const routingAction = evaluation
          ? evaluation.routing_action
          : data.routing_action;

        const routingReasons = evaluation
          ? evaluation.routing_reasons
          : data.routing_reasons;

        document.getElementById("routeAction").textContent =
          routingAction;

        document.getElementById("routeReasons").innerHTML =
          routingReasons
            .map(reason => `<li>${escapeHtml(reason)}</li>`)
            .join("");

        const mandatory = routingAction.includes("Mandatory");

        document
          .getElementById("routePanel")
          .classList
          .toggle("high", mandatory);

        document
          .getElementById("routeAction")
          .classList
          .toggle("high", mandatory);

        setStatus(
          "Manual case evaluated. It was logged separately from all benchmark results."
        );
      } catch (error) {
        setStatus(error.message);
      } finally {
        setBusy(false);
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )