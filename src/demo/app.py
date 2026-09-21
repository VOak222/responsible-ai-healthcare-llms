import json
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

STRESS_TEST_PATH = (
    DATA_DIR / "unseen_medhalt_reasoning_fake_stress_test.jsonl"
)
SOURCE_PATH = DATA_DIR / "medhalt_reasoning_fake_common.csv"
PREDICTIONS_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_test_predictions.csv"
)
ERROR_CASES_PATH = (
    RESULT_DIR / "qwen2_5_3b_unseen_fake_stress_error_cases.csv"
)

LORA_MODEL_NAME = "Qwen2.5-3B + Med-HALT Hallucination LoRA"

app = FastAPI(
    title="Responsible AI Healthcare LLM Demo",
    version="1.0.0",
)


def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def as_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def label_text(label):
    return str(label).replace("_", " ").title()


def make_case(row, scenario, error_type=""):
    predicted_label = as_text(row["predicted_label"])
    true_label = as_text(row["true_label"])

    if error_type == "false_positive_supported_flagged":
        action = "Human review required"
        reason = (
            "The model conservatively flagged a safe-refusal response. "
            "A reviewer should verify that the refusal is appropriate."
        )
    elif predicted_label == "hallucinated":
        action = "Human review required"
        reason = (
            "The response was flagged as hallucinated and should not be "
            "treated as reliable clinical information."
        )
    else:
        action = "Lower-priority human review"
        reason = (
            "The answer was evidence-supported in this controlled benchmark, "
            "but the prototype does not provide clinical clearance."
        )

    return {
        "id": scenario.lower().replace(" ", "_"),
        "scenario": scenario,
        "benchmark": "Unseen Med-HALT Reasoning Fake stress test",
        "question": as_text(row["question"]),
        "evidence": as_text(row["knowledge"]),
        "candidate_answer": as_text(row["answer"]),
        "model_prediction": label_text(predicted_label),
        "reference_label": label_text(true_label),
        "score_supported": round(float(row["score_supported"]), 3),
        "score_hallucinated": round(float(row["score_hallucinated"]), 3),
        "confidence_margin": round(float(row["confidence_margin"]), 3),
        "error_type": error_type or "correct_prediction",
        "routing_action": action,
        "routing_reason": reason,
    }


def load_demo_cases():
    required_paths = [
        STRESS_TEST_PATH,
        SOURCE_PATH,
        PREDICTIONS_PATH,
        ERROR_CASES_PATH,
    ]

    for file_path in required_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    stress_test = pd.DataFrame(load_jsonl(STRESS_TEST_PATH))
    source_data = pd.read_csv(SOURCE_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    error_cases = pd.read_csv(ERROR_CASES_PATH)

    stress_test["id"] = stress_test["id"].astype(str)
    stress_test["source_record_id"] = (
        stress_test["source_record_id"].astype(str)
    )
    source_data["record_id"] = source_data["record_id"].astype(str)
    predictions["id"] = predictions["id"].astype(str)

    lora_predictions = predictions[
        predictions["model"] == LORA_MODEL_NAME
    ].copy()

    if lora_predictions.empty:
        raise RuntimeError("LoRA predictions were not found in the CSV.")

    metadata = stress_test[
        ["id", "source_record_id", "case_type", "question_group_id"]
    ]

    combined = lora_predictions.merge(
        metadata,
        on=["id", "case_type", "question_group_id"],
        how="left",
    ).merge(
        source_data[
            ["record_id", "question", "knowledge", "answer"]
        ],
        left_on="source_record_id",
        right_on="record_id",
        how="left",
    )

    if combined["question"].isna().any():
        raise RuntimeError("Could not connect all predictions to source text.")

    correct_hallucination = combined[
        (combined["true_label"] == "hallucinated")
        & (combined["predicted_label"] == "hallucinated")
    ].iloc[0]

    correct_supported = combined[
        (combined["true_label"] == "supported")
        & (combined["predicted_label"] == "supported")
    ].iloc[0]

    safe_refusal_error = error_cases[
        error_cases["error_type"]
        == "false_positive_supported_flagged"
    ].iloc[0]

    return {
        "correct_hallucination": make_case(
            correct_hallucination,
            "Hallucinated response detected",
        ),
        "correct_supported": make_case(
            correct_supported,
            "Evidence-supported response",
        ),
        "safe_refusal_false_positive": make_case(
            safe_refusal_error,
            "Safe-refusal false positive",
            "false_positive_supported_flagged",
        ),
    }


DEMO_CASES = load_demo_cases()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api/cases")
def get_cases():
    return [
        {
            "id": case_id,
            "scenario": case["scenario"],
        }
        for case_id, case in DEMO_CASES.items()
    ]


@app.get("/api/cases/{case_id}")
def get_case(case_id):
    if case_id not in DEMO_CASES:
        raise HTTPException(status_code=404, detail="Demo case not found.")

    return DEMO_CASES[case_id]


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Responsible AI Healthcare LLM Demo</title>
  <style>
    :root {
      --navy: #0f172a;
      --blue: #2563eb;
      --green: #15803d;
      --red: #b91c1c;
      --amber: #a16207;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #dbe2ea;
      --muted: #64748b;
    }

    * { box-sizing: border-box; }

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
      padding: 26px 34px;
    }

    header h1 {
      margin: 0 0 6px;
      font-size: 27px;
    }

    header p {
      margin: 0;
      color: #cbd5e1;
      max-width: 900px;
    }

    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 34px 42px;
    }

    .notice {
      background: #fff7ed;
      border-left: 4px solid #ea580c;
      padding: 13px 15px;
      border-radius: 6px;
      margin-bottom: 18px;
      font-size: 14px;
    }

    .controls, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
    }

    .controls {
      display: flex;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
    }

    .field {
      flex: 1;
    }

    label {
      display: block;
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 6px;
    }

    select {
      width: 100%;
      padding: 10px;
      border: 1px solid #b8c3d2;
      border-radius: 6px;
      font-size: 14px;
      background: white;
    }

    button {
      padding: 10px 16px;
      color: white;
      background: var(--blue);
      border: 0;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }

    .panel {
      margin-bottom: 18px;
    }

    h2 {
      margin: 0 0 10px;
      font-size: 19px;
    }

    h3 {
      margin: 0 0 8px;
      font-size: 15px;
    }

    .text-box {
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 11px;
      min-height: 62px;
      font-size: 14px;
    }

    .stack {
      display: grid;
      gap: 13px;
    }

    .result {
      border-left: 5px solid var(--blue);
      background: #eff6ff;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .value {
      font-size: 20px;
      font-weight: 700;
      margin-top: 3px;
    }

    .risk {
      display: inline-block;
      margin-top: 4px;
      padding: 4px 9px;
      border-radius: 999px;
      color: white;
      background: var(--amber);
      font-size: 13px;
      font-weight: 700;
    }

    .scores {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }

    .score-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
    }

    .small {
      color: var(--muted);
      font-size: 13px;
    }

    footer {
      color: var(--muted);
      font-size: 13px;
      margin-top: 20px;
    }

    @media (max-width: 800px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .controls, .grid { grid-template-columns: 1fr; display: grid; }
      .scores { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Responsible AI Healthcare LLM Demo</h1>
    <p>Evidence-grounded hallucination review and human-review routing using saved held-out evaluation outputs.</p>
  </header>

  <main>
    <div class="notice">
      <strong>Research prototype:</strong> This page presents reproducible benchmark cases and is not a diagnostic, treatment, or clinical-decision system.
    </div>

    <section class="controls">
      <div class="field">
        <label for="caseSelect">Select an evaluated case</label>
        <select id="caseSelect"></select>
      </div>
      <button onclick="loadSelectedCase()">Load case</button>
    </section>

    <div class="grid">
      <section>
        <div class="panel">
          <h2>Input Case</h2>
          <div class="stack">
            <div>
              <h3>Question</h3>
              <div id="question" class="text-box"></div>
            </div>
            <div>
              <h3>Evidence / Knowledge Context</h3>
              <div id="evidence" class="text-box"></div>
            </div>
            <div>
              <h3>Candidate Answer</h3>
              <div id="answer" class="text-box"></div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div class="panel result">
          <h2>Model Assessment</h2>
          <div class="label">Qwen 2.5 3B + selected LoRA</div>
          <div id="prediction" class="value"></div>

          <div class="scores">
            <div class="score-card">
              <div class="label">Supported score</div>
              <div id="supportedScore" class="value"></div>
            </div>
            <div class="score-card">
              <div class="label">Hallucinated score</div>
              <div id="hallucinatedScore" class="value"></div>
            </div>
          </div>

          <p class="small">
            Scores are comparative model values, not calibrated probabilities.
          </p>
        </div>

        <div class="panel">
          <h2>Human-Review Routing</h2>
          <div id="routingAction" class="risk"></div>
          <p id="routingReason"></p>
          <p class="small">
            Benchmark reference label: <strong id="referenceLabel"></strong><br>
            Case type: <strong id="scenario"></strong>
          </p>
        </div>

        <div class="panel">
          <h2>Governance Note</h2>
          <p class="small">
            The safe-refusal case demonstrates the final model’s known limitation:
            it can conservatively flag an appropriate refusal on an implausible
            prompt. This is why ambiguous cases are routed to a human reviewer.
          </p>
        </div>
      </section>
    </div>

    <footer>
      Source: frozen unseen Med-HALT Reasoning Fake stress-test outputs saved in this local project.
    </footer>
  </main>

  <script>
    async function loadSelectedCase() {
      const caseId = document.getElementById("caseSelect").value;
      const response = await fetch(`/api/cases/${caseId}`);
      const data = await response.json();

      document.getElementById("question").textContent = data.question;
      document.getElementById("evidence").textContent = data.evidence;
      document.getElementById("answer").textContent = data.candidate_answer;
      document.getElementById("prediction").textContent = data.model_prediction;
      document.getElementById("supportedScore").textContent = data.score_supported;
      document.getElementById("hallucinatedScore").textContent = data.score_hallucinated;
      document.getElementById("routingAction").textContent = data.routing_action;
      document.getElementById("routingReason").textContent = data.routing_reason;
      document.getElementById("referenceLabel").textContent = data.reference_label;
      document.getElementById("scenario").textContent = data.scenario;
    }

    async function start() {
      const response = await fetch("/api/cases");
      const cases = await response.json();
      const select = document.getElementById("caseSelect");

      cases.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.scenario;
        select.appendChild(option);
      });

      loadSelectedCase();
    }

    start();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)