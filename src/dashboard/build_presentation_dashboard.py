from pathlib import Path
import html
import pandas as pd

OUT_DIR = Path("reports/dashboard")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "project_trustworthiness_dashboard_presentation.html"

WEEK3 = Path("results/comparison/week3_model_comparison_summary.csv")
EQUITY_ACTIONS = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_action_summary.csv")
EQUITY_SHORTLIST = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_shortlist.csv")

def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def esc(x):
    return "" if pd.isna(x) else html.escape(str(x))

def table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
    return f"<div class='table-wrap'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"

def metric(title, value, note):
    return f"<div class='metric'><span>{esc(title)}</span><strong>{esc(value)}</strong><small>{esc(note)}</small></div>"

def bars(items):
    if not items:
        return "<p class='muted'>No data available.</p>"

    total = max(sum(v for _, v in items), 1)
    out = ""
    for label, value in items:
        pct = round((value / total) * 100, 1)
        out += f"""
        <div class="bar-row">
          <span>{esc(label)}</span>
          <div><i style="width:{pct}%"></i></div>
          <b>{value}</b>
        </div>
        """
    return out

week3 = read_csv(WEEK3)
actions = read_csv(EQUITY_ACTIONS)
shortlist = read_csv(EQUITY_SHORTLIST)

model_rows = []
if not week3.empty:
    for _, r in week3.iterrows():
        model_rows.append([
            r.get("model_or_method", ""),
            r.get("rows", ""),
            r.get("accuracy", ""),
            r.get("f1_score", ""),
            r.get("recall", ""),
            r.get("project_decision", "")
        ])

action_items = []
if not actions.empty and "recommended_action" in actions.columns:
    count_col = "row_count" if "row_count" in actions.columns else "rows"
    for _, r in actions.iterrows():
        action_items.append((r["recommended_action"], int(r[count_col])))

fairness_items = []
if not shortlist.empty and "fairness_category" in shortlist.columns:
    counts = shortlist["fairness_category"].value_counts()
    fairness_items = [(idx, int(val)) for idx, val in counts.items()]

flag_rows = []
if not shortlist.empty and "review_flags" in shortlist.columns:
    expanded = []
    for value in shortlist["review_flags"].dropna():
        for flag in str(value).split(";"):
            flag = flag.strip()
            if flag:
                expanded.append(flag)

    if expanded:
        counts = pd.Series(expanded).value_counts()
        flag_rows = [[idx, int(val)] for idx, val in counts.items()]

project_rows = [
    ["Hallucination Detection", "Qwen 1.5B reached 0.86 accuracy, 0.875 F1, and 0.98 recall on the Med-HALT balanced sample.", "Use Qwen 1.5B as the stronger local LLM candidate for continued validation."],
    ["Logic-Aware Routing", "Accepted hallucinated rows were reduced to 0 after adding routing for negative-question prompts.", "Keep this rule for prompts containing phrases such as NOT correct, incorrect, except, and least likely."],
    ["Fairness Review", "The EquityMedQA full run reviewed 60 responses and produced a 33-row manual review shortlist.", "Use EquityMedQA as the fairness and bias review layer."],
    ["Trustworthiness Layer", "EquityMedQA trust routing identified 18 mandatory review rows, 12 human review rows, and 3 accepted rows.", "Connect fairness review with the overall trustworthiness routing process."],
    ["Demo Readiness", "The framework now brings together hallucination detection, grounding, fairness review, and routing.", "Use this dashboard to present the current project-level progress."]
]

html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Responsible AI Healthcare LLM Dashboard</title>
<style>
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: #f7f8fb;
  color: #172033;
}}
header {{
  background: white;
  padding: 28px 42px 18px;
  border-bottom: 1px solid #dfe4ec;
}}
header h1 {{
  margin: 0 0 8px;
  font-size: 30px;
}}
header p {{
  margin: 0;
  color: #64748b;
  font-size: 16px;
}}
.overview-note {{
  margin-top: 14px;
  max-width: 1050px;
  color: #334155;
  font-size: 15px;
  line-height: 1.45;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 22px 28px 36px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}}
.metric, section {{
  background: white;
  border: 1px solid #dfe4ec;
  border-radius: 8px;
}}
.metric {{
  padding: 15px;
}}
.metric span, .metric small {{
  display: block;
  color: #64748b;
}}
.metric strong {{
  display: block;
  font-size: 30px;
  margin: 6px 0;
}}
section {{
  padding: 18px;
  margin-bottom: 16px;
}}
h2 {{
  margin: 0 0 8px;
  font-size: 20px;
}}
.section-note {{
  margin: 0 0 12px;
  color: #64748b;
  font-size: 14px;
  line-height: 1.4;
}}
.table-wrap {{
  border: 1px solid #dfe4ec;
  border-radius: 8px;
  overflow: hidden;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}}
th, td {{
  border-bottom: 1px solid #dfe4ec;
  padding: 9px 11px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
  line-height: 1.32;
  overflow-wrap: break-word;
}}
th {{
  background: #f1f5f9;
}}
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 220px 1fr 42px;
  gap: 10px;
  align-items: center;
  margin: 10px 0;
  font-size: 14px;
}}
.bar-row div {{
  background: #e5e7eb;
  height: 11px;
  border-radius: 999px;
  overflow: hidden;
}}
.bar-row i {{
  display: block;
  height: 100%;
  background: #1f6feb;
}}
.muted {{
  color: #64748b;
}}
@media print {{
  body {{ background: white; }}
  header {{ padding-top: 20px; }}
  main {{ padding-top: 16px; }}
  section {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<header>
  <h1>Responsible AI Healthcare LLM Dashboard</h1>
  <p>Project-level summary of hallucination detection, grounding, fairness review, and safety routing.</p>
  <div class="overview-note">
    This dashboard summarizes the current safety evaluation work for the healthcare LLM project. It shows how well the model detects incorrect medical answers, how risky responses are routed for review, and how fairness-related prompts are checked before the system is considered reliable.
  </div>
</header>

<main>
  <div class="metrics">
    {metric("Qwen 1.5B Accuracy", "0.86", "Med-HALT balanced sample")}
    {metric("Qwen 1.5B F1", "0.875", "Hallucination detection")}
    {metric("Qwen 1.5B Recall", "0.98", "Hallucination catch rate")}
    {metric("Accepted Hallucinations", "0", "After logic-aware routing")}
  </div>

  <section>
    <h2>Project-Level Summary</h2>
    <p class="section-note">This section summarizes each safety layer and the decision supported by the current results.</p>
    {table(["Layer", "Current Result", "Decision"], project_rows)}
  </section>

  <section>
    <h2>Week 3 Model Comparison</h2>
    <p class="section-note">This comparison shows why Qwen 1.5B is stronger than the earlier baseline and Qwen 0.5B runs.</p>
    {table(["Model / Method", "Rows", "Accuracy", "F1", "Recall", "Decision"], model_rows)}
  </section>

  <div class="two-col">
    <section>
      <h2>EquityMedQA Trust Routing</h2>
      <p class="section-note">This shows how many fairness-related responses were accepted, sent for review, or marked as mandatory human review.</p>
      {bars(action_items)}
    </section>

    <section>
      <h2>Fairness Categories Needing Review</h2>
      <p class="section-note">This groups the review shortlist by fairness category so the main risk areas are easier to identify.</p>
      {bars(fairness_items)}
    </section>
  </div>

  <section>
    <h2>Review Flags Needing Attention</h2>
    <p class="section-note">These flags show the main reasons responses need review, such as missing professional care guidance or possible unsafe clinical advice.</p>
    {table(["Review Flag", "Rows"], flag_rows)}
  </section>
</main>
</body>
</html>
"""

OUT_HTML.write_text(html_doc, encoding="utf-8")
print("Presentation dashboard generated successfully")
print("Saved:", OUT_HTML)
