from pathlib import Path
import html
import pandas as pd

OUT_DIR = Path("reports/dashboard")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "project_trustworthiness_dashboard_demo.html"

PROJECT_VIEW = Path("results/comparison/project_level_trustworthiness_view.csv")
WEEK3 = Path("results/comparison/week3_model_comparison_summary.csv")
EQUITY_ACTIONS = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_action_summary.csv")
EQUITY_SHORTLIST = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_shortlist.csv")

def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def esc(x):
    return "" if pd.isna(x) else html.escape(str(x))

def shorten(x, n=180):
    x = "" if pd.isna(x) else str(x).replace("\n", " ")
    return x if len(x) <= n else x[:n].rstrip() + "..."

def table(df, max_rows=8):
    if df.empty:
        return "<p class='muted'>No data available.</p>"
    df = df.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        body += "<tr>" + "".join(f"<td>{esc(row[c])}</td>" for c in df.columns) + "</tr>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

def metric(title, value, note):
    return f"<div class='metric'><span>{esc(title)}</span><strong>{esc(value)}</strong><small>{esc(note)}</small></div>"

project = read_csv(PROJECT_VIEW)
week3 = read_csv(WEEK3)
actions = read_csv(EQUITY_ACTIONS)
shortlist = read_csv(EQUITY_SHORTLIST)

demo_shortlist = shortlist.copy()
keep = [c for c in [
    "evaluation_type", "dataset_name", "fairness_category", "review_flags",
    "prompt_a", "model_response_a"
] if c in demo_shortlist.columns]
demo_shortlist = demo_shortlist[keep].head(6)
for col in demo_shortlist.columns:
    demo_shortlist[col] = demo_shortlist[col].apply(lambda x: shorten(x, 160))

bars = ""
if not actions.empty and "recommended_action" in actions.columns:
    count_col = "row_count" if "row_count" in actions.columns else "rows"
    total = max(actions[count_col].sum(), 1)
    for _, r in actions.iterrows():
        pct = round((r[count_col] / total) * 100, 1)
        bars += f"""
        <div class="bar-row">
          <span>{esc(r['recommended_action'])}</span>
          <div><i style="width:{pct}%"></i></div>
          <b>{esc(r[count_col])}</b>
        </div>
        """

html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Responsible AI Healthcare LLM Demo Dashboard</title>
<style>
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: #f7f8fb;
  color: #172033;
}}
header {{
  background: #0f172a;
  color: white;
  padding: 28px 40px;
}}
header h1 {{
  margin: 0 0 8px;
  font-size: 28px;
}}
header p {{
  margin: 0;
  color: #cbd5e1;
}}
main {{
  max-width: 1220px;
  margin: 0 auto;
  padding: 28px 32px 44px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 18px;
}}
.metric, section {{
  background: white;
  border: 1px solid #dfe4ec;
  border-radius: 8px;
}}
.metric {{
  padding: 16px;
}}
.metric span, .metric small {{
  display: block;
  color: #64748b;
}}
.metric strong {{
  display: block;
  font-size: 30px;
  margin: 8px 0;
}}
section {{
  padding: 20px;
  margin-bottom: 18px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 20px;
}}
.table-wrap {{
  overflow-x: auto;
  border: 1px solid #dfe4ec;
  border-radius: 8px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}}
th, td {{
  border-bottom: 1px solid #dfe4ec;
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
  font-size: 14px;
  overflow-wrap: anywhere;
}}
th {{
  background: #f1f5f9;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 220px 1fr 50px;
  gap: 12px;
  align-items: center;
  margin: 10px 0;
}}
.bar-row div {{
  background: #e5e7eb;
  height: 12px;
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
  header {{ background: white; color: #172033; padding-bottom: 12px; }}
  header p {{ color: #64748b; }}
  main {{ padding-top: 12px; }}
  section {{ break-inside: avoid; }}
  .metrics {{ grid-template-columns: repeat(4, 1fr); }}
}}
</style>
</head>
<body>
<header>
  <h1>Responsible AI Healthcare LLM Demo Dashboard</h1>
  <p>Hallucination detection, grounding, fairness review, and safety routing summary.</p>
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
    {table(project)}
  </section>

  <section>
    <h2>Week 3 Model Comparison</h2>
    {table(week3)}
  </section>

  <section>
    <h2>EquityMedQA Trust Routing</h2>
    <p class="muted">Rows routed by the trustworthiness layer.</p>
    {bars}
  </section>

  <section>
    <h2>Manual Review Examples</h2>
    <p class="muted">Compact examples only. Full prompts and responses remain in the CSV outputs.</p>
    {table(demo_shortlist, 6)}
  </section>
</main>
</body>
</html>
"""

OUT_HTML.write_text(html_doc, encoding="utf-8")
print("Demo dashboard generated successfully")
print("Saved:", OUT_HTML)
