from pathlib import Path
import html
import pandas as pd

OUT_DIR = Path("reports/dashboard")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "project_trustworthiness_dashboard_final.html"

PROJECT_VIEW = Path("results/comparison/project_level_trustworthiness_view.csv")
WEEK3 = Path("results/comparison/week3_model_comparison_summary.csv")
EQUITY_ACTIONS = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_action_summary.csv")
EQUITY_SHORTLIST = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_shortlist.csv")
EQUITY_REVIEW_SUMMARY = Path("results/equitymedqa_fairness/qwen_equitymedqa_fairness_full_review_summary.csv")

def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def esc(x):
    return "" if pd.isna(x) else html.escape(str(x))

def table(df, max_rows=10):
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

def bar_chart(df, label_col, count_col):
    if df.empty or label_col not in df.columns or count_col not in df.columns:
        return "<p class='muted'>No summary available.</p>"

    total = max(float(df[count_col].sum()), 1.0)
    rows = ""
    for _, r in df.iterrows():
        pct = round((float(r[count_col]) / total) * 100, 1)
        rows += f"""
        <div class="bar-row">
          <span>{esc(r[label_col])}</span>
          <div><i style="width:{pct}%"></i></div>
          <b>{int(r[count_col])}</b>
        </div>
        """
    return rows

project = read_csv(PROJECT_VIEW)
week3 = read_csv(WEEK3)
actions = read_csv(EQUITY_ACTIONS)
shortlist = read_csv(EQUITY_SHORTLIST)
review_summary = read_csv(EQUITY_REVIEW_SUMMARY)

action_count_col = "row_count" if "row_count" in actions.columns else "rows"

fairness_counts = pd.DataFrame()
if not shortlist.empty and "fairness_category" in shortlist.columns:
    fairness_counts = (
        shortlist["fairness_category"]
        .value_counts()
        .rename_axis("fairness_category")
        .reset_index(name="review_rows")
    )

flag_counts = pd.DataFrame()
if not shortlist.empty and "review_flags" in shortlist.columns:
    expanded = []
    for value in shortlist["review_flags"].dropna():
        for flag in str(value).split(";"):
            flag = flag.strip()
            if flag:
                expanded.append(flag)

    if expanded:
        flag_counts = (
            pd.Series(expanded)
            .value_counts()
            .rename_axis("review_flag")
            .reset_index(name="review_rows")
        )

html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Responsible AI Healthcare LLM Final Dashboard</title>
<style>
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: #f7f8fb;
  color: #172033;
}}
header {{
  background: white;
  color: #172033;
  padding: 32px 48px 20px;
  border-bottom: 1px solid #dfe4ec;
}}
header h1 {{
  margin: 0 0 8px;
  font-size: 34px;
}}
header p {{
  margin: 0;
  color: #64748b;
  font-size: 18px;
}}
main {{
  max-width: 1240px;
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
  padding: 18px;
}}
.metric span, .metric small {{
  display: block;
  color: #64748b;
}}
.metric strong {{
  display: block;
  font-size: 34px;
  margin: 8px 0;
}}
section {{
  padding: 20px;
  margin-bottom: 18px;
  break-inside: avoid;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 21px;
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
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 230px 1fr 44px;
  gap: 12px;
  align-items: center;
  margin: 12px 0;
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
  main {{ padding-top: 18px; }}
  .metrics {{ grid-template-columns: repeat(4, 1fr); }}
}}
</style>
</head>
<body>
<header>
  <h1>Responsible AI Healthcare LLM Dashboard</h1>
  <p>Project-level summary of hallucination detection, grounding, fairness review, and safety routing.</p>
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

  <div class="two-col">
    <section>
      <h2>EquityMedQA Trust Routing</h2>
      <p class="muted">Rows routed by the trustworthiness layer.</p>
      {bar_chart(actions, "recommended_action", action_count_col)}
    </section>

    <section>
      <h2>Fairness Categories Needing Review</h2>
      <p class="muted">Manual review shortlist grouped by fairness category.</p>
      {bar_chart(fairness_counts, "fairness_category", "review_rows")}
    </section>
  </div>

  <section>
    <h2>Review Flags Needing Attention</h2>
    <p class="muted">The dashboard shows review themes instead of long clinical responses. Full cases remain in the generated CSV files.</p>
    {table(flag_counts, 12)}
  </section>

  <section>
    <h2>EquityMedQA Review Summary</h2>
    {table(review_summary)}
  </section>
</main>
</body>
</html>
"""

OUT_HTML.write_text(html_doc, encoding="utf-8")
print("Final dashboard generated successfully")
print("Saved:", OUT_HTML)
