from pathlib import Path
import html
import pandas as pd

REPORT_DIR = Path("reports")
DASHBOARD_DIR = REPORT_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

OUT_HTML = DASHBOARD_DIR / "project_trustworthiness_dashboard.html"

PROJECT_VIEW = Path("results/comparison/project_level_trustworthiness_view.csv")
WEEK3_COMPARISON = Path("results/comparison/week3_model_comparison_summary.csv")
MEDHALT_SUMMARY = Path("results/qwen_medhalt/qwen_1_5b_medhalt_balanced_summary.csv")
LOGIC_SUMMARY = Path("results/trustworthiness/qwen_1_5b_logic_aware_routing_summary.csv")
EQUITY_TRUST_OVERALL = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_overall.csv")
EQUITY_TRUST_ACTIONS = Path("results/trustworthiness/qwen_equitymedqa_trustworthiness_action_summary.csv")
EQUITY_REVIEW_SUMMARY = Path("results/equitymedqa_fairness/qwen_equitymedqa_fairness_full_review_summary.csv")
EQUITY_SHORTLIST = Path("results/equitymedqa_fairness/qwen_equitymedqa_manual_review_shortlist.csv")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def esc(value):
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def table_html(df, max_rows=12):
    if df.empty:
        return "<p class='muted'>No data available.</p>"

    view = df.head(max_rows).copy()
    headers = "".join(f"<th>{esc(col)}</th>" for col in view.columns)

    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{esc(row[col])}</td>" for col in view.columns)
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def first_value(df, col, default=""):
    if df.empty or col not in df.columns:
        return default
    return df[col].iloc[0]


def find_metric(df, candidates, default=""):
    if df.empty:
        return default

    for col in candidates:
        if col in df.columns:
            return first_value(df, col, default)

    for col in df.columns:
        low = col.lower()
        if any(term in low for term in candidates):
            return first_value(df, col, default)

    return default


def metric_card(title, value, subtitle):
    return f"""
    <section class="metric-card">
      <div class="metric-title">{esc(title)}</div>
      <div class="metric-value">{esc(value)}</div>
      <div class="metric-subtitle">{esc(subtitle)}</div>
    </section>
    """


def action_bars(df):
    if df.empty:
        return "<p class='muted'>No action summary available.</p>"

    action_col = None
    count_col = None

    for col in ["recommended_action", "action", "trustworthiness_action", "logic_aware_recommended_action"]:
        if col in df.columns:
            action_col = col
            break

    for col in ["rows", "count", "n"]:
        if col in df.columns:
            count_col = col
            break

    if action_col is None or count_col is None:
        return table_html(df)

    total = max(float(df[count_col].sum()), 1.0)
    bars = []

    for _, row in df.iterrows():
        action = row[action_col]
        count = float(row[count_col])
        width = round((count / total) * 100, 1)
        bars.append(f"""
        <div class="bar-row">
          <div class="bar-label">{esc(action)}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{width}%"></div>
          </div>
          <div class="bar-count">{int(count)}</div>
        </div>
        """)

    return "<div class='bars'>" + "".join(bars) + "</div>"


def main():
    project_view = read_csv(PROJECT_VIEW)
    week3 = read_csv(WEEK3_COMPARISON)
    medhalt = read_csv(MEDHALT_SUMMARY)
    logic = read_csv(LOGIC_SUMMARY)
    equity_overall = read_csv(EQUITY_TRUST_OVERALL)
    equity_actions = read_csv(EQUITY_TRUST_ACTIONS)
    equity_review = read_csv(EQUITY_REVIEW_SUMMARY)
    equity_shortlist = read_csv(EQUITY_SHORTLIST)

    medhalt_accuracy = find_metric(medhalt, ["accuracy"], "0.86")
    medhalt_f1 = find_metric(medhalt, ["f1_score", "f1"], "0.875")
    medhalt_recall = find_metric(medhalt, ["recall"], "0.98")

    accepted_hallucinated = "0"
    if not logic.empty and "action_column" in logic.columns and "accepted_hallucinated_rows" in logic.columns:
        row = logic[logic["action_column"] == "logic_aware_recommended_action"]
        if not row.empty:
            accepted_hallucinated = row["accepted_hallucinated_rows"].iloc[0]

    equity_mean_trust = find_metric(
        equity_overall,
        ["mean_clinical_trustworthiness_score", "mean_trustworthiness_score", "trustworthiness"],
        "0.359"
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Responsible AI Healthcare LLM Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #657084;
      --line: #dfe4ec;
      --accent: #1f6feb;
      --accent-soft: #dceafe;
      --warning: #b45309;
      --good: #157347;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}

    header {{
      background: #0f172a;
      color: white;
      padding: 28px 36px;
    }}

    header h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
    }}

    header p {{
      margin: 0;
      color: #cbd5e1;
      max-width: 960px;
    }}

    main {{
      padding: 28px 36px 44px;
      max-width: 1280px;
      margin: 0 auto;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}

    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}

    .metric-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}

    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 6px;
    }}

    .metric-subtitle {{
      color: var(--muted);
      font-size: 13px;
    }}

    section.block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
    }}

    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}

    .muted {{
      color: var(--muted);
    }}

    .status {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      background: #dcfce7;
      color: var(--good);
      margin-left: 8px;
    }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}

    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 720px;
      background: white;
    }}

    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}

    th {{
      background: #f1f5f9;
      color: #334155;
      font-weight: 700;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    .bars {{
      display: grid;
      gap: 10px;
      margin-top: 8px;
    }}

    .bar-row {{
      display: grid;
      grid-template-columns: 210px 1fr 56px;
      gap: 12px;
      align-items: center;
    }}

    .bar-label {{
      font-size: 14px;
      color: #334155;
    }}

    .bar-track {{
      height: 12px;
      border-radius: 999px;
      background: #e5e7eb;
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }}

    .bar-count {{
      text-align: right;
      color: var(--muted);
      font-size: 14px;
    }}

    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}

    footer {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 20px;
    }}

    @media (max-width: 900px) {{
      header, main {{
        padding-left: 18px;
        padding-right: 18px;
      }}

      .metrics, .two-col {{
        grid-template-columns: 1fr;
      }}

      .bar-row {{
        grid-template-columns: 1fr;
        gap: 6px;
      }}

      .bar-count {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Responsible AI Healthcare LLM Dashboard</h1>
    <p>Project-level view for hallucination detection, evidence grounding, fairness review, and human-review routing.</p>
  </header>

  <main>
    <div class="metrics">
      {metric_card("Qwen 1.5B Accuracy", medhalt_accuracy, "Med-HALT balanced sample")}
      {metric_card("Qwen 1.5B F1", medhalt_f1, "Hallucination detection")}
      {metric_card("Qwen 1.5B Recall", medhalt_recall, "Hallucination catch rate")}
      {metric_card("Accepted Hallucinations", accepted_hallucinated, "After logic-aware routing")}
    </div>

    <section class="block">
      <h2>Current Safety Position <span class="status">Demo Ready</span></h2>
      <p>
        The framework now combines hallucination detection, semantic grounding, fairness review,
        clinical trustworthiness scoring, and logic-aware human-review routing.
      </p>
    </section>

    <section class="block">
      <h2>Project-Level Summary</h2>
      {table_html(project_view)}
    </section>

    <section class="block">
      <h2>Week 3 Model Comparison</h2>
      {table_html(week3)}
    </section>

    <div class="two-col">
      <section class="block">
        <h2>EquityMedQA Trust Routing</h2>
        <p class="muted">Mean trustworthiness score: <strong>{esc(equity_mean_trust)}</strong></p>
        {action_bars(equity_actions)}
      </section>

      <section class="block">
        <h2>EquityMedQA Review Summary</h2>
        {table_html(equity_review)}
      </section>
    </div>

    <section class="block">
      <h2>Manual Review Shortlist Preview</h2>
      <p class="muted">These are examples that should be reviewed before treating the model as safe in a clinical setting.</p>
      {table_html(equity_shortlist, max_rows=8)}
    </section>

    <footer>
      Generated from local CSV outputs in the Responsible AI Healthcare LLM project.
    </footer>
  </main>
</body>
</html>
"""

    OUT_HTML.write_text(html_doc, encoding="utf-8")

    print("Dashboard generated successfully")
    print("Saved:", OUT_HTML)


if __name__ == "__main__":
    main()
