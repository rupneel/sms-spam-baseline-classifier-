"""
quality.py — Data Quality Report
==================================
Generates a comprehensive data quality report for the cleaned SMS dataset.
Covers missingness, duplicates, class balance, message length distribution,
and outputs an HTML report.

Key concepts:
  • Profiling a dataset before modelling
  • Generating HTML reports with Jinja2 templates
  • Detecting class imbalance
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless backend — no GUI window
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_FILE   = os.path.join(PROJECT_ROOT, "data", "cleaned", "sms_clean.csv")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports")
FIGURE_DIR   = os.path.join(PROJECT_ROOT, "outputs", "figures")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset produced by ingest.py."""
    if not os.path.exists(CLEAN_FILE):
        print("[quality] ERROR: cleaned data not found. Run ingest.py first.")
        sys.exit(1)
    return pd.read_csv(CLEAN_FILE)


# ---------------------------------------------------------------------------
# QUALITY CHECKS
# ---------------------------------------------------------------------------

def missing_values_report(df: pd.DataFrame) -> dict:
    """Check for missing / null values in every column."""
    total = len(df)
    report = {}
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        report[col] = {
            "missing_count": n_miss,
            "missing_pct": round(n_miss / total * 100, 2),
        }
    return report


def duplicate_report(df: pd.DataFrame) -> dict:
    """Check for any remaining duplicates after cleaning."""
    exact_dupes = int(df.duplicated().sum())
    msg_dupes   = int(df.duplicated(subset=["message"]).sum())
    return {
        "exact_duplicate_rows": exact_dupes,
        "duplicate_messages":   msg_dupes,
        "total_rows":           len(df),
    }


def class_balance(df: pd.DataFrame) -> dict:
    """Return class counts and percentages."""
    counts = df["label"].value_counts().to_dict()
    total  = len(df)
    return {
        label: {"count": cnt, "pct": round(cnt / total * 100, 2)}
        for label, cnt in counts.items()
    }


def message_length_stats(df: pd.DataFrame) -> dict:
    """Compute message-length statistics overall and per class."""
    df = df.copy()
    df["msg_len"] = df["message"].str.len()

    overall = df["msg_len"].describe().to_dict()
    per_class = {}
    for label in ["ham", "spam"]:
        subset = df.loc[df["label"] == label, "msg_len"]
        per_class[label] = subset.describe().to_dict()

    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of ham vs spam counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    counts.plot.bar(ax=ax, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.bar_label(ax.containers[0], fontsize=11, fontweight="bold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[quality] Saved → {path}")
    return path


def plot_message_lengths(df: pd.DataFrame):
    """Histogram of message lengths by class."""
    df = df.copy()
    df["msg_len"] = df["message"].str.len()

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [("ham", "#2ecc71"), ("spam", "#e74c3c")]:
        subset = df.loc[df["label"] == label, "msg_len"]
        ax.hist(subset, bins=60, alpha=0.6, label=label, color=color,
                edgecolor="white", linewidth=0.5)
    ax.set_title("Message Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "message_lengths.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[quality] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_html_report(missing, dupes, balance, lengths):
    """Build a self-contained HTML quality report."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data Quality Report — SMS Spam Classifier</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px;
         margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: .5rem; }}
  h2 {{ color: #2980b9; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #3498db; color: white; }}
  .ok  {{ color: #27ae60; font-weight: bold; }}
  .warn {{ color: #e67e22; font-weight: bold; }}
  .bad {{ color: #c0392b; font-weight: bold; }}
  img {{ max-width: 100%; margin: 1rem 0; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>📊 Data Quality Report</h1>
<p><strong>Dataset:</strong> SMS Spam Collection (cleaned)<br>
   <strong>Rows:</strong> {dupes['total_rows']:,}</p>

<h2>1. Missing Values</h2>
<table>
  <tr><th>Column</th><th>Missing Count</th><th>Missing %</th><th>Status</th></tr>
"""
    for col, info in missing.items():
        status_class = "ok" if info["missing_count"] == 0 else "bad"
        status_text  = "✓ Clean" if info["missing_count"] == 0 else "✗ Has nulls"
        html += (f'  <tr><td>{col}</td><td>{info["missing_count"]}</td>'
                 f'<td>{info["missing_pct"]}%</td>'
                 f'<td class="{status_class}">{status_text}</td></tr>\n')

    html += f"""</table>

<h2>2. Duplicate Analysis</h2>
<table>
  <tr><th>Metric</th><th>Count</th></tr>
  <tr><td>Exact duplicate rows</td><td>{dupes['exact_duplicate_rows']}</td></tr>
  <tr><td>Duplicate messages (text only)</td><td>{dupes['duplicate_messages']}</td></tr>
  <tr><td>Total rows</td><td>{dupes['total_rows']:,}</td></tr>
</table>

<h2>3. Class Balance</h2>
<table>
  <tr><th>Label</th><th>Count</th><th>Percentage</th></tr>
"""
    for label, info in balance.items():
        html += (f'  <tr><td>{label}</td><td>{info["count"]:,}</td>'
                 f'<td>{info["pct"]}%</td></tr>\n')

    html += f"""</table>
<img src="../figures/class_distribution.png" alt="Class Distribution">

<h2>4. Message Length Statistics</h2>
<table>
  <tr><th>Statistic</th><th>Overall</th><th>Ham</th><th>Spam</th></tr>
"""
    for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        overall_val = lengths["overall"].get(stat, "—")
        ham_val     = lengths["per_class"]["ham"].get(stat, "—")
        spam_val    = lengths["per_class"]["spam"].get(stat, "—")

        fmt = lambda v: f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int,)) else str(v)
        html += (f'  <tr><td>{stat}</td><td>{fmt(overall_val)}</td>'
                 f'<td>{fmt(ham_val)}</td><td>{fmt(spam_val)}</td></tr>\n')

    html += """</table>
<img src="../figures/message_lengths.png" alt="Message Lengths">

</body>
</html>
"""

    path = os.path.join(REPORT_DIR, "quality_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[quality] Saved HTML report → {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" SMS Spam Classifier — Data Quality Report")
    print("=" * 60)

    _ensure_dirs()
    df = load_clean_data()

    missing  = missing_values_report(df)
    dupes    = duplicate_report(df)
    balance  = class_balance(df)
    lengths  = message_length_stats(df)

    print(f"\n[quality] Missing values: {missing}")
    print(f"[quality] Duplicates:     {dupes}")
    print(f"[quality] Class balance:  {balance}")

    # Plots
    plot_class_distribution(df)
    plot_message_lengths(df)

    # HTML report
    generate_html_report(missing, dupes, balance, lengths)

    print("=" * 60)
    print(" Quality report complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
