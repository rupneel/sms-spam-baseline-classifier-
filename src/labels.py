import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_FILE   = os.path.join(PROJECT_ROOT, "data", "cleaned", "sms_clean.csv")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports")
FIGURE_DIR   = os.path.join(PROJECT_ROOT, "outputs", "figures")
def _ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
def load_clean_data() -> pd.DataFrame:
    if not os.path.exists(CLEAN_FILE):
        print("[labels] ERROR: cleaned data not found. Run ingest.py first.")
        sys.exit(1)
    return pd.read_csv(CLEAN_FILE)
# CHECK 1 — Allowed label values
def check_allowed_values(df: pd.DataFrame) -> dict:
    unique = sorted(df["label"].unique().tolist())
    expected = ["ham", "spam"]
    is_clean = unique == expected

    result = {
        "expected": expected,
        "found":    unique,
        "pass":     is_clean,
    }
    if is_clean:
        print("[labels] ✓ Label values are exactly {'ham', 'spam'}")
    else:
        print(f"[labels] ✗ Unexpected label values found: {unique}")
    return result
# CHECK 2 — No nulls in the label column
def check_label_nulls(df: pd.DataFrame) -> dict:
    n_null = int(df["label"].isnull().sum())
    is_clean = n_null == 0

    result = {
        "null_count": n_null,
        "pass":       is_clean,
    }
    if is_clean:
        print("[labels] ✓ No null labels")
    else:
        print(f"[labels] ✗ Found {n_null} null label(s)")
    return result
# CHECK 3 — Class distribution & imbalance ratio
def check_class_balance(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    total  = len(df)

    majority_label = counts.index[0]
    minority_label = counts.index[-1]
    imbalance_ratio = round(counts.iloc[0] / counts.iloc[-1], 2)
    result = {
        "counts": {label: int(cnt) for label, cnt in counts.items()},
        "percentages": {
            label: round(int(cnt) / total * 100, 2)
            for label, cnt in counts.items()
        },
        "majority_class":  majority_label,
        "minority_class":  minority_label,
        "imbalance_ratio": imbalance_ratio,
        "total_rows":      total,
    }
    print(f"[labels] Class counts:      {result['counts']}")
    print(f"[labels] Class percentages:  {result['percentages']}")
    print(f"[labels] Imbalance ratio:    {imbalance_ratio}:1 "
          f"({majority_label} / {minority_label})")

    if imbalance_ratio > 3:
        print("[labels] ⚠ Significant class imbalance detected — "
              "use F1 / precision / recall, not just accuracy.")

    return result
# CHECK 4 — Label-vs-message cross-check (spot check)
def spot_check_samples(df: pd.DataFrame, n: int = 5) -> dict:
    samples = {}
    for label in ["ham", "spam"]:
        subset = df.loc[df["label"] == label, "message"]
        chosen = subset.sample(n=min(n, len(subset)), random_state=42)
        samples[label] = chosen.tolist()

    return samples
# CHECK 5 — Majority-class baseline accuracy
def majority_baseline(df: pd.DataFrame) -> dict:
    majority = df["label"].value_counts().index[0]
    baseline_acc = round(
        (df["label"] == majority).mean() * 100, 2
    )

    result = {
        "majority_class":    majority,
        "baseline_accuracy": baseline_acc,
    }
    print(f"\n[labels] Majority-class baseline: predict '{majority}' "
          f"for every message → {baseline_acc}% accuracy")
    return result
# PLOT — Label distribution pie chart
def plot_label_pie(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0, 0.06)  # slightly pop the spam slice
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, explode=explode, startangle=140,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "label_distribution_pie.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[labels] Saved → {path}")
    return path
# SAVE — JSON summary report
def save_report(results: dict) -> str:
    path = os.path.join(REPORT_DIR, "label_sanity_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[labels] Saved report → {path}")
    return path
# MAIN
def main():
    print("=" * 60)
    print(" SMS Spam Classifier — Label Sanity Checks")
    print("=" * 60)

    _ensure_dirs()
    df = load_clean_data()

    results = {}
    results["allowed_values"]   = check_allowed_values(df)
    results["null_check"]       = check_label_nulls(df)
    results["class_balance"]    = check_class_balance(df)
    results["spot_check"]       = spot_check_samples(df, n=5)
    results["majority_baseline"] = majority_baseline(df)

    plot_label_pie(df)
    save_report(results)

    # --- Final verdict -------------------------------------------------------
    all_passed = all([
        results["allowed_values"]["pass"],
        results["null_check"]["pass"],
    ])

    print("\n" + "=" * 60)
    if all_passed:
        print(" ✓ All label sanity checks PASSED")
    else:
        print(" ✗ Some label sanity checks FAILED — review report")
    print("=" * 60)

    return results
if __name__ == "__main__":
    main()
