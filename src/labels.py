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
        print("ERROR: cleaned data not found. Please run ingest.py first.")
        sys.exit(1) #`sys.exit(1)` immediately stops the program and tells the system that it ended with an error.
    return pd.read_csv(CLEAN_FILE) 
# Checks for allowed label values
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
        print("Label values are exactly {'ham', 'spam'}")
    else:
        print(f"Error : Unexpected label values found: {unique}")
    return result
# Checking that there are no null values in the label column
def check_label_nulls(df: pd.DataFrame) -> dict:
    n_null = int(df["label"].isnull().sum())
    is_clean = n_null == 0
    result = {
        "null_count": n_null,
        "pass":       is_clean,
    }
    if is_clean:
        print("No null labels")
    else:
        print(f"Error: Found {n_null} null label(s)")
    return result
# Checking the class distribution & imbalance ratio
def check_class_balance(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    total  = len(df)
    majority_label = counts.index[0] #most = ham
    minority_label = counts.index[-1] #least = spam
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
    print(f"Class counts:      {result['counts']}")
    print(f"Class percentages:  {result['percentages']}")
    print(f"Imbalance ratio:    {imbalance_ratio}:1 "
          f"({majority_label} / {minority_label})")
    if imbalance_ratio > 3:
        print("⚠ Significant class imbalance detected — "
              "use F1 / precision / recall, not just accuracy.")
    return result
# CHECK 5 — Majority-class baseline accuracy
def majority_baseline(df: pd.DataFrame) -> dict:
    majority = df["label"].value_counts().index[0]
    baseline_acc = round(
        (df["label"] == majority).mean() * 100, 2          #true or false
    )
    result = {
        "majority_class":    majority,
        "baseline_accuracy": baseline_acc,
    }
    print(f"\nMajority-class baseline: predict '{majority}' "
          f"for every message → {baseline_acc}% accuracy")
    return result
# PLOT — Label distribution pie chart
def plot_label_pie(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5)) #Makes a square plot (good for pie charts) , fig is the overall container, ax is the actual chart
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0, 0.06)  # slightly pop the spam slice
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%", #Shows percentage text on slices
        colors=colors, explode=explode, startangle=140, #Custom colors for each slice,Separates slices slightly,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "label_distribution_pie.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")
    return path
# SAVE — JSON summary report
def save_report(results: dict) -> str:
    path = os.path.join(REPORT_DIR, "label_sanity_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False) #returns a string instead , False → keeps normal readable text , True (default) → converts to Unicode escape sequences
    print(f"Saved report → {path}")
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
    results["majority_baseline"] = majority_baseline(df)
    plot_label_pie(df)
    save_report(results)
    # Final verdict
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