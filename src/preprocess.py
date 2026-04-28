import os
import re
import sys
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_FILE   = os.path.join(PROJECT_ROOT, "data", "cleaned", "sms_clean.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURE_DIR   = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
def _ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
def load_clean_data() -> pd.DataFrame:
    if not os.path.exists(CLEAN_FILE):
        print("[preprocess] ERROR: cleaned data not found. Run ingest.py first.")
        sys.exit(1)
    return pd.read_csv(CLEAN_FILE)
# STEP 1 — Text cleaning
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\d+", " ", text)                  # digits
    text = re.sub(r"[^a-z\s]", " ", text)             # non-alpha
    text = re.sub(r"\s+", " ", text).strip()           # extra spaces
    return text
def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_message"] = df["message"].apply(clean_text)
    # quick sanity print
    print("[preprocess] Cleaning examples:")
    for i in range(min(3, len(df))):
        print(f"  ORIGINAL : {df.iloc[i]['message'][:80]}")
        print(f"  CLEANED  : {df.iloc[i]['clean_message'][:80]}\n")
    return df
# STEP 2 — Encode labels
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label_map = {"ham": 0, "spam": 1}
    df["label_enc"] = df["label"].map(label_map)
    print(f"[preprocess] Label encoding: {label_map}")
    return df
# STEP 3 — Train / test split (BEFORE fitting the vectoriser)
def split_data(df: pd.DataFrame):
    X = df["clean_message"]
    y = df["label_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[preprocess] Train set: {len(X_train)} rows  |  "
          f"Test set: {len(X_test)} rows")
    print(f"[preprocess] Train spam%: "
          f"{y_train.mean() * 100:.1f}%  |  "
          f"Test spam%: {y_test.mean() * 100:.1f}%")

    return X_train, X_test, y_train, y_test
# STEP 4 — TF-IDF vectorisation
def vectorise(X_train, X_test):
    vectoriser = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_tfidf = vectoriser.fit_transform(X_train)
    X_test_tfidf  = vectoriser.transform(X_test)

    vocab_size = len(vectoriser.vocabulary_)
    print(f"[preprocess] Vocabulary size: {vocab_size:,} features")
    print(f"[preprocess] Train matrix:    {X_train_tfidf.shape}")
    print(f"[preprocess] Test  matrix:    {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectoriser
# STEP 5 — Top features inspection
def inspect_top_features(vectoriser, n: int = 20) -> dict:
    feature_names = vectoriser.get_feature_names_out()
    idf_scores    = vectoriser.idf_
    top_idx = idf_scores.argsort()[::-1][:n]
    top_features = [
        {"feature": feature_names[i], "idf": round(float(idf_scores[i]), 3)}
        for i in top_idx
    ]
    print(f"\n[preprocess] Top {n} features by IDF (most discriminative):")
    for rank, f in enumerate(top_features, 1):
        print(f"  {rank:>2}. {f['feature']:<25} IDF = {f['idf']}")
    return {"top_features": top_features}
# STEP 6 — Plots
def plot_message_length_before_after(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # Before
    df["raw_len"] = df["message"].str.len()
    for label, color in [("ham", "#2ecc71"), ("spam", "#e74c3c")]:
        subset = df.loc[df["label"] == label, "raw_len"]
        axes[0].hist(subset, bins=50, alpha=0.6, color=color,
                     label=label, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Before Cleaning", fontweight="bold")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    # After
    df["clean_len"] = df["clean_message"].str.len()
    for label, color in [("ham", "#2ecc71"), ("spam", "#e74c3c")]:
        subset = df.loc[df["label"] == label, "clean_len"]
        axes[1].hist(subset, bins=50, alpha=0.6, color=color,
                     label=label, edgecolor="white", linewidth=0.5)
    axes[1].set_title("After Cleaning", fontweight="bold")
    axes[1].set_xlabel("Characters")
    axes[1].legend()
    plt.suptitle("Message Length Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "preprocess_length_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[preprocess] Saved → {path}")
    return path
# STEP 7 — Save artefacts
def save_artefacts(X_train_tfidf, X_test_tfidf, y_train, y_test,
                   vectoriser, report: dict):
    # TF-IDF matrices
    save_npz(os.path.join(PROCESSED_DIR, "X_train.npz"), X_train_tfidf)
    save_npz(os.path.join(PROCESSED_DIR, "X_test.npz"),  X_test_tfidf)
    print("[preprocess] Saved TF-IDF matrices → data/processed/")
    # Labels
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"),  index=False)
    print("[preprocess] Saved labels → data/processed/")
    # Vectoriser
    vec_path = os.path.join(PROCESSED_DIR, "tfidf_vectoriser.joblib")
    joblib.dump(vectoriser, vec_path)
    print(f"[preprocess] Saved vectoriser → {vec_path}")
    # Report
    report_path = os.path.join(REPORT_DIR, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] Saved report → {report_path}")
# MAIN
def main():
    print("=" * 60)
    print(" SMS Spam Classifier — Text Preprocessing")
    print("=" * 60)
    _ensure_dirs()
    # 1. Load
    df = load_clean_data()
    print(f"[preprocess] Loaded {len(df)} rows\n")
    # 2. Clean text
    df = apply_cleaning(df)
    # 3. Encode labels
    df = encode_labels(df)
    # 4. Plot before / after
    plot_message_length_before_after(df)
    # 5. Split
    X_train, X_test, y_train, y_test = split_data(df)
    # 6. Vectorise
    X_train_tfidf, X_test_tfidf, vectoriser = vectorise(X_train, X_test)
    # 7. Inspect top features
    feature_report = inspect_top_features(vectoriser)
    # 8. Save everything
    report = {
        "total_rows":     len(df),
        "train_rows":     len(X_train),
        "test_rows":      len(X_test),
        "test_size":      TEST_SIZE,
        "random_state":   RANDOM_STATE,
        "vocab_size":     len(vectoriser.vocabulary_),
        "max_features":   5000,
        "ngram_range":    [1, 2],
        "sublinear_tf":   True,
        "stop_words":     "english",
        **feature_report,
    }
    save_artefacts(X_train_tfidf, X_test_tfidf, y_train, y_test,
                   vectoriser, report)
    print("\n" + "=" * 60)
    print(" Preprocessing complete!")
    print("=" * 60)
    return report
if __name__ == "__main__":
    main()
