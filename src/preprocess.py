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
    os.makedirs(PROCESSED_DIR, exist_ok=True) #don’t crash if folder already exists
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
def load_clean_data() -> pd.DataFrame:
    if not os.path.exists(CLEAN_FILE):
        print("ERROR: cleaned data not found. Run ingest.py first.")
        sys.exit(1) #Stops the program immediately. 1 is the error code
    return pd.read_csv(CLEAN_FILE)
# Text cleaning
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs if those patterns are detected it leaves whitespace on there in the text 
    text = re.sub(r"\d+", " ", text)                  # digits
    text = re.sub(r"[^a-z\s]", " ", text)             # non-alpha
    text = re.sub(r"\s+", " ", text).strip()           # extra spaces
    return text
def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_message"] = df["message"].apply(clean_text) #columns
    return df
# Encode labels
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label_map = {"ham": 0, "spam": 1}
    df["label_enc"] = df["label"].map(label_map) #replaces ham and spam with 0 and 1
    print(f"Label encoding: {label_map}")
    return df
# Train / test split (BEFORE fitting the vectoriser)
def split_data(df: pd.DataFrame):
    X = df["clean_message"]
    y = df["label_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,  #0.20
        random_state=RANDOM_STATE, #42
        stratify=y, #Keeps same class distribution in both train and test sets. , y are the lables 90:10
    )
    print(f"Train set: {len(X_train)} rows  |  "
          f"Test set: {len(X_test)} rows")
    print(f"Train spam%: "
          f"{y_train.mean() * 100:.1f}%  |  "
          f"Test spam%: {y_test.mean() * 100:.1f}%")
    return X_train, X_test, y_train, y_test
# TF-IDF vectorisation
def vectorise(X_train, X_test): #cleaned text
    vectoriser = TfidfVectorizer(
        max_features=5000, # top 5000 important features
        ngram_range=(1, 2), #consider words and phrases of 1-2 words
        sublinear_tf=True, #applies tf=1+log(tf) this Reduces impact of very frequent words
        stop_words="english", #removes common words like a,the,is,etc
    )
    X_train_tfidf = vectoriser.fit_transform(X_train) # learns vocabulary calculates IDF scores and converts text to numbers
    X_test_tfidf  = vectoriser.transform(X_test) #Uses same vocabulary from training,Converts test text into TF-IDF using that vocabulary
    vocab_size = len(vectoriser.vocabulary_)
    print(f"Vocabulary size: {vocab_size:,} features")
    print(f"Train matrix:    {X_train_tfidf.shape}")
    print(f"Test  matrix:    {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectoriser
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
    plt.tight_layout() #Fixes spacing so plots don’t overlap
    path = os.path.join(FIGURE_DIR, "preprocess_length_comparison.png")
    fig.savefig(path, dpi=150) #good quality
    plt.close(fig)
    print(f"Saved -> {path}")
    return path
# STEP 7 — Save artefacts
def save_artefacts(X_train_tfidf, X_test_tfidf, y_train, y_test,
                   vectoriser, report: dict):
    # TF-IDF matrices
    save_npz(os.path.join(PROCESSED_DIR, "X_train.npz"), X_train_tfidf) #saves train data as a compressed sparse matrix
    save_npz(os.path.join(PROCESSED_DIR, "X_test.npz"),  X_test_tfidf)
    print("Saved TF-IDF matrices -> data/processed/")
    # Labels
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"),  index=False)
    print("Saved labels -> data/processed/")
    # Vectoriser
    vec_path = os.path.join(PROCESSED_DIR, "tfidf_vectoriser.joblib") #filename
    joblib.dump(vectoriser, vec_path) #Saves trained TfidfVectorizer ,
    print(f"Saved vectoriser -> {vec_path}")
    # Report
    report_path = os.path.join(REPORT_DIR, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report -> {report_path}")
# MAIN
def main():
    print("=" * 60)
    print(" SMS Spam Classifier — Text Preprocessing")
    print("=" * 60)
    _ensure_dirs()
    # 1. Load
    df = load_clean_data()
    print(f"Loaded {len(df)} rows\n")
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
    # 7. Save everything
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
    }
    save_artefacts(X_train_tfidf, X_test_tfidf, y_train, y_test,
                   vectoriser, report)
    print("\n" + "=" * 60)
    print(" Preprocessing complete!")
    print("=" * 60)
    return report
if __name__ == "__main__":
    main()
