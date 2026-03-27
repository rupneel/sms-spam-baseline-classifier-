"""
train.py — Model Training & Evaluation
========================================
Trains a Multinomial Naive Bayes classifier on the TF-IDF features
produced by preprocess.py, evaluates it on the held-out test set,
and saves the trained model + evaluation report.

Pipeline:
  1. Load artefacts    — TF-IDF matrices and encoded labels
  2. Train model       — Fit MultinomialNB on training data
  3. Predict           — Generate class labels + probability scores
  4. Evaluate          — Precision, Recall, F1, Accuracy, Confusion Matrix
  5. Visualise         — Confusion matrix heatmap
  6. Save              — Model (.joblib) and evaluation report (.json)

Key concepts:
  • MultinomialNB works well with TF-IDF features for text classification
    because it models word frequencies as multinomial distributions.
  • We focus on spam-class F1 (not just accuracy) because the dataset
    is imbalanced (~87% ham, ~13% spam) — accuracy alone is misleading.
  • predict_proba() gives us confidence scores so we can later set
    thresholds for auto-block vs. flag-for-review.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR     = os.path.join(PROJECT_ROOT, "outputs", "models")
REPORT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "reports")
FIGURE_DIR    = os.path.join(PROJECT_ROOT, "outputs", "figures")


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


# =========================================================================
# PART 1 — LOAD ARTEFACTS
# =========================================================================
def load_artefacts():
    """
    Load the TF-IDF matrices and label arrays saved by preprocess.py.

    Files loaded:
      • X_train.npz  — sparse TF-IDF matrix (training features)
      • X_test.npz   — sparse TF-IDF matrix (test features)
      • y_train.csv  — encoded training labels (0=ham, 1=spam)
      • y_test.csv   — encoded test labels

    Why sparse matrices?
      TF-IDF produces mostly zeros (most words don't appear in most
      messages).  Sparse format stores only non-zero values, saving
      memory — a 4000×5000 matrix uses <1 MB sparse vs ~160 MB dense.

    Returns
    -------
    X_train, X_test : scipy.sparse.csr_matrix
    y_train, y_test : numpy.ndarray
    """
    # Check that processed data exists
    for fname in ["X_train.npz", "X_test.npz", "y_train.csv", "y_test.csv"]:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"[train] ERROR: {path} not found. Run preprocess.py first.")
            sys.exit(1)

    # Load sparse TF-IDF matrices
    X_train = load_npz(os.path.join(PROCESSED_DIR, "X_train.npz"))
    X_test  = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))

    # Load labels (they are single-column CSVs)
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze().values

    print(f"[train] Loaded training set:  {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"[train] Loaded test set:      {X_test.shape[0]} rows × {X_test.shape[1]} features")
    print(f"[train] Train spam count:     {y_train.sum()} / {len(y_train)}")
    print(f"[train] Test  spam count:     {y_test.sum()} / {len(y_test)}")

    return X_train, X_test, y_train, y_test


# =========================================================================
# PART 2 — TRAIN NAIVE BAYES
# =========================================================================
def train_model(X_train, y_train):
    """
    Fit a Multinomial Naive Bayes classifier on the training data.

    Why Naive Bayes?
      • It's a strong baseline for text classification — fast to train,
        no hyper-parameter tuning needed for a first pass.
      • "Naive" means it assumes features (words) are independent given
        the class.  This is obviously wrong for language, but in practice
        it works surprisingly well for spam detection.
      • MultinomialNB (not GaussianNB) because our features are TF-IDF
        scores — non-negative values representing word importance.

    Parameters
    ----------
    X_train : sparse matrix (n_samples, n_features)
    y_train : array of ints (0 or 1)

    Returns
    -------
    model : fitted MultinomialNB instance
    """
    model = MultinomialNB()          # default alpha=1.0 (Laplace smoothing)
    model.fit(X_train, y_train)

    print(f"\n[train] Model trained: MultinomialNB")
    print(f"[train] Classes learned: {model.classes_}")
    print(f"[train] Alpha (smoothing): {model.alpha}")

    return model


# =========================================================================
# PART 3 — PREDICT
# =========================================================================
def predict(model, X_test):
    """
    Generate predictions on the test set.

    Two types of output:
      1. y_pred       — hard class labels (0 or 1)
      2. y_pred_proba — probability scores for each class

    Why do we need probabilities?
      Hard labels just say "spam" or "ham", but probabilities tell us
      HOW CONFIDENT the model is.  This lets us later define:
        • High confidence (>90%) → auto-block
        • Medium confidence (60-90%) → flag for human review
        • Low confidence (<60%) → let through

    Parameters
    ----------
    model  : fitted MultinomialNB
    X_test : sparse matrix

    Returns
    -------
    y_pred       : array of ints (0 or 1)
    y_pred_proba : array of shape (n_samples, 2) — [P(ham), P(spam)]
    """
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Quick summary
    n_pred_spam = y_pred.sum()
    n_pred_ham  = len(y_pred) - n_pred_spam
    print(f"\n[train] Predictions on test set:")
    print(f"[train]   Predicted ham:  {n_pred_ham}")
    print(f"[train]   Predicted spam: {n_pred_spam}")
    print(f"[train]   Avg spam probability: {y_pred_proba[:, 1].mean():.4f}")

    return y_pred, y_pred_proba


# =========================================================================
# PART 4 — EVALUATE
# =========================================================================
def evaluate(y_test, y_pred, y_pred_proba):
    """
    Compute all evaluation metrics and print a full report.

    Metrics explained:
      • Accuracy  — % of all messages classified correctly.
                    Misleading here because predicting "ham" for everything
                    already gives ~87%.
      • Precision — Of all messages we CALLED spam, what % actually were?
                    High precision = few false alarms.
      • Recall    — Of all ACTUAL spam, what % did we catch?
                    High recall = few missed spam messages.
      • F1-score  — Harmonic mean of precision and recall.
                    Our PRIMARY metric because it balances both concerns.

    Confusion Matrix layout:
                        Predicted Ham    Predicted Spam
      Actual Ham           TN               FP
      Actual Spam          FN               TP

      TN = True Negative  (ham correctly classified)
      FP = False Positive (ham wrongly called spam — annoying for users)
      FN = False Negative (spam that got through — dangerous)
      TP = True Positive  (spam correctly caught)

    Parameters
    ----------
    y_test       : true labels
    y_pred       : predicted labels
    y_pred_proba : prediction probabilities

    Returns
    -------
    dict with all metrics
    """
    # --- Core metrics (spam = positive class = label 1) ---
    acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred) * 100, 2)
    rec  = round(recall_score(y_test, y_pred) * 100, 2)
    f1   = round(f1_score(y_test, y_pred) * 100, 2)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Full classification report (both classes) ---
    report_text = classification_report(
        y_test, y_pred,
        target_names=["ham", "spam"],
        digits=4,
    )

    # --- Print everything ---
    print(f"\n{'=' * 60}")
    print(f" EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Accuracy  : {acc}%")
    print(f"  Precision : {prec}%  (of predicted spam, how many were real spam?)")
    print(f"  Recall    : {rec}%  (of actual spam, how many did we catch?)")
    print(f"  F1-score  : {f1}%  (balanced measure — our primary metric)")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Ham    Predicted Spam")
    print(f"    Actual Ham       {tn:>8}           {fp:>5}")
    print(f"    Actual Spam      {fn:>8}           {tp:>5}")
    print(f"\n  True Negatives  (TN): {tn}  — ham correctly identified")
    print(f"  False Positives (FP): {fp}  — ham wrongly called spam")
    print(f"  False Negatives (FN): {fn}  — spam that slipped through")
    print(f"  True Positives  (TP): {tp}  — spam correctly caught")
    print(f"\n{report_text}")

    results = {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1_score":    f1,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
        "classification_report": report_text,
    }

    return results


# ---------------------------------------------------------------------------
# PLOT — Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred) -> str:
    """
    Visual heatmap of the confusion matrix.
    Makes it easy to spot where the model is making mistakes.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 16, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Naive Bayes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# SAVE — Model and evaluation report
# ---------------------------------------------------------------------------
def save_model(model) -> str:
    """Save the trained model for later use in scoring."""
    path = os.path.join(MODEL_DIR, "naive_bayes.joblib")
    joblib.dump(model, path)
    print(f"[train] Saved model → {path}")
    return path


def save_evaluation_report(results: dict) -> str:
    """Save all metrics as a JSON file for traceability."""
    path = os.path.join(REPORT_DIR, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[train] Saved report → {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" SMS Spam Classifier — Model Training & Evaluation")
    print("=" * 60)

    _ensure_dirs()

    # PART 1: Load artefacts
    X_train, X_test, y_train, y_test = load_artefacts()

    # PART 2: Train the model
    model = train_model(X_train, y_train)

    # PART 3: Predict on test set
    y_pred, y_pred_proba = predict(model, X_test)

    # PART 4: Evaluate performance
    results = evaluate(y_test, y_pred, y_pred_proba)

    # Visualise
    plot_confusion_matrix(y_test, y_pred)

    # Save
    save_model(model)
    save_evaluation_report(results)

    print("\n" + "=" * 60)
    print(" Training & Evaluation complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
