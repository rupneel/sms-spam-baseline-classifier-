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
  • predict_proba() gives us confidence scores.
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
    confusion_matrix,
    accuracy_score,
)

# CONFIGURATION

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR     = os.path.join(PROJECT_ROOT, "outputs", "models")
REPORT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "reports")
FIGURE_DIR    = os.path.join(PROJECT_ROOT, "outputs", "figures")


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

# Load artefacts

def load_artefacts():
    for fname in ["X_train.npz", "X_test.npz", "y_train.csv", "y_test.csv"]:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"[train] ERROR: {path} not found. Run preprocess.py first.")
            sys.exit(1)

    # Loads the sparse TF-IDF matrices
    X_train = load_npz(os.path.join(PROCESSED_DIR, "X_train.npz"))
    X_test  = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))

    # Loads the labels (they are single-column CSVs)
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze().values

    print(f"Loaded training set:  {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"Loaded test set:      {X_test.shape[0]} rows × {X_test.shape[1]} features")
    print(f"Train spam count:     {y_train.sum()} / {len(y_train)}")
    print(f"Test  spam count:     {y_test.sum()} / {len(y_test)}")

    return X_train, X_test, y_train, y_test


# TRAINING NAIVE BAYES


def train_model(X_train, y_train):

    model = MultinomialNB()          
    model.fit(X_train, y_train)
    print(f"Classes learned: {model.classes_}")
    print(f"Alpha (smoothing): {model.alpha}")

    return model


# PART 3 — PREDICT

def predict(model, X_test):
    y_pred = model.predict(X_test)

    # Quick summary
    n_pred_spam = y_pred.sum()
    n_pred_ham  = len(y_pred) - n_pred_spam
    print(f"\nPredictions on test set:")
    print(f"Predicted ham:  {n_pred_ham}")
    print(f"Predicted spam: {n_pred_spam}")

    return y_pred


# PART 4 — EVALUATE


def evaluate(y_test, y_pred):
    # Core metric 
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()


    # Prints everything 
    print(f"\n{'=' * 60}")
    print(f" EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Accuracy  : {acc}%")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Ham    Predicted Spam")
    print(f"    Actual Ham       {tn:>8}           {fp:>5}")
    print(f"    Actual Spam      {fn:>8}           {tp:>5}")
    print(f"\n  True Negatives  (TN): {tn}  — ham correctly identified")
    print(f"  False Positives (FP): {fp}  — ham wrongly called spam")
    print(f"  False Negatives (FN): {fn}  — spam that slipped through")
    print(f"  True Positives  (TP): {tp}  — spam correctly caught")

    results = {
        "accuracy": acc,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
    }

    return results

# Confusion Matrix Heatmap

def plot_confusion_matrix(y_test, y_pred) -> str:
    
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


# Saving Model and evaluation report

def save_model(model) -> str:
    """Save the trained model for later use."""
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

# MAIN

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
    y_pred = predict(model, X_test)

    # PART 4: Evaluate performance
    results = evaluate(y_test, y_pred)

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
