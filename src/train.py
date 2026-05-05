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
    print(f"Loaded training set:  {X_train.shape[0]} rows × {X_train.shape[1]} features") #shape[0] → number of rows and shape[1] → number of features (unique words/vocabulary size)
    print(f"Loaded test set:      {X_test.shape[0]} rows × {X_test.shape[1]} features") #shape[0] → number of rows and shape[1] → number of features (unique words/vocabulary size)
    print(f"Train spam count:     {y_train.sum()} / {len(y_train)}")
    print(f"Test  spam count:     {y_test.sum()} / {len(y_test)}")
    return X_train, X_test, y_train, y_test
# TRAINING NAIVE BAYES
def train_model(X_train, y_train):
    model = MultinomialNB()          
    model.fit(X_train, y_train) #.fit() trains model using the TF-IDF matrix and the labels
    print(f"Classes learned: {model.classes_}") #built in [0,1] 0-ham and 1-spam , list of possible answers
    print(f"Alpha (smoothing): {model.alpha}") #built in, smoothing parameter
    return model
# PREDICTING
def predict(model, X_test):
    y_pred = model.predict(X_test) #Model takes unseen messages (X_test) and Predicts 0 → ham 1 → spam , just the score part
    # Quick summary
    n_pred_spam = y_pred.sum() #sum() counts the 1s (spam)
    n_pred_ham  = len(y_pred) - n_pred_spam #the rest are ham
    print(f"\nPredictions on test set:")
    print(f"Predicted ham:  {n_pred_ham}")
    print(f"Predicted spam: {n_pred_spam}")
    return y_pred
# EVALUATING
def evaluate(y_test, y_pred):
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() #.ravel() flattens the matrix into a 1D array
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
    path = os.path.join(FIGURE_DIR, "confusion_matrix.png") ####
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved -> {path}")
    return path
# Saving Model and evaluation report
def save_model(model) -> str:
    path = os.path.join(MODEL_DIR, "naive_bayes.joblib")
    joblib.dump(model, path)
    print(f"[train] Saved model -> {path}")
    return path
def save_evaluation_report(results: dict) -> str:
    path = os.path.join(REPORT_DIR, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[train] Saved report -> {path}")
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
