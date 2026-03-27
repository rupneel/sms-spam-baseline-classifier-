"""
score.py — Scoring Demo
=========================
Classifies new SMS messages using the trained Naive Bayes model
and fitted TF-IDF vectoriser saved by train.py and preprocess.py.

Usage:
  # Score a single message from the command line:
  python src/score.py "Congratulations! You've won a free prize"

  # Score hardcoded examples (no arguments):
  python src/score.py

Key concepts:
  • At scoring time we re-use the SAME vectoriser that was fitted on
    training data.  This ensures words are mapped to the same feature
    indices the model learned.
  • We re-use clean_text() from preprocess.py so the input is cleaned
    in exactly the same way as the training data.
"""

import os
import sys
import re
import json
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(PROJECT_ROOT, "outputs", "models", "naive_bayes.joblib")
VECTORISER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tfidf_vectoriser.joblib")
SCORES_DIR     = os.path.join(PROJECT_ROOT, "outputs", "scores")

# Label mapping (same as preprocess.py)
LABEL_MAP = {0: "ham", 1: "spam"}


def _ensure_dirs():
    os.makedirs(SCORES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# TEXT CLEANING  (duplicated from preprocess.py for standalone use)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalise a single SMS message — identical to preprocess.clean_text().

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove digits
      4. Remove punctuation (keep only letters + spaces)
      5. Collapse whitespace

    Why duplicate instead of importing?
      score.py should work as a standalone script.  Importing from
      preprocess.py would require running that module's import chain
      (pandas, sklearn, scipy) even though we only need this one function.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\d+", " ", text)                  # digits
    text = re.sub(r"[^a-z\s]", " ", text)             # non-alpha
    text = re.sub(r"\s+", " ", text).strip()           # extra spaces
    return text


# ---------------------------------------------------------------------------
# LOAD MODEL + VECTORISER
# ---------------------------------------------------------------------------
def load_model_and_vectoriser():
    """
    Load the trained Naive Bayes model and the fitted TF-IDF vectoriser.

    Both were saved as .joblib files:
      • naive_bayes.joblib      — trained by train.py
      • tfidf_vectoriser.joblib — fitted by preprocess.py

    Returns
    -------
    model      : fitted MultinomialNB
    vectoriser : fitted TfidfVectorizer
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[score] ERROR: Model not found at {MODEL_PATH}")
        print("[score] Run train.py first to train the model.")
        sys.exit(1)

    if not os.path.exists(VECTORISER_PATH):
        print(f"[score] ERROR: Vectoriser not found at {VECTORISER_PATH}")
        print("[score] Run preprocess.py first to fit the vectoriser.")
        sys.exit(1)

    model      = joblib.load(MODEL_PATH)
    vectoriser = joblib.load(VECTORISER_PATH)

    print(f"[score] Model loaded:      {MODEL_PATH}")
    print(f"[score] Vectoriser loaded: {VECTORISER_PATH}")
    print(f"[score] Vocabulary size:   {len(vectoriser.vocabulary_):,} features\n")

    return model, vectoriser


# ---------------------------------------------------------------------------
# SCORE A SINGLE MESSAGE
# ---------------------------------------------------------------------------
def score_message(message: str, model, vectoriser) -> dict:
    """
    Classify a single SMS message.

    Pipeline:
      1. Clean the raw text       (clean_text)
      2. Transform to TF-IDF      (vectoriser.transform)
      3. Predict class + proba    (model.predict / predict_proba)

    Parameters
    ----------
    message    : raw SMS text string
    model      : fitted MultinomialNB
    vectoriser : fitted TfidfVectorizer

    Returns
    -------
    dict with keys: message, cleaned, prediction, confidence, probabilities
    """
    # Step 1: Clean
    cleaned = clean_text(message)

    # Step 2: Transform to TF-IDF (must pass a list of strings)
    tfidf_vector = vectoriser.transform([cleaned])

    # Step 3: Predict
    pred_label = model.predict(tfidf_vector)[0]
    pred_proba = model.predict_proba(tfidf_vector)[0]

    result = {
        "message":     message,
        "cleaned":     cleaned,
        "prediction":  LABEL_MAP[pred_label],
        "confidence":  round(float(max(pred_proba)) * 100, 2),
        "probabilities": {
            "ham":  round(float(pred_proba[0]) * 100, 2),
            "spam": round(float(pred_proba[1]) * 100, 2),
        },
    }

    return result


# ---------------------------------------------------------------------------
# BATCH SCORING — example messages
# ---------------------------------------------------------------------------
EXAMPLE_MESSAGES = [
    "Hey, are you coming to the party tonight?",
    "WINNER!! You have been selected for a cash prize of $5000! Call now!",
    "Hi Mom, I'll be home by 6pm. Love you!",
    "FREE entry in a weekly competition. Text WIN to 80085",
    "Can you pick up milk on your way home?",
    "Urgent! Your account has been compromised. Click here to verify.",
    "Meeting rescheduled to 3pm tomorrow. Please confirm.",
    "Congratulations! You've won a free iPhone. Claim now!",
    "Thanks for dinner last night, it was great!",
    "CASH PRIZE! Claim your reward now. Reply STOP to opt out.",
]


def batch_score(messages: list, model, vectoriser) -> list:
    """
    Score a list of messages and return results.

    Parameters
    ----------
    messages   : list of raw SMS strings
    model      : fitted MultinomialNB
    vectoriser : fitted TfidfVectorizer

    Returns
    -------
    list of result dicts
    """
    results = []
    for msg in messages:
        result = score_message(msg, model, vectoriser)
        results.append(result)
    return results


def print_results(results: list):
    """Pretty-print scoring results to the console."""
    print("=" * 70)
    print(" SCORING RESULTS")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        label = r["prediction"].upper()
        conf  = r["confidence"]

        # Colour indicator
        if label == "SPAM":
            indicator = "🚫 SPAM"
        else:
            indicator = "✅ HAM "

        print(f"\n  [{i}] {indicator}  ({conf}% confident)")
        print(f"      Message: {r['message'][:70]}{'…' if len(r['message']) > 70 else ''}")
        print(f"      Ham: {r['probabilities']['ham']}%  |  Spam: {r['probabilities']['spam']}%")

    print(f"\n{'=' * 70}")

    # Summary
    spam_count = sum(1 for r in results if r["prediction"] == "spam")
    ham_count  = len(results) - spam_count
    print(f"  Total: {len(results)} messages  |  "
          f"Ham: {ham_count}  |  Spam: {spam_count}")
    print("=" * 70)


def save_scores(results: list) -> str:
    """Save scoring results as a JSON file."""
    path = os.path.join(SCORES_DIR, "scoring_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[score] Saved results → {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" SMS Spam Classifier — Scoring Demo")
    print("=" * 70)

    _ensure_dirs()

    # Load model and vectoriser
    model, vectoriser = load_model_and_vectoriser()

    # Check if a message was passed via command line
    if len(sys.argv) > 1:
        # Score the command-line message
        user_message = " ".join(sys.argv[1:])
        print(f"  Scoring user message: \"{user_message}\"\n")
        result = score_message(user_message, model, vectoriser)
        print_results([result])
    else:
        # Score the hardcoded example messages
        print(f"  No message provided — scoring {len(EXAMPLE_MESSAGES)} examples\n")
        results = batch_score(EXAMPLE_MESSAGES, model, vectoriser)
        print_results(results)
        save_scores(results)


if __name__ == "__main__":
    main()
