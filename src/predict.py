"""
predict.py — Standalone Inference Script
========================================
Loads the trained model and TF-IDF vectoriser to predict if a 
given SMS message is 'ham' or 'spam'.

Usage:
    python src/predict.py "Your message here"
"""

import os
import re
import sys
import joblib

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH     = os.path.join(PROJECT_ROOT, "data", "processed", "tfidf_vectoriser.joblib")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "outputs", "models", "naive_bayes.joblib")

def clean_text(text: str) -> str:
    """Normalise text using the same logic as preprocess.py."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_artefacts():
    """Load the vectoriser and model from disk."""
    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
        print("[predict] ERROR: Model artifacts not found. Run preprocess.py and train.py first.")
        sys.exit(1)
    
    vectoriser = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    return vectoriser, model

def predict(message: str, vectoriser, model):
    """Clean, vectorise, and predict for a single message."""
    cleaned = clean_text(message)
    features = vectoriser.transform([cleaned])
    
    prediction = model.predict(features)[0]
    # Get probability/confidence
    prob = model.predict_proba(features)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = prob[prediction] * 100
    
    return label, confidence

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python src/predict.py \"<message_text>\"")
        print("Example: python src/predict.py \"Congratulations! You've won a $1000 gift card. Call now!\"")
        sys.exit(0)

    msg = sys.argv[1]
    vectoriser, model = load_artefacts()
    
    label, confidence = predict(msg, vectoriser, model)
    
    print("\n" + "="*40)
    print(f" INPUT: {msg}")
    print("-"*40)
    print(f" RESULT: {label}")
    print(f" CONFIDENCE: {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
