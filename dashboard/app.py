import streamlit as st
import os
import re
import joblib
import json
import pandas as pd
from PIL import Image

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH     = os.path.join(PROJECT_ROOT, "data", "processed", "tfidf_vectoriser.joblib")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "outputs", "models", "naive_bayes.joblib")
REPORT_PATH  = os.path.join(PROJECT_ROOT, "outputs", "reports", "evaluation_report.json")
MATRIX_PATH  = os.path.join(PROJECT_ROOT, "outputs", "figures", "confusion_matrix.png")

# --- UI SETUP ---
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .spam { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .ham { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    vectoriser = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    return vectoriser, model

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings & Metrics")
    st.markdown("---")
    
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            metrics = json.load(f)
        
        st.metric("Model Accuracy", f"{metrics['accuracy']}%")
        
        cm = metrics['confusion_matrix']
        st.write("**Confusion Matrix Counts:**")
        cols = st.columns(2)
        cols[0].write(f"TN: {cm['TN']}")
        cols[1].write(f"FP: {cm['FP']}")
        cols[0].write(f"FN: {cm['FN']}")
        cols[1].write(f"TP: {cm['TP']}")
    
    st.markdown("---")
    st.write("**About:**")
    st.write("This tool uses a Multinomial Naive Bayes model trained on a TF-IDF vectorised SMS dataset.")

# --- MAIN CONTENT ---
st.title("📧 SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's Spam or Ham.")

vec, model = load_model()

if vec is None or model is None:
    st.error("Error: Model artifacts not found. Please run the training pipeline first.")
else:
    # Text Input
    user_input = st.text_area("SMS Message:", placeholder="Type or paste your message here...", height=150)
    
    if st.button("Classify Message"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            # Inference
            cleaned = clean_text(user_input)
            features = vec.transform([cleaned])
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            
            label = "SPAM" if pred == 1 else "HAM"
            conf = probs[pred] * 100
            
            # Display Result
            css_class = "spam" if label == "SPAM" else "ham"
            st.markdown(f'<div class="prediction-box {css_class}">Result: {label}</div>', unsafe_allow_html=True)
            
            st.write(f"**Confidence:** {conf:.2f}%")
            st.progress(conf / 100)
            
            # Probability breakdown
            st.write("---")
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame({
                "Label": ["Ham", "Spam"],
                "Probability": [probs[0], probs[1]]
            })
            st.bar_chart(prob_df.set_index("Label"))

    # Extra Visuals
    with st.expander("Show Detailed Evaluation"):
        if os.path.exists(MATRIX_PATH):
            img = Image.open(MATRIX_PATH)
            st.image(img, caption="Confusion Matrix (from Training)", use_container_width=True)
        else:
            st.info("Confusion matrix visualization not found.")
