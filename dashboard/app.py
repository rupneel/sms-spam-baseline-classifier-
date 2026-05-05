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
    /* Core Theme Overrides */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a1c23;
        border-right: 1px solid #2d2e3a;
    }
    
    /* Fix for Sidebar Toggle: We hide the toolbar but keep the header for the collapse button */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    
    /* Hide Deploy Button and Footer */
    .stAppDeployButton { display: none; }
    footer { visibility: hidden; }
    
    /* Main Content Styling */
    .main {
        background-color: #0e1117;
        padding-top: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
        background: linear-gradient(135deg, #0088ff 0%, #0066cc 100%);
    }
    
    /* Prediction Boxes */
    .prediction-box {
        padding: 30px;
        border-radius: 12px;
        margin: 25px 0;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .spam { 
        background: linear-gradient(135deg, #4d1d24 0%, #2a0a0d 100%);
        color: #ff4d4d;
        border: 1px solid #721c24;
    }
    .ham { 
        background: linear-gradient(135deg, #1d4d2b 0%, #0a2a12 100%);
        color: #4dff88;
        border: 1px solid #155724;
    }
    
    /* Input Area */
    .stTextArea textarea {
        background-color: #1e2128 !important;
        color: #ffffff !important;
        border: 1px solid #3d414a !important;
        border-radius: 10px !important;
    }
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
