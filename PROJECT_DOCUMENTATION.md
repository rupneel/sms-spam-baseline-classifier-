# 📧 SMS SPAM CLASSIFIER - MASTER PROJECT DOCUMENTATION

## 1. PROJECT TITLE AND DESCRIPTION
**Project Name:** Automated SMS Spam Classification System
**Author:** Project Development Team
**Date:** April 2026
**Purpose:** This project implements an advanced machine learning pipeline to classify short text messages (SMS) into 'Ham' (legitimate) or 'Spam'. It serves as a production-grade example of Natural Language Processing (NLP) integration with a modern web frontend.

---

## 2. COMPREHENSIVE TECH STACK (WHAT WE USED)

### 2.1 Core Programming Environment
- **Python 3.10+**: The primary development language, chosen for its vast ecosystem of data science libraries.
- **Virtual Environments (.venv)**: Used to isolate dependencies and ensure reproducibility across different machines.

### 2.2 Data Processing & Analysis
- **Pandas**: Used for high-performance data manipulation, reading CSV/TSV files, and managing the DataFrame structures.
- **NumPy**: Provides support for large, multi-dimensional arrays and matrices.
- **SciPy**: Specifically utilized for storing 'Sparse Matrices' (NPZ format), which efficiently store the TF-IDF feature space without wasting memory on zeros.

### 2.3 Machine Learning & NLP
- **Scikit-Learn (sklearn)**: The backbone of the ML pipeline.
    - `TfidfVectorizer`: For text-to-numerical conversion.
    - `MultinomialNB`: The core classifier algorithm.
    - `train_test_split`: For unbiased model evaluation.
    - `metrics`: For calculating accuracy, confusion matrices, and precision/recall scores.
- **Joblib**: Used for 'Model Serialization'—saving the trained weights to disk so they can be reloaded in the dashboard instantly without retraining.

### 2.4 Visualization & UI
- **Matplotlib**: The foundational plotting library for generating distributions and histograms.
- **Seaborn**: A high-level visualization library used to create professional-grade heatmaps for model evaluation.
- **Streamlit**: A powerful framework used to build the web interface. It allows for fast, Python-based web development without needing HTML/CSS/JS expertise.

---

## 3. FULL PROJECT WORKFLOW (EVERYTHING WE DID)

### 3.1 Data Acquisition and Ingestion
The project began with the raw 'SMS Spam Collection' dataset.
1. **Script:** `src/ingest.py`
2. **Process:**
   - Loading the raw text file using custom delimiters.
   - Dropping unnecessary columns.
   - Removing duplicate entries to prevent the model from overfitting on repeated spam messages.
   - Exporting the cleaned baseline to `data/cleaned/sms_clean.csv`.

### 3.2 Advanced Natural Language Preprocessing
This is the most critical phase where raw text is transformed into a format a machine can understand.
1. **Script:** `src/preprocess.py`
2. **Logic Applied:**
   - **Lowercasing:** Standardizing text to avoid duplicate entries for "OFFER" vs "offer".
   - **URL Removal:** Removing strings starting with `http` or `www` as they vary wildly and don't help in general classification.
   - **Digit Stripping:** Removing numbers, as spam often contains randomized phone numbers or codes that don't help the model learn general patterns.
   - **Special Character Filtering:** Removing non-alphabetic characters using Regex `[^a-z\s]`.
3. **TF-IDF Vectorisation Strategy:**
   - We used **N-gram Range (1, 2)**, meaning the model looks at single words ("cash") and double words ("win cash").
   - We enabled **Sublinear TF**, which squashes the influence of words that appear too many times (using $1 + \log(\text{tf})$).
   - We removed **English Stop Words**, filtering out words like "the", "a", "is" which carry no specific spam/ham meaning.

### 3.3 Model Development and Training
1. **Script:** `src/train.py`
2. **The Classifier:** We chose **Multinomial Naive Bayes**.
   - **Why?** It is probabilistic, extremely fast, and highly effective for text data where the frequency of words is the most important feature.
3. **Training Process:**
   - Loaded the processed sparse matrices (`X_train.npz`).
   - Fitted the model on the 80% training set.
   - Saved the model to `outputs/models/naive_bayes.joblib`.

### 3.4 Rigorous Evaluation
To ensure the model works in the real world, we didn't just look at accuracy.
1. **Confusion Matrix Analysis:**
   - **True Positives (TP):** Spam correctly identified.
   - **True Negatives (TN):** Ham correctly identified.
   - **False Positives (FP):** Ham wrongly called Spam (The "Critical Error").
   - **False Negatives (FN):** Spam that slipped through (The "Annoyance Error").
2. **Visual Reports:** Generated `confusion_matrix.png` to visually see how many messages fell into each category.

### 3.5 Real-Time Inference System
1. **Script:** `src/predict.py`
2. **Functionality:** Built a modular system that can take any string, apply the stored preprocessing logic, and output a prediction instantly.

### 3.6 Deployment (The Web Dashboard)
1. **Folder:** `dashboard/app.py`
2. **Features:**
   - **Interactive Text Area:** Users can paste messages.
   - **Probabilistic Output:** The model doesn't just say "Spam"; it shows the confidence percentage (e.g., "99.8% Spam").
   - **Persistence:** The app loads the `tfidf_vectoriser.joblib` and `naive_bayes.joblib` once at startup and caches them for maximum speed.

---

## 4. DETAILED LINE-BY-LINE LOGIC SUMMARY

### Data Processing Logic:
- **`apply_cleaning(df)`**: Iterates through every message. It acts as a filter, ensuring only clean, semantic words reach the model.
- **`split_data(df)`**: Uses 'Stratification'. This is vital because if the training set randomly gets 0% spam, the model will never learn what spam looks like. Stratification keeps the 13% spam ratio constant in both sets.

### Mathematics of Prediction:
- **TF (Term Frequency)**: Measures how often a word is in a single text.
- **IDF (Inverse Document Frequency)**: Measures how 'unique' a word is. If the word "Free" appears in 500 spam messages but 0 ham messages, its IDF score becomes very high, signaling a strong spam indicator.
- **Log Probabilities**: The model uses the log of probabilities to avoid "Floating Point Underflow" (where numbers become too small for a computer to handle).

---

## 5. RECENT REFINEMENTS AND UPDATES
During the final stages of the project, we:
- **Cleaned the Console Output**: Removed noisy log messages to make the scripts more professional.
- **Added Dynamic Dir Creation**: The code now automatically creates `/data`, `/outputs`, and `/models` folders if they are missing.
- **Optimized Matplotlib**: Switched to the "Agg" backend to allow the script to run on remote servers without a monitor/GUI requirement.
- **Detailed Reporting**: Added `preprocessing_report.json` and `evaluation_report.json` to keep track of versioning and performance metrics over time.

---

## 6. CONCLUSION
This project successfully demonstrates the transition from raw, unstructured data to a polished, interactive web application. By combining the statistical power of Naive Bayes with a modern UI like Streamlit, we have created a tool that is not only accurate but accessible to non-technical users.

---
*End of Documentation — SMS Spam Classifier Project v1.0*
