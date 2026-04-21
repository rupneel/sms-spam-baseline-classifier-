# 📧 SMS Spam Baseline Classifier

A complete machine learning pipeline and interactive dashboard for classifying SMS messages as **Spam** or **Ham** (not spam).

## 🚀 Overview
This project provides an end-to-end solution for SMS spam detection using a **Multinomial Naive Bayes** classifier. It includes data ingestion, automated cleaning, TF-IDF vectorisation, model training, and a web-based dashboard for real-time inference.

## 🏗️ Project Structure
- `src/`: Core Python scripts for the ML pipeline.
  - `ingest.py`: Data downloading and ingestion.
  - `preprocess.py`: Text cleaning and TF-IDF vectorisation.
  - `train.py`: Model training and evaluation.
  - `predict.py`: Standalone command-line inference.
- `dashboard/`: Streamlit web application.
- `data/`: Raw and processed data storage (ignored by git).
- `outputs/`: Model artifacts, evaluation reports, and figures.
- `docs/`: Detailed design documentation (HLD/LLD).

## 🛠️ Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sms-spam-classifier
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Running the Pipeline
Run the scripts in the following order to build the model from scratch:
1. **Ingest Data**: `python src/ingest.py`
2. **Preprocess & Vectorise**: `python src/preprocess.py`
3. **Train Model**: `python src/train.py`

## 🖥️ Using the Dashboard
Launch the interactive web interface:
```bash
streamlit run dashboard/app.py
```

## 🔍 Manual Inference
Test the model via command line:
```bash
python src/predict.py "Congratulations! You've won a $1000 gift card."
```

## 📊 Performance
The model achieves **~96.9% accuracy** on the test set, with a focus on high precision for the spam class to avoid misclassifying important messages (ham).
