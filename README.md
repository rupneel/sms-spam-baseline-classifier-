# SMS Spam Baseline Classifier

A complete data-science pipeline that classifies SMS messages as **spam** or
**ham** (legitimate) using **Naive Bayes** on the UCI SMS Spam Collection.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python run_all.py
```

## Project Structure

```
sms-spam-classifier/
├── data/raw/          ← original downloaded data
├── data/cleaned/      ← cleaned dataset
├── src/               ← Python pipeline scripts
├── outputs/
│   ├── reports/       ← quality & label reports (HTML)
│   ├── models/        ← saved model + vectorizer (.pkl)
│   ├── scores/        ← batch scoring outputs
│   └── figures/       ← plots (PNG)
├── dashboard/         ← interactive HTML dashboard
├── presentation/      ← slide deck (HTML)
├── docs/              ← data dictionary, executive summary
├── requirements.txt
├── run_all.py         ← runs everything in order
└── README.md
```

## Pipeline Steps

| Script          | What it does                                      |
|-----------------|---------------------------------------------------|
| `ingest.py`     | Download, clean, validate, save dataset            |
| `quality.py`    | Data quality report (missingness, duplicates, etc) |
| `labels.py`     | Label sanity checks and class balance              |
| `preprocess.py` | Text normalisation + TF-IDF vectorisation          |
| `model.py`      | Train Naive Bayes + majority-class baseline        |
| `evaluate.py`   | Metrics, confusion matrix, error analysis          |
| `threshold.py`  | Precision-recall trade-off analysis                |
| `score.py`      | Batch scoring with saved model                     |

## Dataset

**SMS Spam Collection** from the UCI Machine Learning Repository.
~5,574 messages (87% ham, 13% spam), English only, collected 2009–2012.
