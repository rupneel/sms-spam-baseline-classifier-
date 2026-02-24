# SMS Spam Baseline Classifier

A complete data-science pipeline that classifies SMS messages as **spam** or
**ham** (legitimate) using **Naive Bayes** on the UCI SMS Spam Collection.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run each script individually from src/
python src/ingest.py
python src/quality.py
```

## Project Structure

```
sms-spam-classifier/
├── data/raw/          ← original downloaded data
├── data/cleaned/      ← cleaned dataset
├── src/               ← Python pipeline scripts
├── outputs/
│   ├── reports/       ← quality report (HTML)
│   └── figures/       ← plots (PNG)
├── docs/              ← data dictionary, problem framing
├── requirements.txt
└── README.md
```

## Completed Steps

| Script          | What it does                                      |
|-----------------|---------------------------------------------------|
| `ingest.py`     | Download, clean, validate, save dataset            |
| `quality.py`    | Data quality report (missingness, duplicates, etc) |

## Dataset

**SMS Spam Collection** from the UCI Machine Learning Repository.
~5,574 messages (87% ham, 13% spam), English only, collected 2009–2012.
