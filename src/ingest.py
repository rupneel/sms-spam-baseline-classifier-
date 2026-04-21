import os
import sys
import urllib.request
import pandas as pd

# CONFIGURATION — paths are relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
CLEANED_DIR  = os.path.join(PROJECT_ROOT, "data", "cleaned")
RAW_FILE     = os.path.join(RAW_DIR, "spam.csv")
CLEAN_FILE   = os.path.join(CLEANED_DIR, "sms_clean.csv")

# Kaggle-hosted mirror of the UCI SMS Spam Collection
DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
)
#Download the dataset (if not already present)
def download_dataset():
    if os.path.exists(RAW_FILE):
        print(f"Raw file already exists: {RAW_FILE}")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading dataset → {RAW_FILE}")
    urllib.request.urlretrieve(DOWNLOAD_URL, RAW_FILE)
    print(f"Download completed")

# Load data (simplified for GitHub TSV)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )
    return df

#  STEP 3 — Detect and remove duplicate messages
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["message"], keep="first")
    df = df.reset_index(drop=True)
    after = len(df)
    removed = before - after
    print(f"Duplicates removed: {removed}  "
          f"({before} → {after} rows)")
    return df

# STEP 4 — Validate the cleaned data
def validate(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["label", "message"], (
        f"Unexpected columns: {list(df.columns)}"
    )

    # Check label values are only 'ham' and 'spam'
    unique_labels = set(df["label"].unique())
    assert unique_labels == {"ham", "spam"}, (
        f"Unexpected label values: {unique_labels}"
    )

    # Check no nulls remain
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values after cleaning"

    # Check we have a reasonable number of rows (at least 4,000)
    assert len(df) >= 4000, f"Too few rows after cleaning: {len(df)}"

    print(f"Validation passed ✓  ({len(df)} rows, "
          f"{df['label'].value_counts().to_dict()})")

# STEP 5 — Save the cleaned dataset
def save_clean_data(df: pd.DataFrame) -> None:
    """Saves the cleaned DataFrame to data/cleaned/sms_clean.csv."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Saved cleaned data → {CLEAN_FILE}")

# MAIN — run the full ingestion pipeline
def main():
    print("=" * 80)
    print(" SMS Spam Classifier — Data Ingestion")
    print("=" * 80)

    # 1. Download
    download_dataset()

    # 2. Load
    print("-" * 80)
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # 3. Remove duplicates
    print("-" * 80)
    df = remove_duplicates(df)

    # 4. Validate
    print("-" * 80)
    validate(df)

    # 5. Save
    print("-" * 80)
    save_clean_data(df)

    # Ending
    print("=" * 80)
    print("Ingestion process completed!")
    print("=" * 80)

    return df

if __name__ == "__main__":
    main()