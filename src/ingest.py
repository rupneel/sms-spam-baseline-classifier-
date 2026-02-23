"""
ingest.py — Data Ingestion & Validation
========================================
Downloads (or loads) the SMS Spam Collection dataset, validates the schema,
detects duplicates, and saves a cleaned version.

Key concepts you'll learn here:
  • Reading CSVs with encoding issues (latin-1)
  • Dropping junk columns that appear in messy files
  • Detecting and removing duplicate rows
  • Standardising column names for downstream code
"""

import os
import sys
import urllib.request
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION — paths are relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
CLEANED_DIR  = os.path.join(PROJECT_ROOT, "data", "cleaned")
RAW_FILE     = os.path.join(RAW_DIR, "spam.csv")
CLEAN_FILE   = os.path.join(CLEANED_DIR, "sms_clean.csv")

# Kaggle-hosted mirror of the UCI SMS Spam Collection
DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
)

# ---------------------------------------------------------------------------
# STEP 1 — Download the dataset (if not already present)
# ---------------------------------------------------------------------------
def download_dataset():
    """
    Downloads the SMS Spam Collection into data/raw/.
    Uses a GitHub-hosted TSV mirror so no Kaggle API key is needed.
    """
    if os.path.exists(RAW_FILE):
        print(f"[ingest] Raw file already exists: {RAW_FILE}")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[ingest] Downloading dataset → {RAW_FILE}")
    urllib.request.urlretrieve(DOWNLOAD_URL, RAW_FILE)
    print(f"[ingest] Download complete ({os.path.getsize(RAW_FILE):,} bytes)")


# ---------------------------------------------------------------------------
# STEP 2 — Load and fix encoding / column issues
# ---------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    """
    Reads the raw CSV/TSV with latin-1 encoding and cleans up the columns.

    Why latin-1?
      The original file contains special characters (£, €, accented letters)
      that break the default UTF-8 decoder. Latin-1 never raises a decode
      error because every byte 0x00–0xFF maps to a valid character.

    Returns
    -------
    pd.DataFrame with columns: ['label', 'message']
    """
    # Try tab-separated first (the GitHub mirror is TSV)
    try:
        df = pd.read_csv(RAW_FILE, sep="\t", header=None,
                         names=["label", "message"], encoding="latin-1")
    except Exception:
        # Fallback: comma-separated Kaggle format (has extra junk columns)
        df = pd.read_csv(RAW_FILE, encoding="latin-1")

    # --- Drop unnamed / junk columns -------------------------------------------
    # The Kaggle version has columns: v1, v2, Unnamed: 2, Unnamed: 3, Unnamed: 4
    # We only need the first two.
    junk_cols = [c for c in df.columns if "unnamed" in str(c).lower()]
    if junk_cols:
        print(f"[ingest] Dropping {len(junk_cols)} junk column(s): {junk_cols}")
        df = df.drop(columns=junk_cols)

    # --- Standardise column names ----------------------------------------------
    # Different sources name the columns v1/v2 or label/message.
    # We rename to a consistent ['label', 'message'].
    if list(df.columns[:2]) != ["label", "message"]:
        df.columns = ["label", "message"] + list(df.columns[2:])
        # Keep only the two columns we need
        df = df[["label", "message"]]
        print("[ingest] Standardised column names → ['label', 'message']")

    return df


# ---------------------------------------------------------------------------
# STEP 3 — Detect and remove duplicate messages
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact-duplicate rows.

    Why remove duplicates?
      If the same message appears in both train and test sets, the model gets
      an unfair advantage (data leakage). Removing duplicates also gives us
      a more honest class-balance picture.

    Returns
    -------
    pd.DataFrame  — de-duplicated, index reset
    """
    before = len(df)
    df = df.drop_duplicates(subset=["message"], keep="first")
    df = df.reset_index(drop=True)
    after = len(df)

    removed = before - after
    print(f"[ingest] Duplicates removed: {removed}  "
          f"({before} → {after} rows)")

    return df


# ---------------------------------------------------------------------------
# STEP 4 — Validate the cleaned data
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """
    Runs basic sanity checks on the cleaned DataFrame.
    Raises an AssertionError if anything looks wrong.
    """
    # Check we have exactly two columns
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

    print(f"[ingest] Validation passed ✓  ({len(df)} rows, "
          f"{df['label'].value_counts().to_dict()})")


# ---------------------------------------------------------------------------
# STEP 5 — Save the cleaned dataset
# ---------------------------------------------------------------------------
def save_clean_data(df: pd.DataFrame) -> None:
    """Saves the cleaned DataFrame to data/cleaned/sms_clean.csv."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"[ingest] Saved cleaned data → {CLEAN_FILE}")


# ---------------------------------------------------------------------------
# MAIN — run the full ingestion pipeline
# ---------------------------------------------------------------------------
def main():
    """
    Orchestrates the entire ingestion pipeline:
      download → load → clean → validate → save
    """
    print("=" * 60)
    print(" SMS Spam Classifier — Data Ingestion")
    print("=" * 60)

    # 1. Download
    download_dataset()

    # 2. Load and fix columns
    df = load_raw_data()
    print(f"[ingest] Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"[ingest] First 3 rows:\n{df.head(3).to_string()}\n")

    # 3. Remove duplicates
    df = remove_duplicates(df)

    # 4. Validate
    validate(df)

    # 5. Save
    save_clean_data(df)

    print("=" * 60)
    print(" Ingestion complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
