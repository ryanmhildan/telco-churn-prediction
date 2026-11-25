"""Download the IBM Telco Customer Churn dataset into data/raw/.

Sources are tried in order and the first one that passes validation wins:
  1. Hugging Face  scikit-learn/churn-prediction   (datasets library)
  2. OpenML        telco-customer-churn, version 1 (sklearn.datasets.fetch_openml)
  3. Kaggle        blastchar/telco-customer-churn  (needs ~/.kaggle/kaggle.json)

No synthetic fallback exists on purpose: if every source fails, the script exits
with instructions to download the CSV manually.

Usage:
    python -m src.download_data            # skip if the raw CSV already exists
    python -m src.download_data --force    # re-download and overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import EXPECTED_COLUMNS, EXPECTED_ROWS, RAW_CSV, RAW_METADATA

EXPECTED_COLUMN_NAMES = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

HF_REPO = "scikit-learn/churn-prediction"
OPENML_NAME = "telco-customer-churn"
OPENML_VERSION = 1
KAGGLE_REF = "blastchar/telco-customer-churn"


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError unless df looks like the original IBM Telco dataset."""
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"unexpected shape {df.shape}, expected ({EXPECTED_ROWS}, {EXPECTED_COLUMNS})")
    if list(df.columns) != EXPECTED_COLUMN_NAMES:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if df["customerID"].duplicated().any():
        raise ValueError("duplicate customerID values")


def from_huggingface() -> tuple[pd.DataFrame, dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(HF_REPO)
    ds = load_dataset(HF_REPO, split="train", revision=info.sha)
    df = ds.to_pandas()
    license_tag = (info.card_data or {}).get("license", "unknown")
    return df, {
        "source": "huggingface",
        "identifier": HF_REPO,
        "url": f"https://huggingface.co/datasets/{HF_REPO}",
        "revision": info.sha,
        "license": license_tag,
    }


def from_openml() -> tuple[pd.DataFrame, dict]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name=OPENML_NAME, version=OPENML_VERSION, as_frame=True, parser="auto")
    df = bunch.frame
    return df, {
        "source": "openml",
        "identifier": f"{OPENML_NAME} v{OPENML_VERSION} (data_id={bunch.details.get('id')})",
        "url": bunch.url,
        "revision": bunch.details.get("version"),
        "license": bunch.details.get("licence", "unknown"),
    }


def from_kaggle() -> tuple[pd.DataFrame, dict]:
    from kaggle.api.kaggle_api_extended import KaggleApi  # raises if not installed

    api = KaggleApi()
    api.authenticate()  # raises if ~/.kaggle/kaggle.json is missing
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_download_files(KAGGLE_REF, path=tmp, unzip=True)
        csv_path = next(Path(tmp).glob("*.csv"))
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return df, {
        "source": "kaggle",
        "identifier": KAGGLE_REF,
        "url": f"https://www.kaggle.com/datasets/{KAGGLE_REF}",
        "revision": None,
        "license": "see Kaggle page",
    }


SOURCES = [("Hugging Face", from_huggingface), ("OpenML", from_openml), ("Kaggle", from_kaggle)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(force: bool = False) -> Path:
    if RAW_CSV.exists() and not force:
        print(f"{RAW_CSV} already exists; use --force to re-download.")
        return RAW_CSV

    errors = []
    for name, loader in SOURCES:
        print(f"Trying {name} ...")
        try:
            df, meta = loader()
            validate(df)
        except Exception as exc:  # any failure means: try the next source
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  failed -> {errors[-1]}")
            continue

        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        meta.update(
            downloaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            file=RAW_CSV.name,
            sha256=sha256(RAW_CSV),
        )
        RAW_METADATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  ok -> saved {RAW_CSV} ({df.shape[0]} rows x {df.shape[1]} columns)")
        return RAW_CSV

    sys.exit(
        "All sources failed:\n  " + "\n  ".join(errors) +
        f"\n\nDownload WA_Fn-UseC_-Telco-Customer-Churn.csv manually from "
        f"https://www.kaggle.com/datasets/{KAGGLE_REF} and save it as {RAW_CSV}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the CSV exists")
    download(force=parser.parse_args().force)
