"""Deterministic cleaning of the raw Telco CSV.

Only row-wise rules that do not learn anything from the data live here, so running
this before the train/test split cannot leak information. Anything that is fitted
(imputation statistics, scaling, encoding) belongs in the Phase 2 pipeline instead.

Usage:
    python -m src.data_cleaning
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CLEAN_CSV, ID_COL, RAW_CSV, TARGET_COL

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV with every column as text, so blanks are not silently guessed."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to float.

    In the raw data, TotalCharges is blank only for customers with tenure == 0: they
    have not been billed yet, so their true total is 0. Blanks for tenure > 0 would be
    genuinely unknown and are left as NaN for the pipeline imputer.
    """
    out = df.copy()
    total = pd.to_numeric(out["TotalCharges"].str.strip().replace("", None), errors="coerce")
    tenure = pd.to_numeric(out["tenure"], errors="coerce")
    total = total.mask(total.isna() & (tenure == 0), 0.0)
    out["TotalCharges"] = total.astype(float)
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of the raw frame with the target encoded as 0/1."""
    out = clean_total_charges(df)
    out[ID_COL] = out[ID_COL].str.strip()
    out["tenure"] = pd.to_numeric(out["tenure"]).astype(int)
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"]).astype(float)
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"]).astype(int)

    text_cols = [c for c in out.columns if c not in NUMERIC_COLS + ["SeniorCitizen", ID_COL]]
    for col in text_cols:
        out[col] = out[col].str.strip()

    unknown_target = set(out[TARGET_COL].unique()) - {"Yes", "No"}
    if unknown_target:
        raise ValueError(f"unexpected target values: {unknown_target}")
    out[TARGET_COL] = (out[TARGET_COL] == "Yes").astype(int)
    return out


def main() -> None:
    clean = clean_data(load_raw())
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False)
    print(f"saved {CLEAN_CSV} {clean.shape}; NaN left: {int(clean.isna().sum().sum())}")


if __name__ == "__main__":
    main()
