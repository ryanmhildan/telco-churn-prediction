"""Single-customer prediction used by the Streamlit app (and unit tests).

Loads the saved pipeline and the cost-based threshold once, then turns a dict of raw
customer fields into: churn probability, the contact decision, the expected value of a
retention offer, and the top SHAP drivers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd

from src.data_cleaning import clean_total_charges
from src.explain import ChurnExplainer
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.threshold import THRESHOLD_PATH
from src.train import MODEL_PATH

ALLOWED_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


@dataclass
class Prediction:
    proba: float
    threshold: float
    contact: bool
    expected_value_idr: float
    drivers: pd.DataFrame


def to_frame(customer: dict) -> pd.DataFrame:
    """Validate a raw customer dict and return a one-row frame in training format."""
    missing = [f for f in FEATURES if f not in customer]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for col, allowed in ALLOWED_VALUES.items():
        if customer[col] not in allowed:
            raise ValueError(f"{col}={customer[col]!r} not in {allowed}")
    if float(customer["tenure"]) < 0 or float(customer["MonthlyCharges"]) < 0:
        raise ValueError("tenure and MonthlyCharges must be non-negative")

    row = {f: customer[f] for f in FEATURES}
    # Keep dependent fields consistent with their parent service.
    if row["InternetService"] == "No":
        row.update({c: "No internet service" for c in INTERNET_ADDONS})
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"

    df = pd.DataFrame([row])
    df["TotalCharges"] = df["TotalCharges"].astype(str).replace({"None": "", "nan": ""})
    df = clean_total_charges(df)  # same rule as training: blank + tenure 0 -> 0
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    decision = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return model, ChurnExplainer(model), decision


def predict(customer: dict, top_k: int = 5) -> Prediction:
    model, explainer, decision = load_artifacts()
    X = to_frame(customer)
    proba = float(model.predict_proba(X)[:, 1][0])
    a = decision["assumptions"]
    expected_value = proba * a["retention_success_rate"] * a["churn_loss_idr"] - a["retention_cost_idr"]
    return Prediction(
        proba=proba,
        threshold=float(decision["threshold"]),
        contact=proba >= decision["threshold"],
        expected_value_idr=float(expected_value),
        drivers=explainer.explain_row(X, top_k=top_k),
    )
