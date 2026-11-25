"""Unit tests for cleaning, preprocessing and single-customer prediction.

The small frames below are hand-written test fixtures that exercise edge cases; they are
not used for any analysis.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_CSV, TARGET_COL
from src.data_cleaning import clean_data, clean_total_charges
from src.preprocessing import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, build_preprocessor,
                               collapse_no_service, make_split)
from src.train import MODEL_PATH


def raw_row(**overrides) -> dict:
    """One raw customer row, all values as text like the raw CSV."""
    row = {
        "customerID": "0000-TEST", "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes",
        "Dependents": "No", "tenure": "5", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": "80.5", "TotalCharges": "402.5", "Churn": "Yes",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = [
        raw_row(),
        raw_row(customerID="0001-TEST", tenure="0", TotalCharges=" ", Churn="No"),
        raw_row(customerID="0002-TEST", InternetService="No", OnlineSecurity="No internet service",
                OnlineBackup="No internet service", DeviceProtection="No internet service",
                TechSupport="No internet service", StreamingTV="No internet service",
                StreamingMovies="No internet service", MonthlyCharges="20.0", TotalCharges="400.0",
                tenure="20", Contract="Two year", Churn="No"),
        raw_row(customerID="0003-TEST", PhoneService="No", MultipleLines="No phone service",
                SeniorCitizen="1", Contract="One year", PaymentMethod="Credit card (automatic)"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cleaning
def test_blank_total_charges_with_zero_tenure_becomes_zero(raw_frame):
    out = clean_total_charges(raw_frame)
    assert out.loc[1, "TotalCharges"] == 0.0
    assert out["TotalCharges"].dtype == float


def test_blank_total_charges_with_positive_tenure_stays_missing():
    df = pd.DataFrame([raw_row(tenure="10", TotalCharges="  ")])
    out = clean_total_charges(df)
    assert np.isnan(out.loc[0, "TotalCharges"])


def test_clean_data_types_and_target(raw_frame):
    out = clean_data(raw_frame)
    assert out[TARGET_COL].tolist() == [1, 0, 0, 1]
    assert out["tenure"].dtype.kind == "i"
    assert out["SeniorCitizen"].dtype.kind == "i"
    assert out["MonthlyCharges"].dtype == float
    assert not out.isna().any().any()


def test_clean_data_does_not_modify_input(raw_frame):
    before = raw_frame.copy()
    clean_data(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)


def test_clean_data_rejects_unknown_target(raw_frame):
    raw_frame.loc[0, TARGET_COL] = "Maybe"
    with pytest.raises(ValueError, match="unexpected target"):
        clean_data(raw_frame)


# ---------------------------------------------------------------- preprocessing
def test_collapse_no_service_maps_to_no(raw_frame):
    out = collapse_no_service(raw_frame)
    assert (out.loc[2, ["OnlineSecurity", "TechSupport", "StreamingMovies"]] == "No").all()
    assert out.loc[3, "MultipleLines"] == "No"
    assert raw_frame.loc[2, "OnlineSecurity"] == "No internet service"  # input untouched


def test_preprocessor_output_shape_names_and_no_nan(raw_frame):
    X = clean_data(raw_frame)[FEATURES]
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()
    assert Xt.shape == (len(X), len(names))
    assert not np.isnan(Xt).any()
    assert not any("customerID" in n for n in names)
    assert not any("No internet service" in n or "No phone service" in n for n in names)
    assert sum(n.startswith("num__") for n in names) == len(NUMERIC_FEATURES)


def test_preprocessor_scaler_is_fitted_on_training_data_only(raw_frame):
    X = clean_data(raw_frame)[FEATURES]
    pre = build_preprocessor().fit(X.iloc[:2])
    scaler = pre.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_, X.iloc[:2][NUMERIC_FEATURES].mean().values)


def test_preprocessor_handles_unknown_category_and_missing_number(raw_frame):
    X = clean_data(raw_frame)[FEATURES]
    pre = build_preprocessor().fit(X)
    odd = X.iloc[[0]].copy()
    odd["Contract"] = "Three year"
    odd["TotalCharges"] = np.nan
    row = pre.transform(odd)
    names = list(pre.get_feature_names_out())
    contract = [i for i, n in enumerate(names) if n.startswith("cat__Contract")]
    assert row[0, contract].sum() == 0
    assert not np.isnan(row).any()


def test_feature_lists_cover_all_model_inputs():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert len(FEATURES) == 19


# ---------------------------------------------------------------- split
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw data not downloaded")
def test_split_is_stratified_deterministic_and_disjoint():
    from src.data_cleaning import load_raw

    df = clean_data(load_raw())
    train_a, test_a = make_split(df)
    train_b, test_b = make_split(df)
    assert train_a["customerID"].equals(train_b["customerID"])
    assert set(train_a["customerID"]).isdisjoint(test_a["customerID"])
    assert len(train_a) + len(test_a) == len(df)
    assert abs(train_a[TARGET_COL].mean() - test_a[TARGET_COL].mean()) < 0.005


# ---------------------------------------------------------------- prediction
def customer(**overrides) -> dict:
    c = {k: v for k, v in raw_row().items() if k in FEATURES}
    c.update(SeniorCitizen=0, tenure=5, MonthlyCharges=80.5, TotalCharges=402.5)
    c.update(overrides)
    return c


def test_to_frame_rejects_invalid_values():
    from src.predict import to_frame

    with pytest.raises(ValueError):
        to_frame(customer(Contract="Lifetime"))
    with pytest.raises(ValueError):
        to_frame(customer(tenure=-1))
    bad = customer()
    bad.pop("Contract")
    with pytest.raises(ValueError, match="missing"):
        to_frame(bad)


def test_to_frame_keeps_dependent_fields_consistent():
    from src.predict import to_frame

    X = to_frame(customer(InternetService="No", OnlineSecurity="Yes", PhoneService="No", MultipleLines="Yes",
                          tenure=0, TotalCharges=None))
    assert X.loc[0, "OnlineSecurity"] == "No internet service"
    assert X.loc[0, "MultipleLines"] == "No phone service"
    assert X.loc[0, "TotalCharges"] == 0.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_predict_returns_probability_and_drivers():
    from src.predict import predict

    result = predict(customer(), top_k=3)
    assert 0.0 <= result.proba <= 1.0
    assert result.contact == (result.proba >= result.threshold)
    assert len(result.drivers) == 3
    assert set(result.drivers["feature"]) <= set(FEATURES)
