"""Train/test split and the scikit-learn preprocessing pipeline.

The preprocessor is never fitted on its own in later phases: it is the first step of
each model Pipeline, so inside cross-validation it is re-fitted on the training folds
only. That is what keeps imputation, scaling and encoding free of data leakage.

Usage:
    python -m src.preprocessing      # make the split, save split IDs, print a summary
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CLEAN_CSV, DATA_PROCESSED_DIR, ID_COL, RANDOM_STATE, TARGET_COL, TEST_SIZE

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
SPLIT_IDS_CSV = DATA_PROCESSED_DIR / "split_ids.csv"


def collapse_no_service(X: pd.DataFrame) -> pd.DataFrame:
    """Map 'No internet service' / 'No phone service' to 'No'.

    Those levels are fully determined by InternetService == 'No' and PhoneService == 'No',
    so keeping them would create seven identical one-hot columns. Collapsing them removes
    the duplication without losing information. Stateless, so it cannot leak.
    """
    X = X.copy()
    for col in INTERNET_ADDONS:
        if col in X:
            X[col] = X[col].replace("No internet service", "No")
    if "MultipleLines" in X:
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
    return X


def build_preprocessor() -> Pipeline:
    """Unfitted preprocessing pipeline: raw feature frame in, numeric matrix out."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",  # customerID and anything unexpected never reach the model
        verbose_feature_names_out=True,
    )
    return Pipeline([
        ("collapse", FunctionTransformer(collapse_no_service, feature_names_out="one-to-one")),
        ("columns", columns),
    ])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET_COL].astype(int)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, dtype={ID_COL: str})


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with a fixed seed. Returns full rows (ID + features + target)."""
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    df = load_clean()
    train, test = make_split(df)

    ids = pd.concat([
        pd.DataFrame({ID_COL: train[ID_COL], "split": "train"}),
        pd.DataFrame({ID_COL: test[ID_COL], "split": "test"}),
    ])
    ids.to_csv(SPLIT_IDS_CSV, index=False)

    X_train, y_train = split_features_target(train)
    X_test, _ = split_features_target(test)
    pre = build_preprocessor().fit(X_train)  # fitted on train only
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)

    print(f"train {train.shape}, churn rate {y_train.mean():.4f}")
    print(f"test  {test.shape}, churn rate {test[TARGET_COL].mean():.4f}")
    print(f"transformed train {Xt_train.shape}, test {Xt_test.shape}")
    print(f"saved split IDs to {SPLIT_IDS_CSV}")


if __name__ == "__main__":
    main()
