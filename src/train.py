"""Model training with stratified cross-validation on the training split only.

Steps:
  A. Compare imbalance strategies (none / class_weight / SMOTE) for each model family
     at default hyperparameters. Result: reweighting/resampling barely moves PR-AUC
     (a ranking metric); it mainly shifts the 0.5 cut-off and inflates probabilities.
  B. Tune each family with RandomizedSearchCV WITHOUT reweighting (FINAL_STRATEGY), so the
     saved model outputs calibrated probabilities. Class imbalance is then handled
     explicitly by the cost-based decision threshold in Phase 5.
  C. Pick the family with the best mean CV PR-AUC, refit it on the full training split,
     save it with joblib, and save its out-of-fold (OOF) probabilities for Phase 5.

The test split is never touched here; it is used once, in Phase 4.
Resampling (SMOTE) lives inside an imblearn Pipeline, so it runs on the training folds
only and the validation fold always keeps its real class balance.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from src.config import ID_COL, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, DATA_PROCESSED_DIR
from src.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, load_clean,
                               make_split, split_features_target)

N_FOLDS = 5
PRIMARY_METRIC = "average_precision"  # PR-AUC
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "neg_brier_score": "neg_brier_score",
}
STRATEGIES = ["none", "class_weight", "smote"]
FINAL_STRATEGY = "none"

MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
OOF_PATH = DATA_PROCESSED_DIR / "oof_predictions.csv"


def make_estimator(family: str):
    if family == "logreg":
        return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RANDOM_STATE)
    if family == "lightgbm":
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    raise ValueError(family)


def build_pipeline(family: str, strategy: str) -> ImbPipeline:
    model = make_estimator(family)
    if strategy == "class_weight":
        model.set_params(class_weight="balanced")
    sampler = SMOTE(random_state=RANDOM_STATE) if strategy == "smote" else "passthrough"
    # imblearn refuses a nested Pipeline as an intermediate step, so inline its steps.
    return ImbPipeline(build_preprocessor().steps + [("sampler", sampler), ("model", model)])


SEARCH_SPACES = {
    "logreg": (20, {
        "model__C": loguniform(1e-3, 1e2),
    }),
    "random_forest": (25, {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 6, 8, 10, 12, 16],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__max_features": ["sqrt", 0.3, 0.5],
    }),
    "lightgbm": (40, {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(4, 64),
        "model__min_child_samples": randint(10, 100),
        "model__subsample": uniform(0.6, 0.4),
        "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(1e-3, 10),
    }),
}


def summarize_cv(results: dict) -> dict:
    out = {}
    for name in SCORING:
        scores = results[f"test_{name}"]
        if name == "neg_brier_score":
            name, scores = "brier", -scores
        out[f"{name}_mean"] = float(np.mean(scores))
        out[f"{name}_std"] = float(np.std(scores))
    return out


def compare_strategies(X, y, cv) -> pd.DataFrame:
    rows = []
    for family in SEARCH_SPACES:
        for strategy in STRATEGIES:
            res = cross_validate(build_pipeline(family, strategy), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            rows.append({"family": family, "strategy": strategy, **summarize_cv(res)})
            print(f"  {family:13s} {strategy:12s} PR-AUC {rows[-1]['average_precision_mean']:.4f}")
    return pd.DataFrame(rows)


def tune(family: str, strategy: str, X, y, cv) -> tuple[ImbPipeline, dict]:
    n_iter, space = SEARCH_SPACES[family]
    search = RandomizedSearchCV(
        build_pipeline(family, strategy), space, n_iter=n_iter, scoring=SCORING,
        refit=PRIMARY_METRIC, cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    i = search.best_index_
    cv_res = {f"test_{m}": np.array([search.cv_results_[f"split{k}_test_{m}"][i] for k in range(N_FOLDS)])
              for m in SCORING}
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return search.best_estimator_, {"family": family, "strategy": strategy,
                                     "best_params": best_params, **summarize_cv(cv_res)}


def main() -> None:
    train, _ = make_split(load_clean())
    X, y = split_features_target(train)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("A. imbalance strategies at default hyperparameters")
    comparison = compare_strategies(X, y, cv)
    comparison.round(4).to_csv(REPORTS_DIR / "cv_strategy_comparison.csv", index=False)

    print(f"B. tuning (strategy: {FINAL_STRATEGY})")
    tuned, fitted = [], {}
    for family in SEARCH_SPACES:
        estimator, summary = tune(family, FINAL_STRATEGY, X, y, cv)
        tuned.append(summary)
        fitted[family] = estimator
        print(f"  {family:13s} ({summary['strategy']}) PR-AUC {summary['average_precision_mean']:.4f} "
              f"+/- {summary['average_precision_std']:.4f}")
    (REPORTS_DIR / "cv_tuned.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in t.items() if k != "best_params"} for t in tuned]).round(4).to_csv(
        REPORTS_DIR / "cv_tuned.csv", index=False)

    best = max(tuned, key=lambda t: t["average_precision_mean"])
    family = best["family"]
    print(f"C. selected: {family} ({best['strategy']})")

    final = fitted[family]  # already refit on the full training split by RandomizedSearchCV
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    oof = cross_val_predict(final, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pd.DataFrame({ID_COL: train[ID_COL], "y_true": y, "proba": oof}).to_csv(OOF_PATH, index=False)

    metadata = {
        "model_family": family,
        "imbalance_strategy": best["strategy"],
        "best_params": best["best_params"],
        "cv": {"n_folds": N_FOLDS, "shuffle": True, "random_state": RANDOM_STATE,
               **{k: v for k, v in best.items() if k.endswith(("_mean", "_std"))}},
        "selection_rule": (f"highest mean CV {PRIMARY_METRIC} across tuned families, all trained "
                           f"without reweighting so probabilities stay calibrated"),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_train": int(len(y)),
        "train_churn_rate": float(y.mean()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "pandas": pd.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {MODEL_PATH}, {METADATA_PATH}, {OOF_PATH}")


if __name__ == "__main__":
    main()
