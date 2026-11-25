"""Evaluate the saved model on the held-out test split (used once, here).

Model selection already happened in Phase 3 on cross-validation. The two other tuned
families are refitted on the training split and scored on test for context only; their
test scores do not change which model is used.

Outputs:
    reports/metrics_test.json        metrics of the selected model (+ bootstrap 95% CI)
    reports/test_comparison.csv      all tuned families on test, for context
    data/processed/test_predictions.csv
    reports/figures/05_pr_roc_test.png, 06_confusion_matrix_test.png, 07_calibration_test.png

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, ID_COL, RANDOM_STATE, REPORTS_DIR
from src.preprocessing import load_clean, make_split, split_features_target
from src.train import METADATA_PATH, MODEL_PATH, build_pipeline

DEFAULT_THRESHOLD = 0.5
N_BOOTSTRAP = 1000
TEST_PRED_PATH = DATA_PROCESSED_DIR / "test_predictions.csv"

LABELS = {"lightgbm": "LightGBM", "logreg": "Logistic Regression", "random_forest": "Random Forest"}
COLORS = {"lightgbm": "#2a78d6", "logreg": "#eb6834", "random_forest": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
})


def compute_metrics(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Threshold-free ranking metrics plus metrics of the hard decision at `threshold`."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y_true, proba, n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> dict:
    """Percentile 95% CI for PR-AUC and ROC-AUC by resampling test rows with replacement."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pr, roc = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        pr.append(average_precision_score(y_true[idx], proba[idx]))
        roc.append(roc_auc_score(y_true[idx], proba[idx]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n_resamples": len(pr), "pr_auc_95ci": q(pr), "roc_auc_95ci": q(roc)}


def plot_pr_roc(y, probas: dict, path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    prevalence = y.mean()
    for fam, p in probas.items():
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        lw = 2.4 if fam == "lightgbm" else 1.6
        ax1.plot(rec, prec, color=COLORS[fam], lw=lw,
                 label=f"{LABELS[fam]} (PR-AUC {average_precision_score(y, p):.3f})")
        ax2.plot(fpr, tpr, color=COLORS[fam], lw=lw, label=f"{LABELS[fam]} (ROC-AUC {roc_auc_score(y, p):.3f})")
    ax1.axhline(prevalence, color=INK_2, lw=1, ls="--", label=f"No skill ({prevalence:.3f})")
    ax2.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="No skill (0.500)")
    ax1.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test)", xlim=(0, 1), ylim=(0, 1.02))
    ax2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)",
            xlim=(0, 1), ylim=(0, 1.02))
    ax1.legend(loc="upper right")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: dict, threshold: float, path) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(matrix, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#104281"]))
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = matrix[i, j] > matrix.max() * 0.5
            ax.text(j, i, f"{matrix[i, j]:,}\n{names[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xticks([0, 1], ["Predicted: stay", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
    ax.grid(False)
    ax.set_title(f"LightGBM confusion matrix (test, threshold {threshold})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y, probas: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="Perfect calibration")
    for fam, p in probas.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, color=COLORS[fam], lw=2 if fam == "lightgbm" else 1.4, marker="o", ms=4,
                label=f"{LABELS[fam]} (Brier {brier_score_loss(y, p):.3f})")
    ax.set(xlabel="Mean predicted probability (decile)", ylabel="Observed churn rate",
           title="Calibration (test)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train, test = make_split(load_clean())
    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)

    model = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    final_family = meta["model_family"]
    probas = {final_family: model.predict_proba(X_test)[:, 1]}

    # Context only: the other tuned families, refitted on train with their CV-tuned params.
    for t in json.loads((REPORTS_DIR / "cv_tuned.json").read_text(encoding="utf-8")):
        if t["family"] != final_family:
            pipe = build_pipeline(t["family"], t["strategy"]).set_params(**t["best_params"]).fit(X_train, y_train)
            probas[t["family"]] = pipe.predict_proba(X_test)[:, 1]

    proba = probas[final_family]
    metrics = compute_metrics(y_test, proba)
    metrics.update(bootstrap_ci(y_test, proba))
    metrics["baselines"] = {
        "no_skill_pr_auc": float(y_test.mean()),
        "always_stay_accuracy": float(1 - y_test.mean()),
    }
    metrics.update({"model_family": final_family, "n_test": int(len(y_test)),
                    "n_test_churn": int(y_test.sum()), "mean_predicted_proba": float(proba.mean())})
    (REPORTS_DIR / "metrics_test.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for fam, p in probas.items():
        m = compute_metrics(y_test, p)
        cm = m.pop("confusion_matrix")
        rows.append({"family": fam, "selected": fam == final_family, **m, **cm})
    pd.DataFrame(rows).round(4).to_csv(REPORTS_DIR / "test_comparison.csv", index=False)

    pd.DataFrame({ID_COL: test[ID_COL], "y_true": y_test, "proba": proba}).to_csv(TEST_PRED_PATH, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = {k: probas[k] for k in ["lightgbm", "logreg", "random_forest"] if k in probas}
    plot_pr_roc(y_test, order, FIGURES_DIR / "05_pr_roc_test.png")
    plot_confusion(metrics["confusion_matrix"], DEFAULT_THRESHOLD, FIGURES_DIR / "06_confusion_matrix_test.png")
    plot_calibration(y_test, order, FIGURES_DIR / "07_calibration_test.png")

    print(json.dumps({k: v for k, v in metrics.items()}, indent=2))
    print(pd.DataFrame(rows).drop(columns=["threshold"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
