"""Cost-based decision threshold.

Decision: contact a customer with a retention offer when predicted churn probability >= t.
Profit is measured against doing nothing:

    contacted churner     (TP): RETENTION_SUCCESS_RATE * CHURN_LOSS_IDR - RETENTION_COST_IDR
    contacted non-churner (FP): -RETENTION_COST_IDR   (offer spent on someone who would stay)
    not contacted         (FN, TN): 0                 (same outcome as doing nothing)

With calibrated probabilities the profit-maximizing rule is "contact when
p * success * loss > cost", i.e. t = cost / (success * loss). We still pick t empirically
on out-of-fold (OOF) training predictions, then report the result once on the test split.

Usage:
    python -m src.threshold
    python -m src.threshold --retention-cost 75000 --churn-loss 2000000 --success-rate 0.4
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (CHURN_LOSS_IDR, DATA_PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR,
                        RETENTION_COST_IDR, RETENTION_SUCCESS_RATE)
from src.evaluate import INK, INK_2, compute_metrics  # also applies the shared plot style

THRESHOLDS = np.round(np.arange(0.01, 0.991, 0.005), 3)
THRESHOLD_PATH = MODELS_DIR / "decision_threshold.json"
OOF_PATH = DATA_PROCESSED_DIR / "oof_predictions.csv"
TEST_PRED_PATH = DATA_PROCESSED_DIR / "test_predictions.csv"


def profit_at_threshold(y_true, proba, threshold: float, cost: float, loss: float, success: float) -> dict:
    y_true, contact = np.asarray(y_true), np.asarray(proba) >= threshold
    tp = int((contact & (y_true == 1)).sum())
    fp = int((contact & (y_true == 0)).sum())
    fn = int((~contact & (y_true == 1)).sum())
    profit = tp * (success * loss - cost) - fp * cost
    return {"threshold": float(threshold), "contacted": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "profit_idr": float(profit), "profit_per_1000_idr": float(profit / len(y_true) * 1000)}


def profit_curve(y_true, proba, cost: float, loss: float, success: float, thresholds=THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame([profit_at_threshold(y_true, proba, t, cost, loss, success) for t in thresholds])


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["profit_idr"].idxmax(), "threshold"])


def break_even_threshold(cost: float, loss: float, success: float) -> float:
    return cost / (success * loss)


def idr(x: float) -> str:
    return f"Rp {x:,.0f}".replace(",", ".")


def plot_profit(oof_curve, test_curve, t_star, t_even, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(oof_curve["threshold"], oof_curve["profit_per_1000_idr"] / 1e6, color="#2a78d6", lw=2,
            label="Train OOF (used to pick the threshold)")
    ax.plot(test_curve["threshold"], test_curve["profit_per_1000_idr"] / 1e6, color="#eb6834", lw=1.6,
            label="Test (reported once)")
    ax.axvline(t_star, color=INK, lw=1, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(t_star + 0.01, ymax * 0.35, f"chosen t = {t_star:.3f}", fontsize=8, color=INK, va="top")
    ax.axvline(0.5, color=INK_2, lw=1, ls=":")
    ax.text(0.51, ymax * 0.35, "default t = 0.5", fontsize=8, color=INK_2, va="top")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="Decision threshold (contact if churn probability >= t)",
           ylabel="Profit per 1,000 customers (million IDR)",
           title=f"Profit vs. threshold (break-even probability {t_even:.3f})", xlim=(0, 1))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(cost: float, loss: float, success: float) -> None:
    oof = pd.read_csv(OOF_PATH)
    test = pd.read_csv(TEST_PRED_PATH)
    assumptions = {"retention_cost_idr": cost, "churn_loss_idr": loss, "retention_success_rate": success}

    oof_curve = profit_curve(oof["y_true"], oof["proba"], cost, loss, success)
    test_curve = profit_curve(test["y_true"], test["proba"], cost, loss, success)
    t_star = best_threshold(oof_curve)
    t_even = break_even_threshold(cost, loss, success)

    y, p = test["y_true"], test["proba"]
    policies = {
        "chosen_threshold": t_star,
        "break_even_threshold": t_even,
        "default_0.5": 0.5,
        "contact_everyone": 0.0,
        "contact_nobody": 1.01,
    }
    rows = []
    for name, t in policies.items():
        r = profit_at_threshold(y, p, t, cost, loss, success)
        m = compute_metrics(y, p, threshold=min(t, 1.0)) if name != "contact_nobody" else None
        rows.append({"policy": name, **r,
                     "precision": m["precision"] if m else None, "recall": m["recall"] if m else 0.0})
    policy_table = pd.DataFrame(rows)
    policy_table.round(4).to_csv(REPORTS_DIR / "threshold_policies_test.csv", index=False)

    # Sensitivity: how the chosen threshold and test profit move with the assumptions.
    sens = []
    for c, l, s in product([25_000, 50_000, 100_000, 200_000], [750_000, 1_500_000, 3_000_000], [0.3, 0.5]):
        t = best_threshold(profit_curve(oof["y_true"], oof["proba"], c, l, s))
        r = profit_at_threshold(y, p, t, c, l, s)
        best_naive = max(profit_at_threshold(y, p, 0.0, c, l, s)["profit_idr"], 0.0)  # contact all vs nobody
        sens.append({"retention_cost_idr": c, "churn_loss_idr": l, "success_rate": s,
                     "break_even": round(break_even_threshold(c, l, s), 4), "chosen_threshold": t,
                     "test_contacted_share": round(r["contacted"] / len(y), 4),
                     "test_recall": round(r["tp"] / int(y.sum()), 4),
                     "test_profit_idr": r["profit_idr"],
                     "test_profit_vs_0.5_idr": r["profit_idr"] - profit_at_threshold(y, p, 0.5, c, l, s)["profit_idr"],
                     "test_profit_vs_best_naive_idr": r["profit_idr"] - best_naive})
    pd.DataFrame(sens).to_csv(REPORTS_DIR / "threshold_sensitivity.csv", index=False)

    chosen = policy_table.set_index("policy").loc["chosen_threshold"]
    result = {
        "assumptions": assumptions,
        "threshold": t_star,
        "break_even_threshold": t_even,
        "selected_on": "train out-of-fold predictions",
        "oof_profit_per_1000_idr": float(oof_curve.loc[oof_curve["threshold"] == t_star, "profit_per_1000_idr"].iloc[0]),
        "test": {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else v))
                 for k, v in chosen.items()},
        "test_n": int(len(y)),
    }
    THRESHOLD_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_profit(oof_curve, test_curve, t_star, t_even, FIGURES_DIR / "08_profit_curve.png")

    print(f"assumptions: cost {idr(cost)}, loss {idr(loss)}, success {success:.0%}")
    print(f"break-even probability: {t_even:.4f} | chosen threshold on OOF: {t_star:.3f}")
    show = policy_table.copy()
    show["profit"] = show["profit_idr"].map(idr)
    print(show.drop(columns=["profit_idr", "profit_per_1000_idr"]).round(4).to_string(index=False))
    print(f"saved {THRESHOLD_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retention-cost", type=float, default=RETENTION_COST_IDR)
    ap.add_argument("--churn-loss", type=float, default=CHURN_LOSS_IDR)
    ap.add_argument("--success-rate", type=float, default=RETENTION_SUCCESS_RATE)
    a = ap.parse_args()
    main(a.retention_cost, a.churn_loss, a.success_rate)
