"""SHAP explanations for the saved LightGBM pipeline.

SHAP values are computed by TreeExplainer on the model's log-odds output. One-hot columns
are summed back to their original feature (e.g. the three Contract_* columns -> Contract),
which is exact for SHAP because contributions are additive. That gives one number per
business feature, which is easier to read and is what the Streamlit app shows.

SHAP explains the model, not the world: a feature that raises predicted risk is
associated with churn in this data, not proven to cause it.

Usage:
    python -m src.explain
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, ID_COL, REPORTS_DIR
from src.evaluate import INK, INK_2  # also applies the shared plot style
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_clean, make_split, split_features_target
from src.train import MODEL_PATH


def preprocessor_of(model) -> Pipeline:
    """The fitted preprocessing steps of the saved pipeline (everything before sampler/model)."""
    return Pipeline([(name, model.named_steps[name]) for name in ("collapse", "columns")])


def encoded_to_original(name: str) -> str:
    """Map an encoded column name (num__tenure, cat__Contract_One year) to its source feature."""
    prefix, rest = name.split("__", 1)
    if prefix == "num":
        return rest
    return max((f for f in CATEGORICAL_FEATURES if rest.startswith(f + "_")), key=len)


class ChurnExplainer:
    def __init__(self, model):
        self.model = model
        self.pre = preprocessor_of(model)
        self.explainer = shap.TreeExplainer(model.named_steps["model"])
        self.encoded_names = list(self.pre.get_feature_names_out())
        self.groups = [encoded_to_original(n) for n in self.encoded_names]

    @property
    def base_value(self) -> float:
        ev = self.explainer.expected_value
        return float(np.ravel(ev)[-1])

    def encoded_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = pd.DataFrame(self.pre.transform(X[FEATURES]), columns=self.encoded_names, index=X.index)
        with warnings.catch_warnings():
            # SHAP warns that LightGBM binary output format changed; both formats are handled below.
            warnings.filterwarnings("ignore", message="LightGBM binary classifier")
            values = self.explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return pd.DataFrame(values, columns=self.encoded_names, index=X.index)

    def feature_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP per original feature (one-hot contributions summed), log-odds units."""
        enc = self.encoded_shap(X)
        return enc.T.groupby(self.groups, sort=False).sum().T[FEATURES]

    def explain_row(self, row: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Top drivers for one customer: feature, its value, SHAP contribution, direction."""
        contrib = self.feature_shap(row).iloc[0]
        out = pd.DataFrame({"feature": contrib.index, "value": row[FEATURES].iloc[0].astype(str).values,
                            "shap": contrib.values})
        out["direction"] = np.where(out["shap"] > 0, "raises risk", "lowers risk")
        return out.reindex(out["shap"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def plot_global_bar(importance: pd.Series, path) -> None:
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.barh(imp.index, imp.values, color="#2a78d6", height=0.65)
    for y, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, y, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set(xlabel="Mean |SHAP| (log-odds), one-hot columns summed per feature",
           title="Global feature importance (test set)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(enc_shap: pd.DataFrame, Xt: pd.DataFrame, path, max_display: int = 15) -> None:
    shap.summary_plot(enc_shap.values, Xt, max_display=max_display, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    plt.title("SHAP beeswarm (encoded features, test set)", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(X: pd.DataFrame, fshap: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    contract_colors = {"Month-to-month": "#eb6834", "One year": "#2a78d6", "Two year": "#1baf7a"}
    for ax, col in zip(axes, NUMERIC_FEATURES):
        for level, color in contract_colors.items():
            m = X["Contract"] == level
            ax.scatter(X.loc[m, col], fshap.loc[m, col], s=6, alpha=0.5, color=color, label=level, linewidths=0)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set(xlabel=col, ylabel=f"SHAP of {col} (log-odds)", title=col)
    axes[0].legend(title="Contract", markerscale=2.5, loc="upper right", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall(explainer: ChurnExplainer, row: pd.DataFrame, title: str, path, max_display: int = 10) -> None:
    contrib = explainer.feature_shap(row).iloc[0]
    exp = shap.Explanation(values=contrib.values, base_values=explainer.base_value,
                           data=row[FEATURES].iloc[0].astype(str).values, feature_names=FEATURES)
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.title(title, fontsize=10, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_examples(test: pd.DataFrame, proba: np.ndarray) -> dict:
    """Three illustrative test customers: highest risk churner, a mid-risk churner, lowest-risk stayer."""
    t = test.assign(proba=proba)
    churners = t[t["Churn"] == 1].sort_values("proba")
    stayers = t[t["Churn"] == 0].sort_values("proba")
    mid = churners.iloc[(churners["proba"] - 0.5).abs().argsort().iloc[0]]
    return {"high_risk": churners.index[-1], "mid_risk": mid.name, "low_risk": stayers.index[0]}


def main() -> None:
    model = joblib.load(MODEL_PATH)
    _, test = make_split(load_clean())
    X_test, _ = split_features_target(test)
    explainer = ChurnExplainer(model)

    enc = explainer.encoded_shap(X_test)
    fshap = explainer.feature_shap(X_test)

    # Sanity check: SHAP values must add up to the model's log-odds output.
    proba = model.predict_proba(X_test)[:, 1]
    logit = np.log(proba / (1 - proba))
    max_gap = float(np.abs(explainer.base_value + enc.sum(axis=1).values - logit).max())
    print(f"additivity check, max |base + sum(SHAP) - logit|: {max_gap:.2e}")

    importance = fshap.abs().mean().sort_values(ascending=False)
    direction = pd.Series({f: np.corrcoef(pd.to_numeric(X_test[f]), fshap[f])[0, 1]
                           for f in NUMERIC_FEATURES})
    importance.rename("mean_abs_shap").round(4).to_csv(REPORTS_DIR / "shap_global_importance.csv")
    fshap.assign(**{ID_COL: test[ID_COL].values}).round(5).to_csv(REPORTS_DIR / "shap_values_test.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Xt = pd.DataFrame(explainer.pre.transform(X_test), columns=explainer.encoded_names)
    plot_global_bar(importance, FIGURES_DIR / "09_shap_global_bar.png")
    plot_beeswarm(enc, Xt, FIGURES_DIR / "10_shap_beeswarm.png")
    plot_dependence(X_test, fshap, FIGURES_DIR / "11_shap_dependence_numeric.png")

    examples = pick_examples(test, proba)
    for i, (label, idx) in enumerate(examples.items()):
        row = X_test.loc[[idx]]
        title = (f"{label.replace('_', ' ')}: customer {test.loc[idx, ID_COL]}, "
                 f"p(churn) = {proba[idx]:.2f}, actual = {'churn' if test.loc[idx, 'Churn'] else 'stay'}")
        plot_waterfall(explainer, row, title, FIGURES_DIR / f"12{'abc'[i]}_shap_waterfall_{label}.png")
        print(f"\n{title}\n{explainer.explain_row(row).round(3).to_string(index=False)}")

    print(f"\nbase value (log-odds): {explainer.base_value:.4f} -> p = {1 / (1 + np.exp(-explainer.base_value)):.4f}")
    print("\nmean |SHAP| per feature:\n" + importance.round(4).to_string())
    print("\ncorr(feature value, its SHAP) for numeric features:\n" + direction.round(3).to_string())


if __name__ == "__main__":
    main()
