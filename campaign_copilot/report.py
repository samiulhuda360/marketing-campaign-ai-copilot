"""Evaluation charts for the README: ROC curves, gains, permutation importance, segments.

    python -m campaign_copilot report      # writes docs/figures/*.png

Steps: clean and engineer features, hold out 20% as a test set, compare models with
stratified 5-fold cross-validation on the rest, refit the best on all training data, and
report on the untouched test set. Because only ~15% of customers respond, accuracy is
misleading (answering "no" for everyone scores 85%), so models are compared on ROC-AUC
and PR-AUC, and the business result is reported as a gains chart: if the company only
contacts its top-scored customers, what share of all responders does it reach?
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from . import data  # noqa: E402
from .config import ROOT, settings  # noqa: E402
from .model import candidates  # noqa: E402

SEED = 42
FIG = ROOT / "docs" / "figures"
INK, ACCENT, ACCENT2, MUTED = "#1f2a30", "#c2410c", "#0f766e", "#9aa5ab"


def load() -> pd.DataFrame:
    return data.load(settings().data_csv).drop(columns=["Id"])


def models(d: pd.DataFrame) -> dict[str, Pipeline]:
    """The same candidates and settings as training, so the charts match the reported metrics."""
    return candidates(d)


def gains(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    captured = np.cumsum(y[order]) / y.sum()
    contacted = np.arange(1, len(y) + 1) / len(y)
    return contacted, captured


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    y = d.pop("Response").to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(d, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, pipe in models(d).items():
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        p = pipe.predict_proba(X_test)[:, 1]
        results[name] = {"cv_roc_auc": round(s["test_roc_auc"].mean(), 3), "cv_roc_auc_sd": round(s["test_roc_auc"].std(), 3),
                         "cv_pr_auc": round(s["test_pr_auc"].mean(), 3), "test_roc_auc": round(roc_auc_score(y_test, p), 3),
                         "test_pr_auc": round(average_precision_score(y_test, p), 3)}
        print(f"{name:24s} CV ROC-AUC {results[name]['cv_roc_auc']:.3f} ± {results[name]['cv_roc_auc_sd']:.3f}  "
              f"test ROC-AUC {results[name]['test_roc_auc']:.3f}  test PR-AUC {results[name]['test_pr_auc']:.3f}")

    best = max((n for n in results if not n.startswith("Baseline")), key=lambda n: results[n]["cv_roc_auc"])
    p_best = fitted[best].predict_proba(X_test)[:, 1]

    # Decision threshold chosen on training folds (out-of-fold), never on the test set: maximise F1.
    oof = np.zeros(len(y_train))
    for tr, va in cv.split(X_train, y_train):
        m = models(d)[best].fit(X_train.iloc[tr], y_train[tr])
        oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    threshold = float(thr[np.argmax(f1[:-1])])
    pred = (p_best >= threshold).astype(int)

    contacted, captured = gains(y_test, p_best)
    top = {f"top_{int(k * 100)}pct": round(float(captured[int(np.ceil(k * len(y_test))) - 1]), 3) for k in (0.1, 0.2, 0.3, 0.5)}
    summary = {"rows_after_cleaning": int(len(d)), "response_rate": round(float(y.mean()), 3), "best_model": best, "models": results,
               "threshold": round(threshold, 3), "test_at_threshold": {"precision": round(precision_score(y_test, pred), 3),
                                                                        "recall": round(recall_score(y_test, pred), 3),
                                                                        "f1": round(f1_score(y_test, pred), 3)},
               "responders_reached_when_contacting": top}

    imp = permutation_importance(fitted[best], X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    summary["top_features"] = {k: round(float(v), 4) for k, v in importance.head(8).items()}
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))

    # Figures
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, colour in zip([n for n in results if not n.startswith("Baseline")], [MUTED, ACCENT2, ACCENT], strict=True):
        fpr, tpr, _ = roc_curve(y_test, fitted[name].predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, color=colour, lw=2.2 if name == best else 1.5, label=f"{name} (AUC {results[name]['test_roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="Random guess (0.50)")
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate"), ax.legend(frameon=False, fontsize=9)
    style(ax, "ROC curves on the held-out test set")
    fig.tight_layout(), fig.savefig(FIG / "roc.png", dpi=160), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(contacted * 100, captured * 100, color=ACCENT, lw=2.4, label=best)
    ax.plot([0, 100], [0, 100], "--", color=MUTED, lw=1, label="Contact customers at random")
    k = 0.2
    ax.scatter([k * 100], [top["top_20pct"] * 100], color=ACCENT, zorder=3)
    ax.annotate(f"Top 20% of customers\nreach {top['top_20pct'] * 100:.0f}% of responders", (k * 100, top["top_20pct"] * 100),
                xytext=(34, 38), fontsize=10, color=INK, arrowprops={"arrowstyle": "->", "color": INK})
    ax.set_xlabel("Customers contacted, highest score first (%)"), ax.set_ylabel("Responders reached (%)"), ax.legend(frameon=False, fontsize=9)
    style(ax, "Cumulative gains: who to contact first")
    fig.tight_layout(), fig.savefig(FIG / "gains.png", dpi=160), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    top_imp = importance.head(10)[::-1]
    ax.barh(top_imp.index, top_imp.values, color=ACCENT2)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    style(ax, f"What drives a response ({best})")
    fig.tight_layout(), fig.savefig(FIG / "importance.png", dpi=160), plt.close(fig)

    df = load()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, label, bins) in zip(axes, [("Recency", "Days since last purchase", [0, 20, 40, 60, 80, 100]),
                                             ("TotalSpend", "Total spend (2 years)", [0, 100, 500, 1000, 1500, 2600]),
                                             ("NumCatalogPurchases", "Catalogue purchases", [0, 1, 3, 6, 10, 29])], strict=True):
        rate = df.groupby(pd.cut(df[col], bins, include_lowest=True), observed=True)["Response"].mean() * 100
        ax.bar(range(len(rate)), rate.values, color=ACCENT2)
        ax.set_xticks(range(len(rate)), [f"{int(i.left)}-{int(i.right)}" for i in rate.index], fontsize=8, rotation=20)
        ax.set_ylabel("Response rate (%)") if col == "Recency" else None
        style(ax, label)
    fig.tight_layout(), fig.savefig(FIG / "response-by-segment.png", dpi=160), plt.close(fig)


if __name__ == "__main__":
    main()
