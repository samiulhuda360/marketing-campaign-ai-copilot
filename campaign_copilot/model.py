"""Train the response model, explain every score with SHAP, and segment the customer base.

    python -m campaign_copilot train

Model selection uses stratified 5-fold cross-validation on 80% of customers; the other 20%
is a test set touched once, for the reported numbers. Only ~15% of customers respond, so
models are compared on ROC-AUC and PR-AUC (answering "no" for everyone already scores 85%
accuracy) and the business result is a gains curve: contact the top-scored customers first,
and see what share of all responders that reaches.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data
from .config import Settings

SEED = 42
LABELS = {  # plain-English names for SHAP drivers and segment profiles
    "Recency": "days since last purchase", "Income": "income", "TotalSpend": "total spend", "MntWines": "wine spend",
    "MntMeatProducts": "meat spend", "MntGoldProds": "gold products spend", "MntFruits": "fruit spend",
    "MntFishProducts": "fish spend", "MntSweetProducts": "sweets spend", "NumCatalogPurchases": "catalogue purchases",
    "NumWebPurchases": "web purchases", "NumStorePurchases": "store purchases", "NumWebVisitsMonth": "web visits per month",
    "NumDealsPurchases": "deal purchases", "TenureDays": "customer tenure", "CatalogShare": "share of purchases by catalogue",
    "DealShare": "share of purchases on deals", "Children": "children at home", "Kidhome": "young children",
    "Teenhome": "teenagers at home", "Age": "age", "TotalPurchases": "total purchases", "Complain": "complained recently",
    "Marital_Status": "marital status", "Education": "education",
}


def _pre(d: pd.DataFrame) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), data.numeric_features(d)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), data.CATEGORICAL),
    ], verbose_feature_names_out=False)


def candidates(d: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Baseline (always 'no')": Pipeline([("pre", _pre(d)), ("m", DummyClassifier(strategy="prior"))]),
        "Logistic regression": Pipeline([("pre", _pre(d)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))]),
        "Random forest": Pipeline([("pre", _pre(d)), ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                                                    class_weight="balanced_subsample",
                                                                                    random_state=SEED, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("pre", _pre(d)), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                                                                l2_regularization=1.0, class_weight="balanced",
                                                                                                random_state=SEED))]),
    }


def gains_at(y: np.ndarray, score: np.ndarray, share: float) -> float:
    order = np.argsort(-score)
    n = int(np.ceil(share * len(y)))
    return float(y[order][:n].sum() / y.sum())


def train(cfg: Settings, log=print) -> dict:
    started = time.time()
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    d = data.load(cfg.data_csv)
    X, y = d[data.features(d)], d[data.TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, pipe in candidates(d).items():
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        fitted[name] = pipe
        results[name] = {"cv_roc_auc": round(float(s["test_roc_auc"].mean()), 3), "cv_roc_auc_sd": round(float(s["test_roc_auc"].std()), 3),
                         "cv_pr_auc": round(float(s["test_pr_auc"].mean()), 3), "test_roc_auc": round(float(roc_auc_score(y_te, p)), 3),
                         "test_pr_auc": round(float(average_precision_score(y_te, p)), 3)}
        log(f"  {name:24s} CV ROC-AUC {results[name]['cv_roc_auc']:.3f}  test ROC-AUC {results[name]['test_roc_auc']:.3f}")

    best = max((n for n in results if not n.startswith("Baseline")), key=lambda n: results[n]["cv_roc_auc"])
    # Decision threshold from out-of-fold training predictions (never from the test set), maximising F1.
    oof = np.zeros(len(y_tr))
    for a, b in cv.split(X_tr, y_tr):
        oof[b] = candidates(d)[best].fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    threshold = float(thr[np.argmax((2 * prec * rec / np.clip(prec + rec, 1e-9, None))[:-1])])
    p_te = fitted[best].predict_proba(X_te)[:, 1]
    pred = (p_te >= threshold).astype(int)
    metrics = {"best_model": best, "threshold": round(threshold, 3), "rows": int(len(d)), "response_rate": round(float(y.mean()), 3),
               "models": results,
               "test_at_threshold": {"precision": round(float(precision_score(y_te, pred)), 3), "recall": round(float(recall_score(y_te, pred)), 3),
                                     "f1": round(float(f1_score(y_te, pred)), 3)},
               "responders_reached": {f"top_{int(k * 100)}pct": round(gains_at(y_te, p_te, k), 3) for k in (0.1, 0.2, 0.3, 0.5)}}

    # Customer scores: each customer is scored by a model that never saw them (5-fold out-of-fold),
    # with probabilities that are calibrated, so a score of 0.30 means about a 30% chance and the sum
    # of scores is an honest expected number of responders. Class weighting helps ranking but inflates
    # probabilities, so the scoring model is trained without it (random forest probabilities were
    # already well calibrated in a comparison: 338 expected vs 334 actual responders); other model
    # types get sigmoid calibration.
    def calibrated():
        if best == "Random forest":
            return Pipeline([("pre", _pre(d)), ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1))])
        return CalibratedClassifierCV(candidates(d)[best], method="sigmoid", cv=5)

    scores = cross_val_predict(calibrated(), X, y, cv=cv, method="predict_proba")[:, 1]
    metrics["calibration"] = {"brier": round(float(brier_score_loss(y, scores)), 4), "expected_responders": round(float(scores.sum()), 1),
                              "actual_responders": int(y.sum())}
    final = calibrated().fit(X, y)                    # scores new customers
    explainer = candidates(d)[best].fit(X, y)         # SHAP reasons (direction and size of each driver)
    drivers = explain(explainer, X)
    segments = segment(d)

    scored = d.copy()
    scored["Score"] = scores.round(4)
    scored["Decile"] = pd.qcut(scores, 10, labels=False, duplicates="drop").astype(int)
    scored["Decile"] = scored["Decile"].max() - scored["Decile"] + 1  # 1 = best tenth
    scored["Segment"] = segments
    scored["TopReasons"] = drivers
    scored.to_parquet(cfg.artifacts / "customers.parquet", index=False)
    joblib.dump(final, cfg.artifacts / "model.joblib")
    joblib.dump(explainer, cfg.artifacts / "explainer.joblib")
    metrics["seconds"] = round(time.time() - started, 1)
    (cfg.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def explain(model: Pipeline, X: pd.DataFrame, top: int = 3) -> list[str]:
    """For each customer, the features that pushed the score up most (SHAP, TreeExplainer),
    with one-hot columns folded back into their original feature."""
    import shap

    Z = model.named_steps["pre"].transform(X)
    names = list(model.named_steps["pre"].get_feature_names_out())
    est = model.named_steps["m"]
    if isinstance(est, (RandomForestClassifier, HistGradientBoostingClassifier)):
        values = shap.TreeExplainer(est).shap_values(Z)
        values = values[..., 1] if values.ndim == 3 else values  # class 1 for binary classifiers
    else:
        values = shap.LinearExplainer(est, Z).shap_values(Z)
    origin = [next((c for c in data.CATEGORICAL if n.startswith(c + "_")), n) for n in names]
    contrib = pd.DataFrame(values, columns=origin).T.groupby(level=0).sum().T
    out = []
    for i, row in contrib.iterrows():
        ups = row[row > 0].sort_values(ascending=False).head(top)
        out.append("; ".join(f"{LABELS.get(f, f)} ({_value(X.iloc[i], f)})" for f in ups.index) or "no strong positive signal")
    return out


def _value(row: pd.Series, feature: str) -> str:
    v = row[feature]
    if isinstance(v, str):
        return v
    if feature == "Recency":
        return f"{int(v)} days"
    if feature in ("CatalogShare", "DealShare"):
        return f"{v:.0%}"
    if feature == "TenureDays":
        return f"{int(v)} days"
    if pd.isna(v):
        return "unknown"
    return f"{v:,.0f}"


TRAITS = {  # how a segment is named after its most distinctive traits
    ("Recency", 1): "lapsed", ("Recency", -1): "recently active", ("TotalSpend", 1): "big spenders",
    ("TotalSpend", -1): "light spenders", ("Income", 1): "high income", ("Income", -1): "budget",
    ("NumCatalogPurchases", 1): "catalogue buyers", ("NumWebPurchases", 1): "online shoppers",
    ("NumStorePurchases", 1): "in-store shoppers", ("NumDealsPurchases", 1): "deal seekers",
    ("NumWebVisitsMonth", 1): "frequent browsers", ("Children", 1): "families", ("Children", -1): "no children",
    ("MntWines", 1): "wine lovers", ("MntMeatProducts", 1): "meat buyers",
}


def segment(d: pd.DataFrame, k: int = 5) -> list[str]:
    """Behavioural segments (k-means on spend, channel and family features), each named after
    the two traits that most set it apart from the average customer."""
    cols = ["TotalSpend", "Income", "Recency", "NumCatalogPurchases", "NumWebPurchases", "NumStorePurchases",
            "NumDealsPurchases", "NumWebVisitsMonth", "Children", "MntWines", "MntMeatProducts"]
    Z = StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)
    zdf = pd.DataFrame(Z, columns=cols)
    names = {}
    for c in range(k):
        centre = zdf[labels == c].mean()
        traits = centre.abs().sort_values(ascending=False).index[:2]
        words = [TRAITS.get((t, 1 if centre[t] > 0 else -1), ("high " if centre[t] > 0 else "low ") + LABELS.get(t, t)) for t in traits]
        names[c] = words[0].capitalize() + ", " + words[1]
    return [names[c] for c in labels]
