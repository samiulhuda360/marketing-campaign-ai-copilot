"""Load the customer file, clean it and engineer features.

The raw file has 2,240 customers of a grocery superstore: demographics, two years of spend
by category, purchases by channel, and Response (1 if they accepted the last campaign offer).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SPEND = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNELS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CATEGORICAL = ["Education", "Marital_Status"]
TARGET = "Response"


def load(csv: Path) -> pd.DataFrame:
    """Clean customers with engineered features; Id is kept for look-ups but is never a feature."""
    d = pd.read_csv(csv)
    d["Dt_Customer"] = pd.to_datetime(d["Dt_Customer"], format="%m/%d/%Y")
    ref = d["Dt_Customer"].max()
    d["Age"] = ref.year - d["Year_Birth"]
    # Data errors: three birth years before 1901 and one income of 666,666.
    d = d[(d["Age"] < 100) & (d["Income"].fillna(0) < 200_000)].copy()
    d["TotalSpend"] = d[SPEND].sum(axis=1)
    d["TotalPurchases"] = d[CHANNELS].sum(axis=1)
    d["Children"] = d["Kidhome"] + d["Teenhome"]
    d["TenureDays"] = (ref - d["Dt_Customer"]).dt.days
    d["DealShare"] = (d["NumDealsPurchases"] / d["TotalPurchases"].clip(lower=1)).round(3)
    d["CatalogShare"] = (d["NumCatalogPurchases"] / d["TotalPurchases"].clip(lower=1)).round(3)
    d["Marital_Status"] = d["Marital_Status"].replace({"Alone": "Single", "YOLO": "Single", "Absurd": "Single", "Together": "Married"})
    d["Education"] = d["Education"].replace({"2n Cycle": "Master"})
    return d.drop(columns=["Year_Birth", "Dt_Customer"]).reset_index(drop=True)


def features(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c not in ("Id", TARGET)]


def numeric_features(d: pd.DataFrame) -> list[str]:
    return [c for c in features(d) if c not in CATEGORICAL]
