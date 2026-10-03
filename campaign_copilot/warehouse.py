"""The scored customer base as a read-only DuckDB database for the AI analyst.

One table, `customers`, plus a `segments` view. The database is built in memory from the
trained artifacts, then locked: external file and network access are switched off and the
configuration is frozen, so SQL written by a language model can only read this data.
"""

from __future__ import annotations

import re
import threading

import duckdb
import pandas as pd

from .config import Settings

COLUMNS = {
    "Id": "customer id",
    "Education": "Basic, Graduation, Master, PhD",
    "Marital_Status": "Single, Married, Divorced, Widow",
    "Income": "yearly household income in dollars (24 missing)",
    "Kidhome": "number of young children", "Teenhome": "number of teenagers", "Children": "Kidhome + Teenhome",
    "Recency": "days since the last purchase",
    "MntWines": "2-year spend on wine", "MntFruits": "spend on fruit", "MntMeatProducts": "spend on meat",
    "MntFishProducts": "spend on fish", "MntSweetProducts": "spend on sweets", "MntGoldProds": "spend on gold products",
    "TotalSpend": "sum of all spend columns",
    "NumDealsPurchases": "purchases made with a discount", "NumWebPurchases": "purchases on the website",
    "NumCatalogPurchases": "purchases from the catalogue", "NumStorePurchases": "purchases in store",
    "TotalPurchases": "web + catalogue + store purchases", "NumWebVisitsMonth": "website visits last month",
    "DealShare": "share of purchases made on a deal (0-1)", "CatalogShare": "share of purchases from the catalogue (0-1)",
    "Complain": "1 if the customer complained in the last 2 years",
    "Age": "age in years", "TenureDays": "days since the customer joined",
    "Response": "1 if the customer accepted the LAST campaign offer (historical outcome)",
    "Score": "model-predicted probability (0-1) of responding to the NEXT offer",
    "Decile": "score decile, 1 = top 10% most likely to respond, 10 = least likely",
    "Segment": "behavioural segment name",
    "TopReasons": "plain-English reasons the model gave this customer a high score",
}

BLOCKED = re.compile(r"\b(insert|update|delete|merge|create|drop|alter|attach|detach|copy|export|import|install|load|pragma|set|reset|"
                     r"call|checkpoint|vacuum|grant|revoke|use)\b", re.I)


class UnsafeQuery(ValueError):
    pass


class Warehouse:
    def __init__(self, customers: pd.DataFrame):
        self._lock = threading.Lock()
        self.con = duckdb.connect()
        self.con.register("_src", customers)
        self.con.execute("CREATE TABLE customers AS SELECT * FROM _src")
        self.con.unregister("_src")
        self.con.execute("""CREATE VIEW segments AS
            SELECT Segment, COUNT(*) AS customers, ROUND(AVG(Score), 3) AS avg_score, ROUND(AVG(Response), 3) AS past_response_rate,
                   ROUND(AVG(TotalSpend)) AS avg_spend, ROUND(AVG(Income)) AS avg_income, ROUND(AVG(Recency)) AS avg_recency_days
            FROM customers GROUP BY Segment ORDER BY avg_score DESC""")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    @classmethod
    def from_artifacts(cls, cfg: Settings) -> "Warehouse":
        path = cfg.artifacts / "customers.parquet"
        if not path.exists():
            raise SystemExit("No trained model yet. Run: python -m campaign_copilot train")
        return cls(pd.read_parquet(path))

    def schema(self) -> str:
        cols = self.con.execute("DESCRIBE customers").fetchall()
        lines = [f"- {name} ({dtype}): {COLUMNS.get(name, '')}" for name, dtype, *_ in cols]
        return "Table customers (one row per customer):\n" + "\n".join(lines) + \
            "\nView segments: Segment, customers, avg_score, past_response_rate, avg_spend, avg_income, avg_recency_days"

    def query(self, sql: str, max_rows: int = 50) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        if ";" in sql:
            raise UnsafeQuery("Run one statement at a time.")
        if not re.match(r"^\s*(select|with)\b", sql, re.I) or BLOCKED.search(re.sub(r"'[^']*'", "''", sql)):
            raise UnsafeQuery("Only read-only SELECT queries are allowed.")
        with self._lock:
            return self.con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {int(max_rows)}").df()


_cache: dict[str, Warehouse] = {}


def get(cfg: Settings) -> Warehouse:
    """One shared in-memory database per artifacts folder."""
    key = str(cfg.artifacts)
    if key not in _cache:
        _cache[key] = Warehouse.from_artifacts(cfg)
    return _cache[key]


def reset() -> None:
    _cache.clear()
