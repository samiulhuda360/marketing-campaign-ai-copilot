"""MCP server: the scored customer base and the response model as tools for any AI assistant
(Claude Desktop, Claude Code, Cursor, ChatGPT and other Model Context Protocol clients).

    python -m campaign_copilot mcp            # stdio transport, for desktop clients

The assistant's own model does the reasoning; these tools give it read-only data access,
customer explanations and scoring for new customers. No LLM key is needed on this side.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from fastmcp import FastMCP

from . import data, warehouse
from .config import settings
from .model import explain

mcp = FastMCP("campaign-copilot", instructions=(
    "Customer data and a response model for a grocery superstore's marketing campaigns. "
    "Read the schema resource first, then use run_sql for analysis. Score is the predicted chance "
    "(0-1) of responding to the next offer; Response is what happened in the last campaign."))


def _wh():
    return warehouse.get(settings())


@mcp.resource("campaign://schema")
def schema() -> str:
    """Tables, columns and what each column means."""
    return _wh().schema()


@mcp.tool
def run_sql(sql: str) -> str:
    """Run one read-only DuckDB SELECT query on the `customers` table or `segments` view (max 50 rows)."""
    return _wh().query(sql, 50).to_json(orient="records", double_precision=4)


@mcp.tool
def segment_summary() -> str:
    """Every behavioural segment with size, average predicted score, past response rate, spend and income."""
    return _wh().query("SELECT * FROM segments").to_json(orient="records", double_precision=3)


@mcp.tool
def top_customers(segment: str = "", limit: int = 20) -> str:
    """The customers most likely to respond, optionally within one segment, with the model's reasons."""
    where = f"WHERE Segment = '{segment.replace(chr(39), chr(39) * 2)}'" if segment else ""
    sql = f"SELECT Id, Score, Decile, Segment, TopReasons FROM customers {where} ORDER BY Score DESC LIMIT {max(1, min(int(limit), 50))}"
    return _wh().query(sql, 50).to_json(orient="records", double_precision=4)


@mcp.tool
def explain_customer(customer_id: int) -> str:
    """One customer's predicted score, decile, segment and the reasons behind the score."""
    df = _wh().query(f"SELECT Id, Score, Decile, Segment, TopReasons, Recency, TotalSpend, Income FROM customers WHERE Id = {int(customer_id)}", 1)
    return df.to_json(orient="records", double_precision=4) if len(df) else json.dumps({"error": f"No customer with Id {customer_id}"})


@mcp.tool
def score_new_customer(customer: dict) -> str:
    """Predict the response probability for a customer who is not in the data. Pass the raw fields
    (Income, Recency, MntWines, NumCatalogPurchases, Education, Marital_Status, ...); missing numbers
    are filled with typical values."""
    cfg = settings()
    model, explainer = joblib.load(cfg.artifacts / "model.joblib"), joblib.load(cfg.artifacts / "explainer.joblib")
    base = data.load(cfg.data_csv)
    cols = data.features(base)
    row = base[cols].median(numeric_only=True).to_dict()
    row.update({c: base[c].mode()[0] for c in data.CATEGORICAL})
    row.update({k: v for k, v in customer.items() if k in cols})
    X = pd.DataFrame([row])[cols]
    return json.dumps({"score": round(float(model.predict_proba(X)[0, 1]), 4), "reasons": explain(explainer, X)[0]})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
