"""Web API and dashboard.

    GET  /                       dashboard
    GET  /api/overview           model metrics and segments
    POST /api/ask                {"question": "..."} -> AI analyst answer with the SQL it ran
    POST /api/plan               {"goal": "...", "budget": 300} -> AI campaign plan
    GET  /api/customers/top      ?segment=&limit=  best prospects with reasons
    GET  /api/customers/{id}     one customer's score and reasons
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__, warehouse
from .config import ROOT, settings

app = FastAPI(title="Campaign Copilot", version=__version__,
              description="Customer response model, SHAP reasons and an AI analyst agent over DuckDB.")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=400)


class PlanIn(BaseModel):
    goal: str = Field(min_length=3, max_length=400)
    budget: int = Field(default=300, ge=10, le=2000)


def _wh():
    try:
        return warehouse.get(settings())
    except SystemExit as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "ui" / "index.html")


@app.get("/api/overview")
def overview():
    cfg = settings()
    wh = _wh()
    metrics = json.loads((cfg.artifacts / "metrics.json").read_text(encoding="utf-8"))
    return {"metrics": metrics, "segments": wh.query("SELECT * FROM segments").to_dict(orient="records"), "llm_model": cfg.llm_model}


@app.post("/api/ask")
def ask(body: AskIn):
    from .agent import Analyst

    try:
        return Analyst(settings(), _wh()).ask(body.question.strip()).to_dict()
    except Exception as exc:  # noqa: BLE001  provider errors (no key, no credit) become a readable message
        raise HTTPException(502, f"The language model call failed: {str(exc)[:200]}") from exc


@app.post("/api/plan")
def plan(body: PlanIn):
    from . import planner

    try:
        return planner.plan(settings(), _wh(), body.goal.strip(), body.budget)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"The language model call failed: {str(exc)[:200]}") from exc


@app.get("/api/customers/top")
def top(segment: str = "", limit: int = 10):
    where = f"WHERE Segment = '{segment.replace(chr(39), chr(39) * 2)}'" if segment else ""
    sql = f"SELECT Id, Score, Decile, Segment, TopReasons FROM customers {where} ORDER BY Score DESC LIMIT {max(1, min(limit, 50))}"
    return _wh().query(sql, 50).to_dict(orient="records")


@app.get("/api/customers/{customer_id}")
def customer(customer_id: int):
    df = _wh().query(f"SELECT * FROM customers WHERE Id = {int(customer_id)}", 1)
    if not len(df):
        raise HTTPException(404, "No such customer")
    return json.loads(df.to_json(orient="records", double_precision=4))[0]
