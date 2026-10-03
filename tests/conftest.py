import json
from types import SimpleNamespace

import pandas as pd
import pytest

from campaign_copilot.warehouse import Warehouse


@pytest.fixture
def customers() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": [1, 2, 3, 4], "Income": [90000.0, 40000.0, 60000.0, None], "Recency": [5, 80, 30, 95],
        "MntWines": [1200, 50, 400, 10], "TotalSpend": [2000, 120, 900, 40], "NumCatalogPurchases": [9, 0, 3, 0],
        "NumWebPurchases": [4, 1, 6, 0], "NumStorePurchases": [8, 2, 5, 1], "NumDealsPurchases": [1, 3, 2, 1],
        "Education": ["PhD", "Basic", "Graduation", "Master"], "Marital_Status": ["Married", "Single", "Married", "Widow"],
        "Response": [1, 0, 1, 0], "Score": [0.8, 0.05, 0.4, 0.02], "Decile": [1, 9, 3, 10],
        "Segment": ["Big spenders", "Lapsed", "Big spenders", "Lapsed"],
        "TopReasons": ["total spend (2,000)", "", "wine spend (400)", ""],
    })


@pytest.fixture
def wh(customers) -> Warehouse:
    return Warehouse(customers)


def tool_call(sql: str, call_id: str = "c1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name="run_sql", arguments=json.dumps({"sql": sql})))


class FakeLLM:
    """Replays scripted assistant turns: each is either a list of SQL strings (tool calls) or a final text."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the agent keeps appending
        turn = self.turns.pop(0)
        if isinstance(turn, list):
            msg = SimpleNamespace(content="", tool_calls=[tool_call(sql, f"c{i}") for i, sql in enumerate(turn)])
        else:
            msg = SimpleNamespace(content=turn, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def fake_llm():
    return FakeLLM
