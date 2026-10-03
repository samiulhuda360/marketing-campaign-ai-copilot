import json

import pytest

from campaign_copilot import data, planner
from campaign_copilot.agent import Analyst
from campaign_copilot.config import ROOT, Settings
from campaign_copilot.evaluate import correct
from campaign_copilot.warehouse import UnsafeQuery

CFG = Settings(llm_api_key="test")


# ---------- data ----------
def test_cleaning_removes_impossible_rows_and_adds_features():
    d = data.load(ROOT / "data" / "superstore_data.csv")
    assert len(d) == 2236 and d["Age"].max() < 100 and d["Income"].max() < 200_000
    assert {"TotalSpend", "TenureDays", "CatalogShare"} <= set(d.columns)
    assert "Id" not in data.features(d) and "Response" not in data.features(d)


# ---------- warehouse guardrails ----------
def test_select_queries_work(wh):
    assert wh.query("SELECT COUNT(*) AS n FROM customers").iloc[0]["n"] == 4
    assert len(wh.query("SELECT * FROM segments")) == 2


@pytest.mark.parametrize("sql", ["DROP TABLE customers", "DELETE FROM customers", "SELECT 1; DROP TABLE customers",
                                 "COPY customers TO 'x.csv'", "INSTALL httpfs", "ATTACH 'other.db'", "SET threads=1"])
def test_writes_and_side_effects_are_refused(wh, sql):
    with pytest.raises(UnsafeQuery):
        wh.query(sql)


def test_file_access_is_disabled_even_inside_a_select(wh):
    with pytest.raises(Exception, match="(?i)disabled|permission"):
        wh.query("SELECT * FROM read_csv('pyproject.toml')")


def test_words_inside_strings_are_not_mistaken_for_commands(wh):
    assert len(wh.query("SELECT * FROM customers WHERE Segment = 'Lapsed' OR TopReasons = 'drop set'")) == 2


# ---------- agent ----------
def test_agent_runs_sql_then_answers(wh, fake_llm):
    llm = fake_llm([["SELECT Segment, AVG(Score) AS s FROM customers GROUP BY Segment ORDER BY s DESC"], "Big spenders score highest (60%)."])
    ans = Analyst(CFG, wh, client=llm).ask("Which segment scores highest?")
    assert ans.answer.startswith("Big spenders") and len(ans.steps) == 1 and ans.steps[0].rows[0]["Segment"] == "Big spenders"
    tool_msg = llm.seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "Big spenders" in tool_msg["content"]


def test_agent_sees_its_sql_error_and_recovers(wh, fake_llm):
    llm = fake_llm([["SELECT nope FROM customers"], ["SELECT COUNT(*) AS n FROM customers"], "There are 4 customers."])
    ans = Analyst(CFG, wh, client=llm).ask("How many customers?")
    assert ans.steps[0].error and not ans.steps[1].error and ans.answer == "There are 4 customers."
    assert "error" in llm.seen[1]["messages"][-1]["content"]


def test_agent_refuses_destructive_sql_and_tells_the_model(wh, fake_llm):
    llm = fake_llm([["DROP TABLE customers"], "I can only read data."])
    ans = Analyst(CFG, wh, client=llm).ask("Delete everything")
    assert "read-only" in ans.steps[0].error and wh.query("SELECT COUNT(*) AS n FROM customers").iloc[0]["n"] == 4


def test_agent_stops_at_the_step_limit(wh, fake_llm):
    llm = fake_llm([["SELECT 1 AS x"]] * 10)
    ans = Analyst(Settings(llm_api_key="t", max_agent_steps=3), wh, client=llm).ask("loop")
    assert len(ans.steps) == 3 and "step limit" in ans.answer


# ---------- planner ----------
def test_planner_returns_structured_plan_with_computed_numbers(wh, fake_llm):
    plan_json = json.dumps({"summary": "Target big spenders first.", "segments": [
        {"segment": "Lapsed", "priority": 2, "why": "Low scores (3.5%).", "offer": "Come-back voucher", "channel": "email",
         "subject": "We miss you", "message": "Here is 10% off."},
        {"segment": "Big spenders", "priority": 1, "why": "Highest score (60%).", "offer": "Wine tasting", "channel": "catalogue",
         "subject": "Your private tasting", "message": "Join us."},
        {"segment": "Invented segment", "priority": 3, "why": "x", "offer": "x", "channel": "sms", "subject": "x", "message": "x"}]})
    out = planner.plan(CFG, wh, "Sell more wine", budget=2, client=fake_llm([plan_json]))
    assert [s["segment"] for s in out["segments"]] == ["Big spenders", "Lapsed"]       # sorted, invented segment dropped
    assert out["segments"][0]["facts"]["customers"] == 2
    assert out["expected_responders"]["top_scored"] == pytest.approx(1.2)               # 0.8 + 0.4, computed in code


# ---------- evaluation scorer ----------
def test_scorer_accepts_correct_rounding_and_percentages():
    assert correct([{"avg": 0.1494}], [], "The share is 14.9%.")
    assert correct([{"avg": 60209.68}], [], "Average income was $60,210.")
    assert correct([{"Segment": "Lapsed"}], [{"Segment": "Lapsed"}], "")
    assert not correct([{"avg": 0.1494}], [], "The share is 16.2%.")
    assert correct([{"Response": 1, "avg": 5.0}], [], "Average 5.0", label_columns=("Response",))
