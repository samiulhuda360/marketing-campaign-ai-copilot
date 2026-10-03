import json

from fastapi.testclient import TestClient

from campaign_copilot import api, warehouse
from campaign_copilot.config import Settings


def client(monkeypatch, wh, tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"rows": 4}), encoding="utf-8")
    cfg = Settings(artifacts=tmp_path, llm_api_key="t")
    monkeypatch.setattr(api, "settings", lambda: cfg)
    monkeypatch.setattr(warehouse, "get", lambda _cfg: wh)
    return TestClient(api.app)


def test_overview_segments_and_customers(monkeypatch, wh, tmp_path):
    c = client(monkeypatch, wh, tmp_path)
    body = c.get("/api/overview").json()
    assert body["metrics"]["rows"] == 4 and body["segments"][0]["Segment"] == "Big spenders"
    top = c.get("/api/customers/top?limit=2").json()
    assert [r["Id"] for r in top] == [1, 3]
    assert c.get("/api/customers/1").json()["Score"] == 0.8
    assert c.get("/api/customers/999").status_code == 404


def test_segment_filter_is_not_injectable(monkeypatch, wh, tmp_path):
    c = client(monkeypatch, wh, tmp_path)
    assert c.get("/api/customers/top", params={"segment": "x' OR '1'='1"}).json() == []


def test_llm_failures_become_readable_errors(monkeypatch, wh, tmp_path):
    c = client(monkeypatch, wh, tmp_path)

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("402 insufficient credits")

    monkeypatch.setattr("campaign_copilot.agent.Analyst", Broken)
    r = c.post("/api/ask", json={"question": "How many customers?"})
    assert r.status_code == 502 and "402" in r.json()["detail"]


def test_dashboard_is_served(monkeypatch, wh, tmp_path):
    assert "Campaign" in client(monkeypatch, wh, tmp_path).get("/").text
