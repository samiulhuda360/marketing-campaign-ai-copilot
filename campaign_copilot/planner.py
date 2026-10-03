"""AI campaign planner: turn segment numbers and the model's reasons into a ranked plan with
offers and message copy, as structured JSON.

The numbers (segment sizes, scores, expected responders for a contact budget) are computed
here in code; the language model only chooses priorities and writes the strategy and copy,
and must return JSON matching PLAN_SCHEMA. Expected responders are the sum of the predicted
probabilities of the customers contacted, which is what the model's scores mean.
"""

from __future__ import annotations

import json
from collections import Counter

from .config import Settings
from .warehouse import Warehouse

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "segments"],
    "properties": {
        "summary": {"type": "string", "description": "2-3 sentences: the overall plan and why"},
        "segments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["segment", "priority", "why", "offer", "channel", "subject", "message"],
            "properties": {
                "segment": {"type": "string"},
                "priority": {"type": "integer", "description": "1 = contact first"},
                "why": {"type": "string", "description": "one sentence citing the numbers given"},
                "offer": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "catalogue", "web", "in-store", "sms"]},
                "subject": {"type": "string", "description": "email subject or headline, max 60 characters"},
                "message": {"type": "string", "description": "message body, max 50 words"},
            }}},
    },
}

SYSTEM = """You are a senior CRM marketing strategist. Using ONLY the facts provided, write a campaign plan.
Rank segments by expected value (predicted response and spend). Match the channel to how each segment buys.
Offers must be plausible for a grocery superstore. Cite at least one provided number in each 'why'.
Return JSON only, matching the schema."""


def facts(wh: Warehouse, budget: int) -> dict:
    """Everything the planner may use, computed from the data."""
    seg = wh.query("SELECT * FROM segments", 20).to_dict(orient="records")
    reasons = wh.query("SELECT Segment, TopReasons FROM customers WHERE Decile <= 3", 5000)
    for s in seg:
        texts = reasons[reasons["Segment"] == s["Segment"]]["TopReasons"]
        common = Counter(r.split(" (")[0] for t in texts for r in t.split("; ") if r)
        s["top_model_reasons"] = [name for name, _ in common.most_common(3)]
        channels = wh.query(f"""SELECT ROUND(AVG(NumWebPurchases),1) AS web, ROUND(AVG(NumCatalogPurchases),1) AS catalogue,
                                       ROUND(AVG(NumStorePurchases),1) AS store, ROUND(AVG(NumDealsPurchases),1) AS deals
                                FROM customers WHERE Segment = '{s['Segment'].replace("'", "''")}'""", 1).iloc[0].to_dict()
        s["avg_purchases_by_channel"] = channels
    top = wh.query("SELECT COUNT(*) AS n, SUM(Score) AS expected "
                   f"FROM (SELECT Score FROM customers ORDER BY Score DESC LIMIT {int(budget)})", 1).iloc[0]
    everyone = wh.query("SELECT COUNT(*) AS n, SUM(Score) AS expected FROM customers", 1).iloc[0]
    return {"segments": seg, "contact_budget": int(budget),
            "expected_responders_if_top_scored_contacted": round(float(top["expected"]), 1),
            "expected_responders_if_random_contacted": round(float(everyone["expected"]) * budget / float(everyone["n"]), 1)}


def plan(cfg: Settings, wh: Warehouse, goal: str, budget: int = 300, client=None) -> dict:
    if client is None:
        import openai

        client = openai.OpenAI(api_key=cfg.llm_api_key or "not-needed", base_url=cfg.llm_base_url, timeout=90)
    f = facts(wh, budget)
    user = f"Campaign goal: {goal}\n\nFacts (JSON):\n{json.dumps(f, default=str)}"
    extra = {"reasoning": {"effort": "low"}} if "openrouter.ai" in cfg.llm_base_url else {}  # reasoning models: think briefly
    resp = client.chat.completions.create(
        model=cfg.llm_model, temperature=0.4, max_tokens=max(cfg.max_tokens, 4000), extra_body=extra or None,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "campaign_plan", "strict": True, "schema": PLAN_SCHEMA}})
    choice = resp.choices[0]
    raw = choice.message.content or ""
    if "{" not in raw:
        reason = "it ran out of tokens while reasoning" if getattr(choice, "finish_reason", "") == "length" else "it returned no JSON"
        raise RuntimeError(f"The model did not return a plan ({reason}). Try again, or raise LLM_MAX_TOKENS.")
    out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    by_name = {s["Segment"]: s for s in f["segments"]}
    lookup = {name.lower(): name for name in by_name}  # tolerate case differences in segment names
    for s in out.get("segments", []):
        s["segment"] = lookup.get(str(s.get("segment", "")).strip().lower(), s.get("segment"))
    out["segments"] = sorted([s for s in out.get("segments", []) if s.get("segment") in by_name], key=lambda s: s["priority"])
    if not out["segments"]:
        raise RuntimeError("The model's plan did not name any of the real segments. Try again.")
    for s in out["segments"]:
        s["facts"] = {k: by_name[s["segment"]][k] for k in ("customers", "avg_score", "past_response_rate", "avg_spend")}
    out["budget"] = f["contact_budget"]
    out["expected_responders"] = {"top_scored": f["expected_responders_if_top_scored_contacted"],
                                  "random": f["expected_responders_if_random_contacted"]}
    out["model"] = cfg.llm_model
    return out
