"""The AI analyst: a tool-calling LLM that answers questions by writing SQL over the scored
customer base, reading the results (or the error) and trying again, then explaining.

Tools: `run_sql` (read-only SELECT on DuckDB). The schema, with what each column means, is in
the system prompt, so the model rarely needs to explore. Every query it ran is returned
with the answer, so a person can check the numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .config import Settings
from .warehouse import UnsafeQuery, Warehouse

SYSTEM = """You are a marketing data analyst for a grocery superstore. Answer the user's question using ONLY
numbers you obtained by calling run_sql on the database described below. Rules:
1. Always query before answering; never guess or invent numbers.
2. Use DuckDB SQL. Round money to whole dollars and rates to one decimal place as percentages.
3. 'Score' is the predicted chance of responding to the next offer; 'Response' is what happened last campaign.
4. If a query fails, read the error, fix the query and try again.
5. Finish with a short answer (2-5 sentences or a small markdown table), then one practical recommendation.

{schema}"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SELECT query on the customers table or segments view. Returns up to 50 rows as JSON.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string", "description": "A single SELECT or WITH query"}},
                       "required": ["sql"]},
    },
}]


@dataclass
class Step:
    sql: str
    rows: list[dict] = field(default_factory=list)
    error: str = ""


@dataclass
class AgentAnswer:
    question: str
    answer: str
    steps: list[Step]
    model: str
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class Analyst:
    def __init__(self, cfg: Settings, warehouse: Warehouse, client=None):
        if client is None:
            import openai

            client = openai.OpenAI(api_key=cfg.llm_api_key or "not-needed", base_url=cfg.llm_base_url, timeout=60)
        self.cfg, self.wh, self.client = cfg, warehouse, client

    def ask(self, question: str) -> AgentAnswer:
        started = time.time()
        messages = [{"role": "system", "content": SYSTEM.format(schema=self.wh.schema())}, {"role": "user", "content": question}]
        steps: list[Step] = []
        for _ in range(self.cfg.max_agent_steps):
            resp = self.client.chat.completions.create(model=self.cfg.llm_model, messages=messages, tools=TOOLS, temperature=0,
                                                       max_tokens=self.cfg.max_tokens)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                return AgentAnswer(question, (msg.content or "").strip(), steps, self.cfg.llm_model, round(time.time() - started, 1))
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for call in calls:
                step = self._run(call.function.arguments)
                steps.append(step)
                payload = {"error": step.error} if step.error else {"rows": step.rows}
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload, default=str)[:12000]})
        return AgentAnswer(question, "I could not finish the analysis within the step limit; the queries I ran are listed below.",
                           steps, self.cfg.llm_model, round(time.time() - started, 1))

    def _run(self, arguments: str) -> Step:
        try:
            sql = json.loads(arguments or "{}").get("sql", "")
        except json.JSONDecodeError:
            return Step("", error="Arguments were not valid JSON; send {\"sql\": \"SELECT ...\"}.")
        try:
            df = self.wh.query(sql, self.cfg.max_rows)
            return Step(sql, rows=json.loads(df.to_json(orient="records", double_precision=4)))
        except UnsafeQuery as exc:
            return Step(sql, error=str(exc))
        except Exception as exc:  # noqa: BLE001  DuckDB errors go back to the model so it can fix the query
            return Step(sql, error=str(exc).splitlines()[0][:400])
