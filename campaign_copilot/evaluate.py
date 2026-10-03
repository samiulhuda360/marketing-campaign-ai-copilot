"""Measure the AI analyst on questions with known answers (execution accuracy).

    python -m campaign_copilot eval              # run the agent (LLM_MODEL) on every question
    python -m campaign_copilot eval --rescore    # re-score all saved runs, no model calls

Each model's run is saved to eval/runs/<model>.json; eval/results.md compares all saved runs.

For each question in eval/questions.jsonl the gold SQL is run. The agent is correct when every
value in the gold result is found in what the agent produced: the rows its own queries returned
or the numbers in its final answer. Numbers match when the agent's value is a correct rounding
of the gold value (the agent is told to show money in whole dollars and rates as percentages
with one decimal), so 14.9% matches 0.1494 and $60,210 matches 60,209.68. Text matches exactly,
ignoring case. Grouping labels listed in a question's `label_columns` are not scored. This checks the data behind the answer, not its wording.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time

from . import warehouse
from .agent import Analyst
from .config import ROOT, settings

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _decimals(v: float, text: str | None = None) -> int:
    s = text if text is not None else repr(float(v))
    s = s.replace(",", "")
    if "." not in s or s.endswith(".0") and text is None:
        return 0
    return len(s.split(".")[1])


def _candidates(rows: list[dict], answer: str) -> tuple[list[tuple[float, int]], set[str]]:
    nums, texts = [], set()
    for row in rows:
        for v in row.values():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                nums.append((float(v), _decimals(v)))
            else:
                texts.add(str(v).strip().lower())
    for m in NUMBER.findall(answer):
        try:
            nums.append((float(m.replace(",", "")), _decimals(0, m)))
        except ValueError:
            pass
    return nums, texts


def _matches(gold: float, nums: list[tuple[float, int]]) -> bool:
    for v, dec in nums:
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if abs(v - gold) <= tol or abs(v / 100 - gold) <= tol / 100:  # plain, or shown as a percentage
            return True
    return False


def correct(gold_rows: list[dict], agent_rows: list[dict], answer: str, label_columns: tuple = ()) -> bool:
    """label_columns: grouping keys such as Response 0/1 that name a row rather than answer the question."""
    nums, texts = _candidates(agent_rows, answer)
    for row in gold_rows:
        for col, v in row.items():
            if col in label_columns:
                continue
            if v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not _matches(float(v), nums):
                    return False
            elif str(v).strip().lower() not in texts and str(v).strip().lower() not in answer.lower():
                return False
    return True


def _with_retry(fn, tries: int = 4):
    """Rate limits (HTTP 429) are a provider condition, not a wrong answer: wait and retry."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) or i == tries - 1:
                raise
            time.sleep(20 * (i + 1))


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "_", model.lower()).strip("_")


def run(model_rows: list[dict] | None, cfg, wh, questions: list[dict]) -> list[dict]:
    """Score saved rows again (model_rows) or ask the live agent (model_rows is None)."""
    saved = {r["id"]: r for r in model_rows} if model_rows is not None else None
    analyst = Analyst(cfg, wh) if saved is None else None
    rows = []
    for q in questions:
        gold = json.loads(wh.query(q["gold_sql"], 50).to_json(orient="records", double_precision=6))
        if saved is not None:
            r = dict(saved[q["id"]])
        else:
            time.sleep(float(os.getenv("EVAL_PAUSE", "0")))  # free model tiers allow ~20 requests a minute
            t = time.time()
            try:
                ans = _with_retry(lambda question=q["question"]: analyst.ask(question))
                r = {"id": q["id"], "question": q["question"], "queries": len(ans.steps), "errors_recovered": sum(1 for s in ans.steps if s.error),
                     "seconds": round(time.time() - t, 1), "answer": ans.answer, "sql": [s.sql for s in ans.steps],
                     "rows": [row for s in ans.steps for row in s.rows][:200]}
            except Exception as exc:  # noqa: BLE001
                r = {"id": q["id"], "question": q["question"], "queries": 0, "errors_recovered": 0, "seconds": round(time.time() - t, 1),
                     "answer": f"ERROR {exc}"[:200], "sql": [], "rows": []}
            print(f"{q['id']} {'?'} {r['seconds']:5.1f}s  {q['question']}")
        r["correct"] = not r["answer"].startswith("ERROR") and correct(gold, r.get("rows", []), r["answer"], tuple(q.get("label_columns", ())))
        rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    cfg = settings()
    wh = warehouse.get(cfg)
    questions = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    runs_dir = ROOT / "eval" / "runs"
    runs_dir.mkdir(exist_ok=True)
    if "--rescore" not in argv:
        rows = run(None, cfg, wh, questions)
        (runs_dir / f"{_slug(cfg.llm_model)}.json").write_text(json.dumps({"model": cfg.llm_model, "rows": rows}, indent=2), encoding="utf-8")
    summary, detail = [], []
    for path in sorted(runs_dir.glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = run(saved["rows"], cfg, wh, questions)
        saved["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
        saved["rows"] = rows
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        med = statistics.median(r["seconds"] for r in rows)
        summary.append(f"| `{saved['model']}` | **{saved['accuracy']:.0%}** ({sum(r['correct'] for r in rows)}/{len(rows)}) | {med:.1f} s |")
        detail += ["", f"### `{saved['model']}`", "", "| | Question | Agent's answer (first sentence) | Queries | Seconds |", "|---|---|---|---|---|"]
        for r in rows:
            first = r["answer"].replace("\n", " ").split(". ")[0].replace("|", "/")[:140]
            detail.append(f"| {'✓' if r['correct'] else '✗'} | {r['question']} | {first} | {r['queries']} | {r['seconds']} |")
        print(f"{saved['model']}: execution accuracy {saved['accuracy']:.0%}")
    lines = ["# AI analyst evaluation", "", f"{len(questions)} business questions with gold SQL (`eval/questions.jsonl`), scored by execution "
             "accuracy (see `campaign_copilot/evaluate.py`).", "", "| Model | Execution accuracy | Median time per question |",
             "|---|---|---|", *summary, *detail]
    (ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote eval/results.md")


if __name__ == "__main__":
    main()
