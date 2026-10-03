"""Command line: `python -m campaign_copilot <command>`.

    train     train and compare models, score and explain every customer, build segments
    report    write the evaluation charts to docs/figures
    ask       ask the AI analyst a question
    plan      ask the AI planner for a campaign plan
    eval      measure the analyst's accuracy on questions with known answers
    serve     run the dashboard and API on http://127.0.0.1:8000
    mcp       run the MCP server (stdio) for Claude Desktop, Cursor and other MCP clients
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import settings


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="campaign_copilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    sub.add_parser("report")
    pa = sub.add_parser("ask")
    pa.add_argument("question", nargs="+")
    pp = sub.add_parser("plan")
    pp.add_argument("goal", nargs="+")
    pp.add_argument("--budget", type=int, default=300)
    pe = sub.add_parser("eval")
    pe.add_argument("--rescore", action="store_true", help="re-score the saved run without model calls")
    ps = sub.add_parser("serve")
    ps.add_argument("--port", type=int, default=8000)
    sub.add_parser("mcp")
    a = p.parse_args(argv)
    cfg = settings()

    if a.cmd == "train":
        from .model import train

        m = train(cfg)
        print(json.dumps({k: v for k, v in m.items() if k != "models"}, indent=2))
    elif a.cmd == "report":
        from .report import main as report

        report()
    elif a.cmd == "ask":
        from . import warehouse
        from .agent import Analyst

        ans = Analyst(cfg, warehouse.get(cfg)).ask(" ".join(a.question))
        for s in ans.steps:
            print(f"-- SQL{' (error: ' + s.error + ')' if s.error else ''}\n{s.sql}\n")
        print(ans.answer)
    elif a.cmd == "plan":
        from . import planner, warehouse

        print(json.dumps(planner.plan(cfg, warehouse.get(cfg), " ".join(a.goal), a.budget), indent=2))
    elif a.cmd == "eval":
        from .evaluate import main as evaluate

        evaluate(["--rescore"] if a.rescore else [])
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("campaign_copilot.api:app", host="127.0.0.1", port=a.port)
    elif a.cmd == "mcp":
        from .mcp_server import main as mcp

        mcp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
