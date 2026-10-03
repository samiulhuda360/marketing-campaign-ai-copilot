# Campaign Copilot: AI agent + ML for marketing campaign targeting

![CI](https://github.com/samiulhuda360/marketing-campaign-ai-copilot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-server-c2410c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Predict which customers will respond to a marketing offer, explain every prediction, and ask an AI analyst
agent questions about the customer base in plain English.** A propensity model with SHAP explanations, a
tool-calling LLM agent that writes and runs SQL on DuckDB, an AI campaign planner with structured output,
and an MCP server so Claude, Cursor or ChatGPT can use it all as tools.

![Dashboard: KPIs, the AI analyst answering "How many lapsed customers are still in the top 3 deciles?" with the SQL it ran, segments ranked by predicted response, and the best prospects with SHAP reasons](docs/screenshots/dashboard.png)

| | |
|---|---|
| **Response model** | Random forest chosen by cross-validation over logistic regression and gradient boosting. **Test ROC-AUC 0.89**; contacting the top 20% of customers reaches **70% of all responders** (3.5x random) |
| **Honest scores** | Every customer is scored out-of-fold by a model that never saw them; probabilities are calibrated: **338 expected vs 334 actual responders** |
| **Explainable** | SHAP reasons for every customer, e.g. *"total spend (2,077); income (92,859); meat spend (921)"* |
| **AI analyst agent** | LLM tool calling + DuckDB text-to-SQL with read-only guardrails and self-correction. **12/12 correct** on a gold-SQL evaluation, ~2 s per question, with GPT-4.1-mini **and** a free open-weight model |
| **AI campaign planner** | Structured JSON output (JSON Schema): priorities, offers, channels and message copy, grounded in computed segment numbers |
| **MCP server** | 5 tools + a schema resource over the Model Context Protocol (FastMCP), usable from any MCP client |

## Results

**Model comparison** (2,236 customers, 14.9% responded; stratified 5-fold CV on 80%, untouched 20% test set):

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Baseline (always "no") | 0.500 | 0.500 | 0.150 |
| Logistic regression | 0.846 | 0.895 | 0.635 |
| **Random forest** (selected on CV) | **0.863** | 0.893 | 0.545 |
| Gradient boosting | 0.860 | 0.876 | 0.545 |

Accuracy is not reported on purpose: answering "no" for everyone already scores 85%. At the decision threshold
(chosen on out-of-fold training predictions, never on the test set) the model finds **78% of responders with 51%
precision**, against a 15% base rate.

| Who to contact first | What drives a response |
|---|---|
| ![Cumulative gains chart: the top 20% of customers by score contain 70% of responders](docs/figures/gains.png) | ![Permutation importance: days since last purchase matters most](docs/figures/importance.png) |

![Response rate by days since last purchase, total spend and catalogue purchases](docs/figures/response-by-segment.png)

**AI analyst evaluation** ([`eval/results.md`](eval/results.md)): 12 business questions with gold SQL, scored by
execution accuracy (the agent's data must contain the gold values; correct rounding such as 14.9% for 0.1494 counts).

| LLM | Execution accuracy | Median time |
|---|---|---|
| `openai/gpt-4.1-mini` | **12/12** | 2.0 s |
| `qwen/qwen3.8-27b:free` (open-weight, free tier) | **12/12** | 3.2 s |

## The AI campaign planner

Give it a goal and a contact budget. Segment sizes, scores and expected responders are computed in code; the LLM
only ranks segments and writes the strategy and copy, as JSON validated against a schema.

![Campaign plan for a premium wine and cheese launch: 3 prioritised segments with offers, channels and message copy; 143 expected responders from 300 contacts vs 45 at random](docs/screenshots/planner.png)

## How it works

```
superstore_data.csv ─► cleaning + feature engineering (tenure, total spend, channel shares)
        │
        ├─► model comparison (CV) ─► random forest ─► out-of-fold calibrated scores ─► SHAP reasons per customer
        ├─► k-means behavioural segments (named from their distinctive traits)
        ▼
  DuckDB (in memory, read-only, file and network access disabled)
        │
        ├─► AI analyst: LLM ⇄ run_sql tool, sees errors and retries, answers with the SQL it ran
        ├─► AI planner: facts computed in SQL ─► LLM ─► JSON-schema plan
        ├─► FastAPI + dashboard
        └─► MCP server: run_sql · segment_summary · top_customers · explain_customer · score_new_customer
```

**Guardrails for LLM-written SQL:** one statement only, `SELECT`/`WITH` only, a blocklist for write and side-effect
keywords, and DuckDB itself locked with `enable_external_access = false` and `lock_configuration = true`, so even a
cleverly written `SELECT` cannot read files or reach the network. Tests cover `DROP`, `COPY`, `ATTACH`, `INSTALL`,
`read_csv` and multi-statement injection.

## Run it

```bash
git clone https://github.com/samiulhuda360/marketing-campaign-ai-copilot && cd marketing-campaign-ai-copilot
pip install -e ".[dev]"
cp .env.example .env                      # add OPENROUTER_API_KEY, or point LLM_BASE_URL at Ollama / vLLM
python -m campaign_copilot train          # ~2 min: compare models, score + explain customers, segments
python -m campaign_copilot serve          # dashboard on http://127.0.0.1:8000
python -m campaign_copilot ask "Which segment should get a wine offer, and why?"
python -m campaign_copilot plan "Launch a premium wine range" --budget 300
python -m campaign_copilot eval           # agent accuracy on eval/questions.jsonl
python -m campaign_copilot report         # regenerate the charts
```

Free option: `LLM_MODEL=qwen/qwen3.8-27b:free` on OpenRouter scored 12/12 (rate-limited). Fully local: run Ollama and
set `LLM_BASE_URL=http://localhost:11434/v1`.

**Use it from Claude Desktop (MCP)**: add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "campaign-copilot": { "command": "python", "args": ["-m", "campaign_copilot", "mcp"], "cwd": "/path/to/marketing-campaign-ai-copilot" }
  }
}
```

Then ask Claude things like *"Which segment has the most high-scoring customers who haven't bought in 60 days?"*;
it calls `run_sql` and the other tools itself.

## Project layout

```
campaign_copilot/  data · model (training, calibration, SHAP, segments) · warehouse (DuckDB guardrails)
                   agent (tool-calling analyst) · planner (structured output) · mcp_server · api · evaluate · report
ui/index.html      dashboard (no build step)
eval/              questions.jsonl (gold SQL) · runs/ (one file per LLM) · results.md
tests/             21 tests: cleaning, SQL guardrails, agent loop and error recovery, planner, scorer, API (fake LLM, no key)
original/          the 2023 group notebook this project grew out of (see below)
```

## Background

This started as a 2023 university group assignment (Md. Mohidul Islam, Mohammad Atik Ibna Shams, Quazi Raihan
Shahriar, Md Samiul Huda, Milton Raj Bonghsi), kept unchanged in [`original/`](original/). Its tuned model reached
0.58 recall but only 0.11 precision on the test set, below the always-"no" baseline on accuracy, because it was tuned
for recall alone on an imbalanced target. This version rebuilds the modelling with imbalance-aware metrics,
leakage-free scoring and calibration, and adds the explanation, agent, planner and MCP layers.

**Data:** Superstore Marketing Campaign dataset (Kaggle): 2,240 customers, 22 columns. 4 rows are removed as data
errors (birth years before 1901, one income of 666,666).

## Tech stack

Python · scikit-learn · SHAP · pandas · DuckDB · OpenAI-compatible LLM APIs (OpenRouter, Ollama, vLLM) · tool calling ·
JSON Schema structured outputs · FastMCP (Model Context Protocol) · FastAPI · Matplotlib · pytest · Ruff · GitHub Actions

**Keywords:** AI agent, LLM agent, text-to-SQL, agentic analytics, tool calling, function calling, MCP server,
propensity modelling, customer response prediction, marketing analytics, CRM, explainable AI (XAI), SHAP,
probability calibration, customer segmentation, k-means, DuckDB, FastAPI, data science.

## Author

[Samiul Huda](https://github.com/samiulhuda360) · MSc Data Science · Auckland, New Zealand. MIT licence.
