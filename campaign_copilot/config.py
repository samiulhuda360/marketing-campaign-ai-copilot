"""Settings from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    data_csv: Path = field(default_factory=lambda: Path(_env("COPILOT_DATA", str(ROOT / "data" / "superstore_data.csv"))))
    artifacts: Path = field(default_factory=lambda: Path(_env("COPILOT_ARTIFACTS", str(ROOT / "artifacts"))))
    # Any OpenAI-compatible endpoint with tool calling: OpenRouter, OpenAI, or a local Ollama / vLLM server.
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", _env("OPENROUTER_API_KEY")))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-4.1-mini"))
    max_agent_steps: int = 6
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "900")))
    max_rows: int = 50


def settings() -> Settings:
    return Settings()
