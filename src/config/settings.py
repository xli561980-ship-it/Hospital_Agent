from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    timeout_seconds: float
    openai_api_key: str
    openai_model: str
    google_api_key: str
    gemini_model: str


def env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_agent_now() -> datetime:
    """
    Use real current time by default, with deterministic demos/evals via env.
    Accepted examples: 2026-05-11, 2026-05-11T09:00:00
    """
    raw = os.getenv("HOSPITAL_AGENT_NOW", "").strip()
    if raw:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                if fmt == "%Y-%m-%d":
                    return dt.replace(hour=9, minute=0, second=0)
                return dt
            except ValueError:
                pass
    return datetime.now()


def load_llm_settings() -> LLMSettings:
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
    except Exception:
        timeout = 8.0
    return LLMSettings(
        provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        timeout_seconds=timeout,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip(),
    )

