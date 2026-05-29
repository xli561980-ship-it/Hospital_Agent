from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from agent import GRAPH, _get_llm, _llm_content_to_text  # noqa: E402


def _configured_model() -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "")
    return os.getenv("OPENAI_MODEL", "")


def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    has_google_key = bool(os.getenv("GOOGLE_API_KEY"))
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    timeout = os.getenv("LLM_TIMEOUT_SECONDS", "20")

    print(f"provider={provider}")
    print(f"model={_configured_model()}")
    print(f"timeout={timeout}")
    print(f"google_key_set={has_google_key}")
    print(f"openai_key_set={has_openai_key}")

    llm = _get_llm()
    model_reply = llm.invoke([HumanMessage(content="请只回复 OK")])
    print("model_smoke=", _llm_content_to_text(getattr(model_reply, "content", None)).strip())

    out = GRAPH.invoke(
        {"messages": [("human", "我25岁 女，头疼挂什么科？")]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    ai_messages = [m for m in (out.get("messages") or []) if getattr(m, "type", "") == "ai"]
    answer = str(getattr(ai_messages[-1], "content", "")) if ai_messages else ""

    print(f"intent_source={out.get('intent_source')}")
    print(f"slot_extract_source={out.get('slot_extract_source')}")
    print(f"triage_match_source={out.get('triage_match_source')}")
    print(f"department={out.get('department')}")
    print("agent_answer=")
    print(answer)


if __name__ == "__main__":
    main()
