import os
import uuid

import streamlit as st

def _load_env():
    # Optional: load .env if present (safe: .env is gitignored)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


_load_env()

from agent import GRAPH

st.set_page_config(page_title="智能导诊台 - 多意图识别驱动", page_icon="🏥", layout="centered")
st.title("智能导诊台 - 多意图识别驱动")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[{"role": "user"|"assistant", "content": str}]


def _queue_prompt(text: str) -> None:
    st.session_state.pending_prompt = text


with st.sidebar:
    st.subheader("快速测试")
    if st.button("头疼挂什么科？", use_container_width=True):
        _queue_prompt("我25岁 女，头疼挂什么科？")
    if st.button("抽血在几楼？", use_container_width=True):
        _queue_prompt("抽血在几楼？")
    if st.button("想看周一值班医生", use_container_width=True):
        _queue_prompt("需要，帮我看周一值班医生")

    st.divider()
    st.caption("环境变量（可选）")
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        st.code(
            "\n".join(
                [
                    "export LLM_PROVIDER=gemini",
                    "export GOOGLE_API_KEY=...",
                    "export GEMINI_MODEL=gemini-1.5-flash",
                ]
            )
        )
    else:
        st.code(
            "\n".join(
                [
                    "export LLM_PROVIDER=openai",
                    "export OPENAI_API_KEY=...",
                    "export OPENAI_MODEL=gpt-4o-mini",
                ]
            )
        )
    if st.button("清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pop("pending_prompt", None)


for m in st.session_state.chat_history:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])


prompt = st.chat_input("请输入你的问题，例如：头疼挂什么科？/ 抽血在哪？")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                out = GRAPH.invoke(
                    {"messages": [("human", prompt)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                # get last AI message
                msgs = out.get("messages", []) or []
                last_ai = None
                for mm in reversed(msgs):
                    if getattr(mm, "type", "") == "ai":
                        last_ai = mm
                        break
                answer = str(getattr(last_ai, "content", "")) if last_ai else "（没有生成回复）"
            except Exception as e:
                hint = (
                    "提示：如果你希望使用 Gemini 意图分类，请设置 LLM_PROVIDER=gemini 且配置 GOOGLE_API_KEY。"
                    if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "gemini"
                    else "提示：如果你希望使用 OpenAI 意图分类，请设置 LLM_PROVIDER=openai 且配置 OPENAI_API_KEY。"
                )
                answer = f"运行出错：{e}\n\n{hint}"

        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
