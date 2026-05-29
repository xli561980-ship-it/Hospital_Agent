import uuid

import streamlit as st


def _load_env():
    # Optional: load .env if present (safe: .env is usually gitignored)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


_load_env()

from agent import GRAPH
from src.config.settings import env_flag, load_llm_settings


st.set_page_config(page_title="Hospital Guide Agent", page_icon="H", layout="centered")
st.title("Hospital Guide Agent")
st.caption("智能医院导诊与挂号辅助 Agent 解决方案原型")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[{"role": "user"|"assistant", "content": str}]


def _queue_prompt(text: str) -> None:
    st.session_state.pending_prompt = text


def _debug_summary(state: dict) -> dict:
    locations = []
    for item in (state.get("location_results") or [])[:3]:
        if not isinstance(item, dict):
            continue
        locations.append(
            {
                "service": item.get("service"),
                "building": item.get("building"),
                "floor": item.get("floor"),
                "room": item.get("room"),
            }
        )

    schedule = []
    for item in (state.get("schedule_window") or state.get("schedule_candidates") or [])[:3]:
        if not isinstance(item, dict):
            continue
        schedule.append({"day": item.get("day"), "items": (item.get("items") or [])[:3]})

    return {
        "intent": state.get("intent"),
        "intent_source": state.get("intent_source"),
        "current_phase": state.get("current_phase"),
        "action": state.get("action"),
        "department": state.get("department"),
        "is_emergency": state.get("is_emergency"),
        "triage_match_source": state.get("triage_match_source"),
        "location_results": locations,
        "schedule_summary": schedule,
    }


def _last_ai_text(state: dict) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "ai":
            return str(getattr(msg, "content", "") or "")
    return "（没有生成回复）"


def _seed_demo_conversation() -> None:
    demo = st.query_params.get("demo")
    if demo != "clarification" or st.session_state.get("demo_seeded") == demo:
        return
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4())
    prompts = [
        "我25岁 女，头疼挂什么科？",
        "不是突然的，没有呕吐，没有发热，也没有肢体无力。",
    ]
    last_state = {}
    for prompt in prompts:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        last_state = GRAPH.invoke(
            {"messages": [("human", prompt)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )
        st.session_state.chat_history.append({"role": "assistant", "content": _last_ai_text(last_state)})
    st.session_state.last_debug_summary = _debug_summary(last_state)
    st.session_state.demo_seeded = demo


_seed_demo_conversation()


with st.sidebar:
    st.subheader("项目说明")
    st.markdown(
        "- 当前是 Mock Demo / solution prototype。\n"
        "- 只支持科室推荐、院内位置查询、挂号与排班辅助。\n"
        "- 不提供疾病诊断、治疗建议、用药建议，也不判断病情严重程度。"
    )

    st.divider()
    st.subheader("快速测试")
    if st.button("头疼挂什么科？", use_container_width=True):
        _queue_prompt("我25岁 女，头疼挂什么科？")
    if st.button("抽血在几楼？", use_container_width=True):
        _queue_prompt("抽血在几楼？")
    if st.button("想看周一值班医生", use_container_width=True):
        _queue_prompt("需要，帮我看周一值班医生")
    if st.button("问诊断和用药", use_container_width=True):
        _queue_prompt("我是不是脑梗？吃什么药？")

    st.divider()
    with st.expander("示例场景", expanded=False):
        st.markdown(
            "- 有限多轮澄清：宽泛症状先追问，再进入科室推荐。\n"
            "- 普通科室推荐：年龄、性别、症状槽位抽取。\n"
            "- 院内位置查询：从位置知识库返回楼层和路线。\n"
            "- 多轮号源查询：推荐科室后继续查询排班。\n"
            "- 医疗安全边界：拒绝诊断、治疗和用药建议。"
        )

    with st.expander("运行模式", expanded=False):
        settings = load_llm_settings()
        key_configured = bool(settings.google_api_key) if settings.provider == "gemini" else bool(settings.openai_api_key)
        st.write(f"LLM Provider: `{settings.provider}`")
        st.write(f"LLM Key: `{'已配置' if key_configured else '未配置，使用本地规则回退'}`")
        st.write(f"Intent LLM: `{'ON' if env_flag('USE_INTENT_LLM', True) else 'OFF'}`")
        st.write(f"Triage LLM: `{'ON' if env_flag('USE_TRIAGE_LLM', True) else 'OFF'}`")

    st.session_state.show_debug_summary = st.checkbox(
        "显示结构化摘要",
        value=st.session_state.get("show_debug_summary", False),
        help="只展示公开状态字段和工具结果摘要，不展示内部推理链。",
    )

    if st.session_state.get("last_debug_summary") and st.session_state.show_debug_summary:
        with st.expander("上一轮结构化摘要", expanded=False):
            st.json(st.session_state.last_debug_summary)

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pop("pending_prompt", None)
        st.session_state.pop("last_debug_summary", None)


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
                msgs = out.get("messages", []) or []
                last_ai = None
                for mm in reversed(msgs):
                    if getattr(mm, "type", "") == "ai":
                        last_ai = mm
                        break
                answer = str(getattr(last_ai, "content", "")) if last_ai else "（没有生成回复）"
                st.session_state.last_debug_summary = _debug_summary(out)
            except Exception as e:
                hint = "提示：请检查 LLM Provider 与对应 Key 是否已在环境变量中配置；未配置时可关闭 LLM 开关走本地规则回退。"
                answer = f"运行出错：{e}\n\n{hint}"
                st.session_state.last_debug_summary = {}

        st.markdown(answer)
        if st.session_state.get("show_debug_summary") and st.session_state.get("last_debug_summary"):
            with st.expander("结构化摘要", expanded=False):
                st.json(st.session_state.last_debug_summary)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
