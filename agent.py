from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config.settings import env_flag, load_agent_now, load_llm_settings
from src.guardrails.medical_safety import (
    compliant_replacement,
    contains_forbidden_medical,
    strip_forbidden_tech,
)
from src.schemas.agent_state import Action, Intent, Phase, State
from tools import get_doctor_schedule, search_location


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mock_data"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _load_now() -> datetime:
    return load_agent_now()


NOW = _load_now()


def _format_now_text() -> str:
    return f"{NOW:%Y-%m-%d}（{WEEKDAYS_CN[NOW.weekday()]}）"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


TRIAGE_RULES_RAW = _load_json(DATA_DIR / "triage_rules.json")
TRIAGE_RULES: list[dict] = (
    TRIAGE_RULES_RAW
    if isinstance(TRIAGE_RULES_RAW, list)
    else list((TRIAGE_RULES_RAW or {}).get("rules") or [])
)
LOCATIONS_RAW = _load_json(DATA_DIR / "locations.json")
LOCATION_ITEMS: list[dict] = list((LOCATIONS_RAW or {}).get("locations") or []) if isinstance(LOCATIONS_RAW, dict) else []
ROUTING_KNOWLEDGE = _load_json(DATA_DIR / "routing_knowledge.json")


def _knowledge_list(section: str, key: str) -> list[str]:
    raw = ((ROUTING_KNOWLEDGE or {}).get(section) or {}).get(key, [])
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if str(x).strip()]


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(p and p in text for p in phrases)


def _contains_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns if p)


def _norm_match_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _char_chunks(text: str, size: int = 4) -> set[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", _norm_match_text(text))
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def _location_doc_text(item: dict) -> str:
    bits = [
        str(item.get("service") or ""),
        " ".join(str(x) for x in (item.get("aliases") or [])),
        str(item.get("building") or ""),
        str(item.get("floor") or ""),
        str(item.get("room") or ""),
    ]
    return " ".join(x for x in bits if x.strip())


def _location_kb_match_score(text: str) -> float:
    """
    Score whether a navigation-like query mentions an existing location item.
    This keeps location intent tied to locations.json instead of a separate hand-written service list.
    """
    q = _norm_match_text(text)
    if not q or not LOCATION_ITEMS:
        return 0.0

    best = 0.0
    generic = {"中心", "门诊", "科", "一区", "楼", "区", "全层", "医院", "门诊楼"}
    for item in LOCATION_ITEMS:
        doc = _location_doc_text(item)
        doc_n = _norm_match_text(doc)
        score = 0.0

        raw_phrases = [
            str(item.get("service") or ""),
            str(item.get("building") or ""),
            str(item.get("room") or ""),
            *[str(x) for x in (item.get("aliases") or [])],
        ]
        for raw in raw_phrases:
            parts = [p for p in re.split(r"[\s/,_\-()（）|]+", raw) if p]
            candidates = [raw, *parts]
            for phrase in candidates:
                pn = _norm_match_text(phrase)
                if len(pn) < 2 or pn in generic:
                    continue
                if pn in q:
                    score += min(len(pn), 10)

        q_chunks = _char_chunks(q, 4)
        doc_chunks = _char_chunks(doc_n, 4)
        if q_chunks and doc_chunks:
            score += min(len(q_chunks & doc_chunks), 6) * 1.5
        q_chunks3 = _char_chunks(q, 3)
        doc_chunks3 = _char_chunks(doc_n, 3)
        if q_chunks3 and doc_chunks3:
            score += min(len(q_chunks3 & doc_chunks3), 5) * 0.8

        best = max(best, score)
    return best


def _load_symptom_replacements() -> tuple[tuple[str, str], ...]:
    raw = (ROUTING_KNOWLEDGE or {}).get("symptom_replacements", [])
    out: list[tuple[str, str]] = []
    if not isinstance(raw, list):
        return tuple(out)
    for item in raw:
        if isinstance(item, list) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and item.get("from") and item.get("to"):
            out.append((str(item["from"]), str(item["to"])))
    return tuple(out)


def _load_derived_phrases() -> tuple[tuple[tuple[str, ...], str], ...]:
    raw = (ROUTING_KNOWLEDGE or {}).get("derived_phrases", [])
    out: list[tuple[tuple[str, ...], str]] = []
    if not isinstance(raw, list):
        return tuple(out)
    for item in raw:
        if not isinstance(item, dict):
            continue
        required = item.get("when_all_present")
        append = item.get("append")
        if isinstance(required, list) and append:
            out.append((tuple(str(x) for x in required if str(x).strip()), str(append)))
    return tuple(out)


def _get_llm():
    """
    Shared chat model: intent classification + final reply generation.
    """
    settings = load_llm_settings()
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.2,
            google_api_key=settings.google_api_key,
            request_timeout=settings.timeout_seconds,
            retries=0,
        )

    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


SYSTEM_PROMPT = (
    "你是协和医院的导诊机器人，只做导诊与院内指引。\n"
    "合规红线：严禁进行疾病诊断；严禁提供治疗/用药建议；严禁判断病情严重程度。\n"
    "你必须仅依据我提供的结构化数据（科室、建议、号源信息、位置指引）来回复，不得自行编造。\n"
    "风格要求：简明扼要、直接、客观；不输出问候语；不输出安慰/关怀语。\n"
    "信息不足时，可以为科室路由追问少量必要信息；追问只围绕年龄、性别、部位、主要表现、伴随表现，不做诊断或处置建议。\n"
)


def _last_user_text(state: State) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _normalize_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g in {"男", "male", "m"}:
        return "male"
    if g in {"女", "female", "f"}:
        return "female"
    return "any"


def _infer_gender(text: str) -> Optional[str]:
    t = text or ""
    if re.search(r"\b(male|m)\b", t, re.IGNORECASE) or re.search(
        r"(?<![男女])(男|男性|男士|男生|男孩|男童|男婴|男宝宝)(?!女)", t
    ):
        return "male"
    if re.search(r"\b(female|f)\b", t, re.IGNORECASE) or re.search(
        r"(?<![男女])(女|女性|女士|女生|女孩|女童|女婴|女宝宝)(?!男)", t
    ):
        return "female"
    return None


def _normalize_pregnancy_status(p: str) -> str:
    p = (p or "").strip().lower()
    if p in {"yes", "y", "true", "pregnant", "怀孕", "孕", "妊娠"}:
        return "yes"
    if p in {"no", "n", "false", "not_pregnant", "未孕", "未怀孕", "没怀孕", "没有怀孕"}:
        return "no"
    return "any"


def _infer_pediatric_age(text: str) -> Optional[int]:
    t = text or ""
    if re.search(r"(新生儿|刚出生|出生后|生后|满月内)", t):
        return 0
    if re.search(r"(婴儿|婴幼儿|宝宝|婴孩|幼婴)", t):
        return 1
    if re.search(r"(幼儿|儿童|孩子|小孩|患儿)", t):
        return 5
    return None


def _parse_age_gender_symptom(text: str) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    age = None
    gender = None
    pregnancy_status = None
    symptom = None

    m = re.search(r"(\d{1,3})\s*岁", text)
    if m:
        try:
            age = int(m.group(1))
        except Exception:
            pass
    if age is None:
        m2 = re.search(r"\b(\d{1,3})\b", text)
        if m2:
            try:
                cand = int(m2.group(1))
                if 0 < cand <= 120:
                    age = cand
            except Exception:
                pass
    if age is None:
        age = _infer_pediatric_age(text)

    gender = _infer_gender(text)

    if re.search(r"(未孕|未怀孕|没怀孕|没有怀孕|非孕期)", text):
        pregnancy_status = "no"
    elif re.search(r"(怀孕|孕期|妊娠|孕\d+\s*周|孕\d+\s*月)", text):
        pregnancy_status = "yes"

    # Only treat explicit "symptom label" patterns as structured symptom,
    # avoid capturing colloquial "感觉..." which often appears inside the real symptom description.
    m3 = re.search(r"(主要症状|症状|不舒服|主要是)\s*[:：]\s*(.+)$", text)
    if m3:
        symptom = m3.group(2).strip()

    if not symptom:
        cleaned = text
        cleaned = re.sub(r"\d{1,3}\s*岁", "", cleaned)
        cleaned = re.sub(r"\b\d{1,3}\b", "", cleaned)
        cleaned = re.sub(r"(男|女|male|female|m|f)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(我)?(怀孕了?|孕期|妊娠|未孕|未怀孕|没怀孕|没有怀孕|非孕期)", "", cleaned)
        cleaned = re.sub(r"(挂什么科|看什么科|去哪个科|去什么科|挂哪个科|导诊|咨询|我想问)", "", cleaned)
        cleaned = re.sub(r"^\s*(我|本人|患者|病人)\s*", "", cleaned)
        cleaned = re.sub(r"^[\s，,。！？?]+", "", cleaned)
        cleaned = cleaned.strip(" ，,。！？? ")
        if cleaned:
            symptom = cleaned

    if symptom:
        s = symptom.strip()
        # Treat "no other symptoms" statements as non-symptom
        if re.fullmatch(r"(没有其他症状|无其他症状|没有别的症状|无别的症状|无其他不适|没有其他不适|无不适|没有不适)", s):
            symptom = None
    return age, gender, pregnancy_status, symptom


def _classify_intent_regex(text: str) -> Intent:
    """
    Fast offline fallback when LLM intent is disabled or fails.
    """
    t = text or ""
    location_nav = _contains_any(t, _knowledge_list("intent_keywords", "location_nav"))
    location_service = _contains_any(t, _knowledge_list("intent_keywords", "location_service"))
    location_relation = _contains_any(t, _knowledge_list("intent_keywords", "location_relation"))
    explicit_location = _contains_any(t, _knowledge_list("intent_keywords", "explicit_location"))
    location_from_kb = _location_kb_match_score(t) >= 3.0
    if explicit_location or ((location_service or location_from_kb) and (location_nav or location_relation)):
        return "location_query"
    if _contains_any(t, _knowledge_list("intent_keywords", "other_keywords")) or _contains_pattern(
        t, _knowledge_list("intent_keywords", "other_patterns")
    ):
        return "other"
    return "triage_consult"


def _user_resets_triage(text: str) -> bool:
    return bool(re.search(r"(我还有别的症状|还有别的症状|换一个人看病|换个医生|换医生|重新分诊|重新推荐|换个科室)", text))


def _user_asks_schedule(text: str) -> bool:
    return bool(re.search(r"(什么时候有号|有号吗|啥时候|什么时候|未来|换一天|改天|怎么挂号|挂号|预约|排班|值班|医生|号源)", text))


def _needs_future_search(text: str) -> bool:
    return bool(re.search(r"(未来|换一天|改天|之后|明天|后天)", text))


def _infer_start_day(text: str) -> str:
    if re.search(r"(今天|今日|现在|这会儿|啥时候|什么时候)", text):
        return WEEKDAYS[NOW.weekday()]

    m = re.search(r"(周[一二三四五六日天]|星期[一二三四五六日天])", text)
    if m:
        cn = m.group(1)
        cn_map = {
            "周一": "Monday",
            "星期一": "Monday",
            "周二": "Tuesday",
            "星期二": "Tuesday",
            "周三": "Wednesday",
            "星期三": "Wednesday",
            "周四": "Thursday",
            "星期四": "Thursday",
            "周五": "Friday",
            "星期五": "Friday",
            "周六": "Saturday",
            "星期六": "Saturday",
            "周日": "Sunday",
            "星期日": "Sunday",
            "周天": "Sunday",
            "星期天": "Sunday",
        }
        return cn_map.get(cn, WEEKDAYS[NOW.weekday()])

    m2 = re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1).capitalize()

    return WEEKDAYS[NOW.weekday()]


SYMPTOM_REPLACEMENTS = _load_symptom_replacements()
DERIVED_PHRASES = _load_derived_phrases()


def _normalize_symptom_text(s: str) -> str:
    s = (s or "").strip()
    for a, b in SYMPTOM_REPLACEMENTS:
        s = s.replace(a, b)
    for required, append in DERIVED_PHRASES:
        if append and append not in s and all(part in s for part in required):
            s += append
    s = s.replace("疼", "痛")
    s = re.sub(r"\s+", "", s)
    return s


def _char_ngrams(s: str, n: int = 2) -> set[str]:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _keyword_match_score(query: str, keyword: str) -> float:
    if not keyword or len(keyword) < 2:
        return 0.0
    if keyword in query:
        return min(len(keyword), 14) * 2.0

    # Reward partial phrase overlap so colloquial text can match structured symptom phrases.
    kw_bigrams = _char_ngrams(keyword, 2)
    q_bigrams = _char_ngrams(query, 2)
    if not kw_bigrams or not q_bigrams:
        return 0.0
    overlap = len(kw_bigrams & q_bigrams)
    coverage = overlap / len(kw_bigrams)
    precision = overlap / len(q_bigrams)
    if coverage >= 0.45 or (len(keyword) <= 4 and coverage >= 0.5 and precision >= 0.08):
        return coverage * min(len(keyword), 12)
    return 0.0


def _safe_triage_advice(rule: dict) -> str:
    if bool(rule.get("is_emergency")):
        return "建议优先前往急诊分诊台，由现场医护确认就诊入口。"
    return "建议到对应专科门诊，由线下医生进一步评估。"


def _triage_candidate_payload(candidates: list[dict]) -> list[dict]:
    payload: list[dict] = []
    for c in candidates:
        r = c.get("rule") or {}
        if not isinstance(r, dict):
            continue
        payload.append(
            {
                "rule_id": str(c.get("rule_id") or ""),
                "symptoms": r.get("symptoms") or r.get("symptom_keywords") or [],
                "department": r.get("recommended_department") or r.get("department"),
                "is_emergency": bool(r.get("is_emergency")),
                "retrieval_score": round(float(c.get("retrieval_score", 0.0)), 4),
                "retrieval_matched_symptoms": c.get("matched_by_retrieval", []),
            }
        )
    return payload


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = re.sub(r"\s+", "", str(item or "")).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(str(item).strip())
    return out


def _ensure_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


COMMON_TRIAGE_DEPARTMENTS = [
    "急诊科",
    "急诊科-胸痛中心",
    "心血管内科",
    "呼吸与危重症医学科",
    "消化内科",
    "神经内科",
    "神经外科",
    "普外科急诊",
    "骨科-脊柱外科",
    "骨科-关节外科",
    "妇科",
    "产科",
    "儿科",
    "眼科",
    "耳鼻咽喉科",
    "口腔科",
    "皮肤科",
    "泌尿外科",
    "内分泌科",
    "肾脏内科",
    "血液内科",
    "风湿免疫科",
    "心理医学科",
    "全科/内科（综合门诊）",
]

DEPARTMENT_ALIASES = {
    "心内科": "心血管内科",
    "呼吸科": "呼吸与危重症医学科",
    "神内": "神经内科",
    "神经科": "神经内科",
    "普外": "普外科急诊",
    "骨科": "骨科-脊柱外科",
    "妇产科": "妇科",
    "耳鼻喉科": "耳鼻咽喉科",
    "口腔": "口腔科",
    "皮肤": "皮肤科",
    "心理科": "心理医学科",
    "急诊": "急诊科",
}


def _known_department_names() -> list[str]:
    depts: list[str] = list(COMMON_TRIAGE_DEPARTMENTS)
    for r in TRIAGE_RULES:
        if not isinstance(r, dict):
            continue
        dep = str(r.get("recommended_department") or r.get("department") or "").strip()
        if dep:
            depts.append(dep)
    try:
        schedules = _load_json(DATA_DIR / "doctor_schedules.json")
        if isinstance(schedules, list):
            depts.extend(str(x.get("department") or "").strip() for x in schedules if isinstance(x, dict))
        elif isinstance(schedules, dict):
            depts.extend(str(x).strip() for x in (schedules.get("departments") or {}).keys())
    except Exception:
        pass
    return _dedupe_keep_order([x for x in depts if x])


def _coerce_department_name(value: str | None) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw in DEPARTMENT_ALIASES:
        raw = DEPARTMENT_ALIASES[raw]
    known = _known_department_names()
    if raw in known:
        return raw
    for dep in known:
        if raw and (raw in dep or dep in raw):
            return dep
    return None


def _recent_user_texts(state: State, limit: int = 6) -> list[str]:
    out: list[str] = []
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            out.append(str(m.content))
        if len(out) >= limit:
            break
    return list(reversed(out))


def _merge_text_slot(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    old = str(existing or "").strip()
    cur = str(new or "").strip()
    if not old:
        return cur or None
    if not cur:
        return old
    old_n = _normalize_symptom_text(old)
    cur_n = _normalize_symptom_text(cur)
    if cur_n in old_n:
        return old
    if old_n in cur_n and len(cur_n) > len(old_n):
        return f"{old}；{cur}"
    return f"{old}；{cur}"


def _merge_state_list(state: State, key: str, values: list[str]) -> None:
    state[key] = _dedupe_keep_order(_ensure_str_list(state.get(key)) + [str(x) for x in values if str(x).strip()])


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


TRIAGE_INTERVIEW_LLM_SYSTEM = (
    "你是医院导诊系统的分诊访谈规划模块。你不是诊断医生，不能做疾病诊断、治疗或用药建议。\n"
    "你的任务是：根据用户多轮描述，内部生成可能的症状方向/候选科室，然后决定是继续追问还是已足够推荐就诊科室。\n"
    "必须遵守：\n"
    "1. candidate_departments 和 recommended_department 必须来自 allowed_departments。\n"
    "2. possible_conditions 只作为内部分诊方向，不要写成确诊；可为空。\n"
    "3. 如果有突发胸痛、明显呼吸困难、肢体无力、意识异常、大量出血等红旗表现，推荐急诊入口。\n"
    "4. 信息不足时，不要硬推荐科室；提出一个最能缩小候选范围的问题。\n"
    "5. 问题要利用用户已经否认的信息；用户说没别处疼，就不要重复问哪里疼。\n"
    "6. 仅输出 JSON，不要 Markdown。\n"
    "JSON 格式：{\"primary_symptom\": string|null, \"possible_conditions\": [string], "
    "\"candidate_departments\": [string], \"positive_findings\": [string], \"negative_findings\": [string], "
    "\"recommended_department\": string|null, \"is_emergency\": boolean, \"confidence\": 0到1, "
    "\"next_questions\": [string], \"reason\": string}"
)


def _triage_interview_llm(state: State) -> Optional[dict]:
    if not _use_triage_llm() or not state.get("symptom"):
        return None

    ranked = _rank_triage_candidates(
        age=state.get("age"),
        gender=state.get("gender"),
        pregnancy_status=state.get("pregnancy_status"),
        symptom=state.get("symptom"),
        limit=8,
    )
    candidate_rules = _triage_candidate_payload(ranked)
    allowed_departments = _known_department_names()
    context = {
        "age": state.get("age"),
        "gender": state.get("gender"),
        "pregnancy_status": state.get("pregnancy_status"),
        "current_symptom_summary": state.get("symptom"),
        "primary_symptom": state.get("primary_symptom"),
        "positive_findings": state.get("triage_positive_findings", []),
        "negative_findings": state.get("triage_negative_findings", []),
        "previous_questions": state.get("triage_followup_questions", []),
        "candidate_departments_so_far": state.get("triage_candidate_departments", []),
        "recent_user_messages": _recent_user_texts(state),
        "retrieved_rules": candidate_rules,
        "allowed_departments": allowed_departments,
    }
    try:
        llm = _get_llm()
        msg = llm.invoke(
            [
                SystemMessage(content=TRIAGE_INTERVIEW_LLM_SYSTEM),
                HumanMessage(content=json.dumps(context, ensure_ascii=False, indent=2)),
            ]
        )
        obj = _extract_json_object(_llm_content_to_text(getattr(msg, "content", None)))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    obj["_candidate_rules"] = candidate_rules
    return obj


def _apply_triage_interview_plan(state: State, plan: dict) -> Optional[tuple[Optional[str], Optional[str], bool, float, list[str]]]:
    primary = str(plan.get("primary_symptom") or state.get("primary_symptom") or "").strip()
    if primary:
        state["primary_symptom"] = primary

    possible_conditions = _ensure_str_list(plan.get("possible_conditions"))[:6]
    state["triage_possible_conditions"] = possible_conditions

    candidate_depts = []
    for dep in _ensure_str_list(plan.get("candidate_departments"))[:6]:
        clean = _coerce_department_name(dep)
        if clean:
            candidate_depts.append(clean)
    state["triage_candidate_departments"] = _dedupe_keep_order(candidate_depts)

    _merge_state_list(state, "triage_positive_findings", _ensure_str_list(plan.get("positive_findings"))[:8])
    _merge_state_list(state, "triage_negative_findings", _ensure_str_list(plan.get("negative_findings"))[:8])

    questions = [q.rstrip("？?。 ") + "？" for q in _ensure_str_list(plan.get("next_questions")) if q.strip()]
    state["triage_followup_questions"] = questions[:1]
    state["triage_interview_reason"] = str(plan.get("reason") or "").strip() or None
    if plan.get("_candidate_rules") is not None:
        state["triage_candidate_rules"] = list(plan.get("_candidate_rules") or [])

    dep = _coerce_department_name(str(plan.get("recommended_department") or ""))
    is_emergency = bool(plan.get("is_emergency"))
    confidence = max(0.0, min(_safe_float(plan.get("confidence"), 0.0), 1.0))
    if is_emergency and not dep:
        dep = _coerce_department_name("急诊科")
    if dep and (is_emergency or confidence >= 0.65):
        advice = "建议优先前往急诊分诊台，由现场医护确认就诊入口。" if is_emergency else "建议到对应专科门诊，由线下医生进一步评估。"
        matched = _ensure_str_list(plan.get("positive_findings"))[:5]
        return dep, advice, is_emergency, confidence, matched
    return None


def _broad_candidate_departments(symptom: str) -> list[str]:
    t = _normalize_symptom_text(symptom or "")
    groups = [
        (r"(胸闷|胸痛|心口|心慌|气短|憋气)", ["急诊科", "心血管内科", "呼吸与危重症医学科", "消化内科", "心理医学科"]),
        (r"(头痛|头晕|眩晕|麻木|无力|抽搐)", ["急诊科", "神经内科", "神经外科", "耳鼻咽喉科", "眼科"]),
        (r"(腹痛|肚子痛|胃痛|恶心|呕吐|腹泻|便血)", ["急诊科", "消化内科", "普外科急诊", "妇科", "泌尿外科"]),
        (r"(咳嗽|咳痰|发热|发烧|喘|呼吸困难)", ["急诊科", "呼吸与危重症医学科", "感染内科"]),
        (r"(尿痛|尿频|尿急|血尿|腰痛)", ["泌尿外科", "肾脏内科", "急诊科"]),
        (r"(皮疹|瘙痒|红斑|脱发)", ["皮肤科", "变态反应科", "风湿免疫科"]),
        (r"(关节|腰痛|腿痛|骨折|扭伤|外伤)", ["骨科-关节外科", "骨科-脊柱外科", "骨科-创伤骨科", "疼痛科"]),
        (r"(眼痛|视力|飞蚊|眼红)", ["眼科", "急诊科"]),
        (r"(耳|鼻|咽|喉|嗓子|吞咽)", ["耳鼻咽喉科", "口腔科"]),
    ]
    for pattern, departments in groups:
        if re.search(pattern, t):
            return departments
    return ["全科/内科（综合门诊）"]


def _fallback_interview_plan(state: State) -> Optional[dict]:
    if not state.get("symptom"):
        return None
    ranked = _rank_triage_candidates(
        age=state.get("age"),
        gender=state.get("gender"),
        pregnancy_status=state.get("pregnancy_status"),
        symptom=state.get("symptom"),
        limit=6,
    )
    candidate_rules = _triage_candidate_payload(ranked)
    candidate_depts = []
    for c in candidate_rules:
        if float(c.get("retrieval_score") or 0.0) >= 0.5:
            dep = _coerce_department_name(str(c.get("department") or ""))
            if dep:
                candidate_depts.append(dep)
    if not candidate_depts:
        candidate_depts = _broad_candidate_departments(str(state.get("symptom") or ""))
    question = "请说明最不舒服的位置、主要表现，以及是否伴随发热、呼吸困难、恶心呕吐、麻木无力或出血"
    negative_findings = []
    if re.search(r"(没|没有|无).{0,6}(别的|其他).{0,6}(疼|痛|不舒服)", str(state.get("symptom") or "")):
        negative_findings.append("其他部位疼痛")
        question = "请确认是否突然发作，是否伴呼吸困难、心慌出汗、咳嗽喘息、反酸烧心、麻木无力或意识异常"
    return {
        "primary_symptom": state.get("primary_symptom") or state.get("symptom"),
        "possible_conditions": [],
        "candidate_departments": candidate_depts[:5],
        "positive_findings": [],
        "negative_findings": negative_findings,
        "recommended_department": None,
        "is_emergency": False,
        "confidence": 0.0,
        "next_questions": [question],
        "reason": "fallback interview",
        "_candidate_rules": candidate_rules,
    }


def _candidate_symptom_hints(state: State, limit: int = 6) -> list[str]:
    current = _normalize_symptom_text(str(state.get("symptom") or ""))
    hints: list[str] = []
    for c in (state.get("triage_candidate_rules") or [])[:3]:
        if not isinstance(c, dict):
            continue
        try:
            if float(c.get("retrieval_score") or 0.0) < 0.5:
                continue
        except Exception:
            continue
        for raw in c.get("symptoms") or c.get("symptom_keywords") or []:
            hint = str(raw or "").strip()
            if not hint:
                continue
            if current and _normalize_symptom_text(hint) in current:
                continue
            hints.append(hint)
    return _dedupe_keep_order(hints)[:limit]


def _build_triage_followup(state: State) -> str:
    missing = state.get("_missing_fields", []) or []
    if "主要症状" in missing:
        parts: list[str] = []
        if "年龄" in missing:
            parts.append("年龄")
        if "性别" in missing:
            parts.append("性别（男/女）")
        parts.append("主要症状（哪里不舒服/哪里疼，最明显的表现是什么）")
        return "为了推荐科室，请补充：" + "、".join(parts) + "。"

    candidate_depts = _dedupe_keep_order(_ensure_str_list(state.get("triage_candidate_departments")))[:5]
    primary = str(state.get("primary_symptom") or state.get("symptom") or "").strip()
    lines = []
    if candidate_depts:
        lines.append("根据目前描述，可能涉及：" + "、".join(candidate_depts) + "。")
    else:
        lines.append("目前信息还不足以推荐具体科室。")

    recorded = []
    if primary:
        recorded.append(primary)
    recorded.extend(_ensure_str_list(state.get("triage_positive_findings")))
    for item in _ensure_str_list(state.get("triage_negative_findings")):
        recorded.append(item if re.match(r"^(无|没有|没|否认|不伴)", item) else "无" + item)
    recorded = _dedupe_keep_order(recorded)
    if recorded:
        lines.append("已记录：" + "、".join(recorded) + "。")

    if missing:
        profile_parts = []
        if "年龄" in missing:
            profile_parts.append("年龄")
        if "性别" in missing:
            profile_parts.append("性别（男/女）")
        if profile_parts:
            lines.append("请先补充：" + "、".join(profile_parts) + "。")

    questions = _ensure_str_list(state.get("triage_followup_questions"))
    if questions:
        q = questions[0]
        lines.append(q if q.startswith("请") else "请确认：" + q)
    else:
        hints = _candidate_symptom_hints(state)
        question = "请说明最不舒服的位置、主要表现，以及是否有伴随症状。"
        if hints:
            question = f"请确认是否伴随：{'、'.join(hints[:5])}。"
        lines.append(question)
    lines.append("如果是突发胸痛、呼吸困难、肢体无力、意识异常或大量出血，请优先去急诊分诊台。")
    return "\n".join(lines)


def _triage_match(
    age: Optional[int],
    gender: Optional[str],
    pregnancy_status: Optional[str],
    symptom: Optional[str],
) -> tuple[Optional[str], Optional[str], bool, float, list[str]]:
    """
    Fix matching bug: use char-ngram TF-IDF similarity between symptom and each rule's symptom list.
    This allows '头疼/头痛' to match entries like '持续性剧烈头痛'.
    """
    if not symptom:
        return None, None, False, 0.0, []

    gender_n = _normalize_gender(gender or "any")
    pregnancy_n = _normalize_pregnancy_status(pregnancy_status or "any")
    age_v = int(age) if isinstance(age, int) else None
    q = str(symptom).strip()
    if not q:
        return None, None, False, 0.0, []

    texts: list[str] = []
    rules: list[dict] = []
    rule_kws: list[list[str]] = []
    for r in TRIAGE_RULES:
        if not isinstance(r, dict):
            continue
        age_min = int(r.get("age_min", 0))
        age_max = int(r.get("age_max", 200))
        rg = _normalize_gender(str(r.get("gender", "any")))
        if age_v is not None and not (age_min <= age_v <= age_max):
            continue
        if rg not in {"any", gender_n}:
            continue
        rp = _normalize_pregnancy_status(str(r.get("pregnancy_status", "any")))
        if pregnancy_n != "any" and rp not in {"any", pregnancy_n}:
            continue
        kws = r.get("symptoms") or r.get("symptom_keywords") or []
        if not isinstance(kws, list) or not kws:
            continue
        nkws = [_normalize_symptom_text(str(x)) for x in kws if x]
        texts.append(" ".join([x for x in nkws if x]))
        rules.append(r)
        rule_kws.append([x for x in nkws if x])

    if not texts:
        return None, None, False, 0.0, []

    qn = _normalize_symptom_text(q)
    best_i_kw = -1
    best_kw_score = 0.0
    best_matched: list[str] = []
    for i, kws in enumerate(rule_kws):
        score = 0.0
        matched: list[str] = []
        for kw in kws:
            part_score = _keyword_match_score(qn, kw)
            if part_score > 0:
                score += part_score
                matched.append(kw)
        if bool(rules[i].get("is_emergency")) and len(matched) >= 2:
            score += 3.0
        if score > best_kw_score:
            best_kw_score = score
            best_i_kw = i
            best_matched = matched
    if best_i_kw >= 0 and (best_kw_score >= 3.5 or len(best_matched) >= 2):
        r = rules[best_i_kw]
        dep = str(r.get("recommended_department") or r.get("department") or "").strip() or None
        return dep, _safe_triage_advice(r), bool(r.get("is_emergency")), min(best_kw_score / 30.0, 1.0), best_matched

    best_i = 0
    best_score = -1.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        X = vec.fit_transform(texts)
        qv = vec.transform([qn])
        sims = cosine_similarity(qv, X).flatten()
        best_i = int(sims.argmax())
        best_score = float(sims[best_i])
    except Exception:
        # fallback: substring overlap
        for i, t in enumerate(texts):
            score = 0
            for kw in set(re.split(r"[\s,，。/]+", t)):
                if kw and kw in qn:
                    score += len(kw)
            if score > best_score:
                best_score = float(score)
                best_i = i

    # Use semantic similarity only when it is strong enough; weak scores trigger a follow-up instead of a guess.
    if best_score <= 0.001:
        for i, t in enumerate(texts):
            if any(tok and tok in qn for tok in t.split()):
                best_i = i
                best_score = 0.01
                break
    if best_score <= 0.001:
        return None, None, False, 0.0, []

    r = rules[best_i]
    dep = str(r.get("recommended_department") or r.get("department") or "").strip() or None
    matched = [kw for kw in rule_kws[best_i] if _keyword_match_score(qn, kw) > 0]
    if not matched and best_score < 0.18:
        return None, None, False, best_score, []
    return dep, _safe_triage_advice(r), bool(r.get("is_emergency")), min(best_score, 1.0), matched


def _use_triage_llm() -> bool:
    return env_flag("USE_TRIAGE_LLM", True)


def _rank_triage_candidates(
    age: Optional[int],
    gender: Optional[str],
    pregnancy_status: Optional[str],
    symptom: Optional[str],
    limit: int = 8,
) -> list[dict]:
    if not symptom:
        return []

    gender_n = _normalize_gender(gender or "any")
    pregnancy_n = _normalize_pregnancy_status(pregnancy_status or "any")
    age_v = int(age) if isinstance(age, int) else None
    qn = _normalize_symptom_text(str(symptom))
    if not qn:
        return []

    entries: list[dict] = []
    texts: list[str] = []
    for idx, r in enumerate(TRIAGE_RULES):
        if not isinstance(r, dict):
            continue
        age_min = int(r.get("age_min", 0))
        age_max = int(r.get("age_max", 200))
        rg = _normalize_gender(str(r.get("gender", "any")))
        if age_v is not None and not (age_min <= age_v <= age_max):
            continue
        if rg not in {"any", gender_n}:
            continue
        rp = _normalize_pregnancy_status(str(r.get("pregnancy_status", "any")))
        if pregnancy_n != "any" and rp not in {"any", pregnancy_n}:
            continue

        raw_kws = r.get("symptoms") or r.get("symptom_keywords") or []
        if not isinstance(raw_kws, list) or not raw_kws:
            continue
        kws = [_normalize_symptom_text(str(x)) for x in raw_kws if str(x).strip()]
        if not kws:
            continue

        matched = [kw for kw in kws if _keyword_match_score(qn, kw) > 0]
        kw_score = sum(_keyword_match_score(qn, kw) for kw in kws)
        if bool(r.get("is_emergency")) and len(matched) >= 2:
            kw_score += 3.0

        entries.append(
            {
                "rule_index": idx,
                "rule_id": f"rule_{idx}",
                "rule": r,
                "keywords": kws,
                "matched_by_retrieval": matched,
                "keyword_score": float(kw_score),
                "semantic_score": 0.0,
            }
        )
        texts.append(" ".join(kws))

    if not entries:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        X = vec.fit_transform(texts)
        qv = vec.transform([qn])
        sims = cosine_similarity(qv, X).flatten()
        for i, sim in enumerate(sims):
            entries[i]["semantic_score"] = float(sim)
    except Exception:
        pass

    for entry in entries:
        entry["retrieval_score"] = float(entry["keyword_score"]) + float(entry["semantic_score"]) * 10.0

    entries.sort(key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
    return entries[: max(1, limit)]


TRIAGE_MATCH_LLM_SYSTEM = (
    "你是医院导诊系统的语义分诊裁决模块。你的任务不是诊断疾病，也不是给治疗建议，"
    "而是在给定候选知识库规则中选择最匹配的一条。\n"
    "严格要求：\n"
    "1. 只能从 candidate_rules 中选择，不得创造新科室或新规则。\n"
    "2. 主要依据用户症状语义匹配 symptoms，考虑同义表达和口语表达。\n"
    "3. 年龄、性别、孕期已由系统用于候选过滤，但如明显冲突应选择 null。\n"
    "4. 如果没有足够匹配的候选，selected_rule_id 必须为 null。\n"
    "5. 仅输出 JSON，不要解释、不要 Markdown。\n"
    "JSON 格式：{\"selected_rule_id\": string|null, \"confidence\": 0到1, "
    "\"matched_symptoms\": [string], \"reason\": string}"
)


def _triage_match_llm(
    age: Optional[int],
    gender: Optional[str],
    pregnancy_status: Optional[str],
    symptom: Optional[str],
) -> Optional[dict]:
    if not _use_triage_llm() or not symptom:
        return None

    candidates = _rank_triage_candidates(age, gender, pregnancy_status, symptom, limit=8)
    if not candidates:
        return None

    candidate_payload = _triage_candidate_payload(candidates)
    by_id = {}
    for c in candidates:
        rid = str(c["rule_id"])
        by_id[rid] = c

    try:
        llm = _get_llm()
        msg = llm.invoke(
            [
                SystemMessage(content=TRIAGE_MATCH_LLM_SYSTEM),
                HumanMessage(
                    content=(
                        "patient_profile:\n"
                        f"{json.dumps({'age': age, 'gender': gender, 'pregnancy_status': pregnancy_status}, ensure_ascii=False)}\n\n"
                        f"user_symptom:\n{symptom}\n\n"
                        "candidate_rules:\n"
                        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
                    )
                ),
            ]
        )
        raw = _llm_content_to_text(getattr(msg, "content", None)).strip()
        obj = _extract_json_object(raw)
    except Exception:
        return None

    selected = obj.get("selected_rule_id")
    reason = str(obj.get("reason") or "").strip()
    matched = obj.get("matched_symptoms") or []
    if not isinstance(matched, list):
        matched = []
    try:
        confidence = float(obj.get("confidence", 0.7 if selected else 0.0))
    except Exception:
        confidence = 0.7 if selected else 0.0
    confidence = max(0.0, min(confidence, 1.0))

    if selected is None or str(selected).lower() in {"", "none", "null"}:
        return {
            "department": None,
            "triage_advice": None,
            "is_emergency": False,
            "confidence": confidence,
            "matched_symptoms": [str(x) for x in matched if str(x).strip()],
            "rule_id": None,
            "source": "llm_no_match",
            "candidate_rules": candidate_payload,
            "reason": reason,
        }

    selected_id = str(selected).strip()
    c = by_id.get(selected_id)
    if not c:
        return None

    r = c["rule"]
    dep = str(r.get("recommended_department") or r.get("department") or "").strip() or None
    return {
        "department": dep,
        "triage_advice": _safe_triage_advice(r),
        "is_emergency": bool(r.get("is_emergency")),
        "confidence": confidence,
        "matched_symptoms": [str(x) for x in matched if str(x).strip()] or c.get("matched_by_retrieval", []),
        "rule_id": selected_id,
        "source": "llm_semantic_match",
        "candidate_rules": candidate_payload,
        "reason": reason,
    }


def _summarize_schedule_item(item: dict) -> str:
    name = str(item.get("name") or item.get("doctor_name") or "").strip()
    title = str(item.get("title") or "").strip()
    time = str(item.get("time") or "").strip()
    sub = str(item.get("sub_specialty") or "").strip()
    who = "，".join([x for x in [name, title, sub] if x]) or "门诊医生"
    return f"{who}：{time}" if time else who


def _search_next_available(dept: str, start_day: str, horizon: int = 6) -> list[dict]:
    if start_day not in WEEKDAYS:
        start_day = WEEKDAYS[NOW.weekday()]
    start_idx = WEEKDAYS.index(start_day)

    candidates: list[dict] = []
    for offset in range(0, horizon + 1):
        day = WEEKDAYS[(start_idx + offset) % 7]
        raw = get_doctor_schedule(dept, day)
        avail = (raw or {}).get("available") or []
        if avail:
            candidates.append({"day": day, "items": [_summarize_schedule_item(x) for x in avail[:5]]})
            break
    return candidates


def _collect_schedule_window(dept: str, start_day: str, horizon: int = 6, max_days_with_avail: int = 3) -> list[dict]:
    """
    Collect up to N days with availability from start_day forward within horizon.
    Used to proactively suggest alternatives when today is full.
    """
    if start_day not in WEEKDAYS:
        start_day = WEEKDAYS[NOW.weekday()]
    start_idx = WEEKDAYS.index(start_day)

    out: list[dict] = []
    for offset in range(0, horizon + 1):
        day = WEEKDAYS[(start_idx + offset) % 7]
        raw = get_doctor_schedule(dept, day)
        avail = (raw or {}).get("available") or []
        if avail:
            out.append({"day": day, "items": [_summarize_schedule_item(x) for x in avail[:5]]})
        if len(out) >= max_days_with_avail:
            break
    return out


_strip_forbidden_tech = strip_forbidden_tech


def _llm_content_to_text(content) -> str:
    """
    Gemini (and some providers) may return content as a list of parts, e.g.
    [{"type":"text","text":"...","extras":{...}}, ...]
    We only keep concatenated text fields.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if isinstance(p, dict):
                txt = p.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join([x.strip() for x in parts if x and x.strip()]).strip()
    # fallback
    return str(content)


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return json.loads(m.group(0) if m else raw)


SLOT_LLM_SYSTEM = (
    "你是医院导诊系统的信息抽取模块，只从用户原文抽取结构化字段，不诊断、不推荐科室。\n"
    "输出且仅输出 JSON：{\"age\": int|null, \"gender\": \"male\"|\"female\"|null, "
    "\"pregnancy_status\": \"yes\"|\"no\"|\"any\"|null, \"symptom\": string|null}\n"
    "规则：新生儿/刚出生/出生后/生后可记为 age=0；婴儿/宝宝可按 age=1；"
    "如果用户没有明确给出字段则填 null；symptom 保留主要不适描述，去掉挂号/导诊客套话。"
)


def _coerce_llm_age(value) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            m = re.search(r"\d{1,3}", value)
            if not m:
                return None
            value = int(m.group(0))
        age = int(value)
        if 0 <= age <= 120:
            return age
    except Exception:
        return None
    return None


def _coerce_llm_gender(value) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if raw in {"male", "m", "男", "男性", "男士", "男生", "男孩", "男童", "男婴"}:
        return "male"
    if raw in {"female", "f", "女", "女性", "女士", "女生", "女孩", "女童", "女婴"}:
        return "female"
    return None


def _coerce_llm_pregnancy(value) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if raw in {"yes", "y", "true", "pregnant", "怀孕", "孕", "妊娠", "孕期"}:
        return "yes"
    if raw in {"no", "n", "false", "not_pregnant", "未孕", "未怀孕", "没怀孕", "没有怀孕"}:
        return "no"
    if raw in {"any", "unknown", "不详", "未知"}:
        return "any"
    return None


def _parse_age_gender_symptom_llm(text: str) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    if not _use_triage_llm() or not (text or "").strip():
        return None, None, None, None
    try:
        llm = _get_llm()
        msg = llm.invoke(
            [
                SystemMessage(content=SLOT_LLM_SYSTEM),
                HumanMessage(content=f"用户原文：{text}"),
            ]
        )
        obj = _extract_json_object(_llm_content_to_text(getattr(msg, "content", None)))
    except Exception:
        return None, None, None, None

    age = _coerce_llm_age(obj.get("age"))
    gender = _coerce_llm_gender(obj.get("gender"))
    pregnancy_status = _coerce_llm_pregnancy(obj.get("pregnancy_status"))
    symptom = str(obj.get("symptom") or "").strip()
    if not symptom or re.fullmatch(r"(没有其他症状|无其他症状|没有别的症状|无别的症状|无其他不适|没有其他不适|无不适|没有不适)", symptom):
        symptom = None
    return age, gender, pregnancy_status, symptom


INTENT_LLM_SYSTEM = (
    "你是医院导诊系统的意图分类模块。只根据用户最后一句话做分类，不要解释。\n"
    "输出且仅输出一个 JSON 对象，格式：{\"intent\":\"location_query\"|\"triage_consult\"|\"other\"}\n\n"
    "定义：\n"
    "- location_query：询问院内/院区地点、楼层、窗口编号、某科怎么走、检查/取药/办事地点、路线指引等。\n"
    "- triage_consult：身体不适、症状描述、该挂什么科/看什么科、分诊建议需求等。\n"
    "- other：与院内就诊导诊明显无关（天气、机票、院外餐馆闲聊、个人喜好等）。"
    "院内缴费/挂号方式等流程性问题若明显不是问“去哪办”，可归为 other。\n"
)


def _classify_intent_llm(text: str) -> Optional[Intent]:
    """
    Structured LLM intent; returns None on failure so caller can fall back to regex.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        llm = _get_llm()
        msg = llm.invoke(
            [
                SystemMessage(content=INTENT_LLM_SYSTEM),
                HumanMessage(content=f"用户：{text}"),
            ]
        )
        raw = _llm_content_to_text(getattr(msg, "content", None)).strip()
        obj = _extract_json_object(raw)
        v = str(obj.get("intent", "")).strip()
        if v in {"location_query", "triage_consult", "other"}:
            return v  # type: ignore[return-value]
    except Exception:
        pass
    return None


def classify_intent(state: State) -> State:
    text = _last_user_text(state)
    use_llm = env_flag("USE_INTENT_LLM", True)
    intent: Optional[Intent] = None
    if use_llm:
        intent = _classify_intent_llm(text)
        if intent is not None:
            state["intent_source"] = "llm"
    if intent is None:
        intent = _classify_intent_regex(text)
        state["intent_source"] = "regex_fallback" if use_llm else "regex_only"
    state["intent"] = intent
    return state


_contains_forbidden_medical = contains_forbidden_medical
_compliant_replacement = compliant_replacement


def _finalize(state: State, draft: str) -> State:
    reply = _strip_forbidden_tech(draft)
    if state.get("action") != "LOCATION" and _contains_forbidden_medical(reply):
        reply = _compliant_replacement(state.get("department"))
    state.setdefault("messages", [])
    state["messages"].append(AIMessage(content=reply))
    return state


# -------------------------
# Graph nodes
# -------------------------
def ingest_and_transition(state: State) -> State:
    text = _last_user_text(state)
    state.setdefault("current_phase", "INIT")

    if not state.get("intent"):
        state["intent"] = _classify_intent_regex(text)

    if state.get("intent") == "other":
        state["action"] = "OTHER"
        state["_missing_fields"] = []
        return state

    if state["intent"] == "location_query":
        state["action"] = "LOCATION"
        return state

    if state.get("current_phase") == "RECOMMENDED" and _user_asks_schedule(text):
        state["current_phase"] = "SCHEDULE"
        state["action"] = "SCHEDULE"
        return state

    llm_age, llm_gender, llm_pregnancy_status, llm_symptom = _parse_age_gender_symptom_llm(text)
    rule_age, rule_gender, rule_pregnancy_status, rule_symptom = _parse_age_gender_symptom(text)
    age = llm_age if llm_age is not None else rule_age
    gender = llm_gender if llm_gender is not None else rule_gender
    pregnancy_status = llm_pregnancy_status if llm_pregnancy_status is not None else rule_pregnancy_status
    symptom = llm_symptom if llm_symptom is not None else rule_symptom
    state["slot_extract_source"] = "llm_with_rule_fallback" if any(
        x is not None for x in (llm_age, llm_gender, llm_pregnancy_status, llm_symptom)
    ) else "rule"

    if age is not None:
        state["age"] = age
    if gender is not None:
        state["gender"] = gender
    if pregnancy_status is not None:
        state["pregnancy_status"] = pregnancy_status
    # Do not let generic "no other symptoms" overwrite existing symptom unless user explicitly resets triage.
    if symptom is not None and symptom.strip():
        s = symptom.strip()
        if not (
            s in {"没有其他症状", "无其他症状", "没有别的症状", "无别的症状", "无其他不适", "没有其他不适", "无不适", "没有不适"}
            and state.get("symptom")
            and not _user_resets_triage(text)
        ):
            if state.get("current_phase") == "TRIAGE" and state.get("symptom") and not _user_resets_triage(text):
                state["symptom"] = _merge_text_slot(state.get("symptom"), s)
            else:
                state["symptom"] = s
            if not state.get("primary_symptom"):
                state["primary_symptom"] = s

    # triage intent
    if _user_resets_triage(text):
        state["current_phase"] = "TRIAGE"
        state["department"] = None
        state["triage_advice"] = None
        state["is_emergency"] = False
        state["match_confidence"] = None
        state["matched_symptoms"] = []
        state["matched_rule_id"] = None
        state["triage_match_source"] = None
        state["triage_candidate_rules"] = []
        state["triage_llm_reason"] = None
        state["triage_candidate_departments"] = []
        state["triage_possible_conditions"] = []
        state["triage_positive_findings"] = []
        state["triage_negative_findings"] = []
        state["triage_followup_questions"] = []
        state["triage_interview_reason"] = None
        state["primary_symptom"] = None

    if state["current_phase"] in {"INIT", "TRIAGE"}:
        state["action"] = "TRIAGE"
        return state

    if state["current_phase"] == "RECOMMENDED":
        if _user_asks_schedule(text):
            state["current_phase"] = "SCHEDULE"
            state["action"] = "SCHEDULE"
        else:
            state["action"] = "TRIAGE"
        return state

    if state["current_phase"] == "SCHEDULE":
        state["action"] = "SCHEDULE"
        return state

    state["action"] = "TRIAGE"
    return state


def extract_location(state: State) -> State:
    text = _last_user_text(state)
    results = search_location(text, k=3)
    state["location_results"] = [
        {
            "service": r.service,
            "building": r.building,
            "floor": r.floor,
            "room": r.room,
            "hours": r.hours,
            "directions": r.directions,
        }
        for r in results
    ]
    state["schedule_candidates"] = []
    state["schedule_window"] = []
    return state


def extract_triage(state: State) -> State:
    age = state.get("age")
    gender = state.get("gender")
    pregnancy_status = state.get("pregnancy_status")
    symptom = state.get("symptom")

    missing = []
    if age is None:
        missing.append("年龄")
    if not gender:
        missing.append("性别")
    if not symptom:
        missing.append("主要症状")
    state["_missing_fields"] = missing
    if "主要症状" in missing:
        state["current_phase"] = "TRIAGE"
        state["triage_followup_questions"] = []
        return state

    dep = None
    advice = None
    is_emergency = False
    confidence = 0.0
    matched_symptoms: list[str] = []

    interview_plan = _triage_interview_llm(state)
    if interview_plan is not None:
        interview_match = _apply_triage_interview_plan(state, interview_plan)
        state["matched_rule_id"] = None
        state["triage_match_source"] = "llm_interview"
        state["triage_llm_reason"] = state.get("triage_interview_reason")
        if interview_match is not None:
            dep, advice, is_emergency, confidence, matched_symptoms = interview_match
    else:
        dep, advice, is_emergency, confidence, matched_symptoms = _triage_match(
            age=age,
            gender=gender,
            pregnancy_status=pregnancy_status,
            symptom=symptom,
        )
        state["matched_rule_id"] = None
        state["triage_match_source"] = "rule_fallback"
        ranked = _rank_triage_candidates(age, gender, pregnancy_status, symptom, limit=8)
        state["triage_candidate_rules"] = _triage_candidate_payload(ranked)
        state["triage_llm_reason"] = None

        if not dep:
            fallback_plan = _fallback_interview_plan(state)
            if fallback_plan:
                _apply_triage_interview_plan(state, fallback_plan)

    state["department"] = dep
    state["triage_advice"] = advice
    state["is_emergency"] = bool(is_emergency)
    state["match_confidence"] = confidence
    state["matched_symptoms"] = matched_symptoms
    if not dep:
        state["current_phase"] = "TRIAGE"
        state["registration_steps"] = []
        state["registration_location"] = None
        state["schedule_candidates"] = []
        state["schedule_window"] = []
        return state

    state["_missing_fields"] = []
    state["current_phase"] = "RECOMMENDED"

    # Proactively prepare registration + schedule window (no need for user to ask).
    state["registration_steps"] = [
        "如需挂号：可在门诊大厅自助机/人工窗口进行挂号与缴费，或使用医院官方小程序/APP 线上挂号。",
        "现场挂号一般需要：身份证/就诊卡/医保凭证（如有），并按提示完成取号/签到。",
        "若你愿意，我可以继续帮你查看今天以及接下来几天的可用号源，并帮你一起挑选合适的时间段。",
    ]
    # try locate registration desk for in-hospital directions (optional)
    reg = search_location("挂号", k=1)
    state["registration_location"] = (
        {
            "service": reg[0].service,
            "building": reg[0].building,
            "floor": reg[0].floor,
            "room": reg[0].room,
            "hours": reg[0].hours,
            "directions": reg[0].directions,
        }
        if reg
        else None
    )

    # schedule window: always include today + next days (Mon..)
    start_day = WEEKDAYS[NOW.weekday()]
    if state.get("department") and not state.get("is_emergency"):
        state["schedule_day"] = start_day
        # first available day (may be today or later)
        state["schedule_candidates"] = _search_next_available(state["department"], start_day, horizon=6)
        state["schedule_window"] = _collect_schedule_window(state["department"], start_day, horizon=6, max_days_with_avail=3)
    else:
        state["schedule_candidates"] = []
        state["schedule_window"] = []
    return state


def extract_schedule(state: State) -> State:
    text = _last_user_text(state)
    dep = state.get("department")
    start_day = _infer_start_day(text)
    state["schedule_day"] = start_day

    if state.get("is_emergency"):
        state["schedule_candidates"] = []
        state["schedule_window"] = []
        return state

    if not dep:
        state["schedule_candidates"] = []
        return state

    # always: check today first; if none and user says "未来/换一天" or asks "啥时候"，search next.
    want_future = _needs_future_search(text) or bool(re.search(r"(啥时候|什么时候)", text))
    candidates = _search_next_available(dep, start_day, horizon=6) if want_future else _search_next_available(dep, start_day, horizon=6)
    state["schedule_candidates"] = candidates
    state["schedule_window"] = _collect_schedule_window(dep, start_day, horizon=6, max_days_with_avail=3)
    return state


def llm_responder(state: State) -> State:
    text = _last_user_text(state)
    action = state.get("action") or "TRIAGE"

    if action == "OTHER":
        return _finalize(
            state,
            "我只能回答与本院导诊相关的问题（科室推荐、院内位置与路线、挂号与号源查询）。请提出与就诊相关的问题。",
        )

    missing = state.get("_missing_fields", []) or []
    if "主要症状" in missing:
        # Block LLM free-form generation when core fields are missing, but ask a usable follow-up.
        return _finalize(state, _build_triage_followup(state))

    context = {
        "now": _format_now_text(),
        "current_phase": state.get("current_phase", "INIT"),
        "intent": state.get("intent", "triage_consult"),
        "action": action,
        "age": state.get("age"),
        "gender": state.get("gender"),
        "pregnancy_status": state.get("pregnancy_status"),
        "symptom": state.get("symptom"),
        "department": state.get("department"),
        "routing_note": state.get("triage_advice"),
        "is_emergency": state.get("is_emergency", False),
        "match_confidence": state.get("match_confidence"),
        "matched_symptoms": state.get("matched_symptoms", []),
        "matched_rule_id": state.get("matched_rule_id"),
        "triage_match_source": state.get("triage_match_source"),
        "candidate_departments": state.get("triage_candidate_departments", []),
        "possible_conditions_internal": state.get("triage_possible_conditions", []),
        "positive_findings": state.get("triage_positive_findings", []),
        "negative_findings": state.get("triage_negative_findings", []),
        "followup_questions": state.get("triage_followup_questions", []),
        "missing_fields": missing,
        "location_results": state.get("location_results", []),
        "schedule_day": state.get("schedule_day"),
        "schedule_candidates": state.get("schedule_candidates", []),
        "schedule_window": state.get("schedule_window", []),
        "registration_steps": state.get("registration_steps", []),
        "registration_location": state.get("registration_location"),
    }

    # Only let LLM talk when we have concrete extracted conclusions to fill in.
    if action == "TRIAGE" and not state.get("department"):
        return _finalize(state, _build_triage_followup(state))

    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"用户最后一句：{text}\n\n"
                "可用数据（只能基于此回复）：\n"
                f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                "请输出极简、直接、客观的导诊结论。\n"
                "强制要求：\n"
                "- 不要问候，不要安慰，不要解释背景。\n"
                "- 不诊断；不提用药/治疗；不评估严重程度。\n"
                "- 若 is_emergency 为 true，只给急诊入口，不输出号源或预约建议。\n"
                "- 只做“填空式结论”，尽量使用短句，每句一行。\n"
                "- 只能使用上面的结构化数据，不得自行补充。\n\n"
                "输出格式建议（按可用数据填充，缺项就跳过该行）：\n"
                "推荐科室：XXX\n"
                "号源：今天（周一）无号；最近可挂：周三 张主任 AM 有号\n"
                "挂号地点：门诊楼 1F 大厅A区\n"
            )
        ),
    ]
    try:
        llm = _get_llm()
        msg = llm.invoke(prompt)
        draft = _llm_content_to_text(getattr(msg, "content", None)).strip()
    except Exception:
        # minimal safe fallback
        if action == "LOCATION":
            items = (state.get("location_results") or [])[:1]
            if items:
                it = items[0]
                draft = f"位置：{it.get('building','')} {it.get('floor','')} {it.get('room','')}\n路线：{it.get('directions','')}".strip()
            else:
                draft = "请说明要查询的地点/项目（例如：抽血、挂号、药房、CT）。"
        else:
            dep = state.get("department") or "相关科室"
            if state.get("is_emergency"):
                draft = f"推荐就诊入口：{dep}\n{state.get('triage_advice') or '建议优先前往急诊分诊台。'}".strip()
                return _finalize(state, draft)
            # schedule_window: prefer earliest available
            win = state.get("schedule_window") or []
            if win:
                day = win[0].get("day")
                first = (win[0].get("items") or [""])[0]
                draft = f"推荐科室：{dep}\n号源：最近可挂：{day} {first}".strip()
            else:
                draft = f"推荐科室：{dep}".strip()

    return _finalize(state, draft)


def _route(state: State) -> str:
    action = state.get("action") or "TRIAGE"
    if action == "LOCATION":
        return "extract_location"
    if action == "SCHEDULE":
        return "extract_schedule"
    if action == "OTHER":
        return "llm_responder"
    return "extract_triage"


def build_graph():
    g = StateGraph(State)
    g.add_node("classify_intent", classify_intent)
    g.add_node("ingest_and_transition", ingest_and_transition)
    g.add_node("extract_location", extract_location)
    g.add_node("extract_triage", extract_triage)
    g.add_node("extract_schedule", extract_schedule)
    g.add_node("llm_responder", llm_responder)

    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "ingest_and_transition")
    g.add_conditional_edges(
        "ingest_and_transition",
        _route,
        {
            "extract_location": "extract_location",
            "extract_triage": "extract_triage",
            "extract_schedule": "extract_schedule",
            "llm_responder": "llm_responder",
        },
    )
    g.add_edge("extract_location", "llm_responder")
    g.add_edge("extract_triage", "llm_responder")
    g.add_edge("extract_schedule", "llm_responder")
    g.add_edge("llm_responder", END)
    return g.compile(checkpointer=MemorySaver())


GRAPH = build_graph()


def run_turn(user_text: str, thread_id: str) -> list[BaseMessage]:
    out = GRAPH.invoke({"messages": [("human", user_text)]}, config={"configurable": {"thread_id": thread_id}})
    return out.get("messages", [])
