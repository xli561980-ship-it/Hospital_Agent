from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
MOCK_DATA_DIR = BASE_DIR / "mock_data"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load_locations() -> Dict[str, Any]:
    return _read_json(MOCK_DATA_DIR / "locations.json")


@lru_cache(maxsize=8)
def _load_triage_rules() -> Dict[str, Any]:
    return _read_json(MOCK_DATA_DIR / "triage_rules.json")


@lru_cache(maxsize=8)
def _load_doctor_schedules() -> Dict[str, Any]:
    return _read_json(MOCK_DATA_DIR / "doctor_schedules.json")


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _format_location_item(item: Dict[str, Any]) -> str:
    parts = [
        f"服务：{item.get('service')}",
        f"位置：{item.get('building')} {item.get('floor')} {item.get('room')}",
    ]
    if item.get("hours"):
        parts.append(f"开放时间：{item.get('hours')}")
    if item.get("directions"):
        parts.append(f"路线：{item.get('directions')}")
    return "\n".join(parts)


def _build_location_corpus() -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    data = _load_locations()
    items: List[Dict[str, Any]] = data.get("locations", [])
    docs: List[str] = []
    ids: List[str] = []
    for idx, it in enumerate(items):
        alias = " ".join(it.get("aliases", []) or [])
        doc = f"{it.get('service','')} {alias} {it.get('building','')} {it.get('floor','')} {it.get('room','')} {it.get('directions','')}"
        docs.append(_normalize_text(doc))
        ids.append(f"loc_{idx}")
    return items, ids, docs


@dataclass
class LocationSearchResult:
    service: str
    building: str
    floor: str
    room: str
    hours: Optional[str]
    directions: Optional[str]
    score: float

    def to_human_text(self) -> str:
        item = {
            "service": self.service,
            "building": self.building,
            "floor": self.floor,
            "room": self.room,
            "hours": self.hours,
            "directions": self.directions,
        }
        return _format_location_item(item)


def search_location(query: str, k: int = 3) -> List[LocationSearchResult]:
    """
    读取 mock_data/locations.json，并对 locations 做语义/相似度检索。

    - 优先：chromadb + langchain embedding（若环境已安装且可用）
    - 兜底：scikit-learn TF-IDF + 余弦相似度
    """
    query_n = _normalize_text(query)
    items, ids, docs = _build_location_corpus()

    if not docs:
        return []

    # Try Chroma (optional)
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        # For demo, keep an in-memory Chroma. Requires OPENAI_API_KEY.
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            metadatas=[{"idx": i} for i in range(len(docs))],
            ids=ids,
            collection_name="hospital_locations_mock",
        )
        results = vectordb.similarity_search_with_score(query_n, k=min(k, len(docs)))

        out: List[LocationSearchResult] = []
        for doc, score in results:
            idx = int(doc.metadata.get("idx"))
            it = items[idx]
            out.append(
                LocationSearchResult(
                    service=it.get("service", ""),
                    building=it.get("building", ""),
                    floor=it.get("floor", ""),
                    room=it.get("room", ""),
                    hours=it.get("hours"),
                    directions=it.get("directions"),
                    score=float(score),
                )
            )
        return out
    except Exception:
        pass

    # Fallback: TF-IDF cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Chinese-friendly n-gram TF-IDF (works reasonably for short queries like “抽血在几楼”)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        X = vectorizer.fit_transform(docs)
        q = vectorizer.transform([query_n])
        sims = cosine_similarity(q, X).flatten()
        ranked = sims.argsort()[::-1][: min(k, len(items))]

        out = []
        for i in ranked:
            it = items[int(i)]
            out.append(
                LocationSearchResult(
                    service=it.get("service", ""),
                    building=it.get("building", ""),
                    floor=it.get("floor", ""),
                    room=it.get("room", ""),
                    hours=it.get("hours"),
                    directions=it.get("directions"),
                    score=float(sims[int(i)]),
                )
            )
        return out
    except Exception:
        # Last resort: simple keyword containment scoring
        tokens = [t for t in re.split(r"[\s/,_\-]+", query_n) if t]
        scored: List[Tuple[int, float]] = []
        for i, d in enumerate(docs):
            score = sum(1.0 for t in tokens if t in d)
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i, score in scored[: min(k, len(items))]:
            it = items[int(i)]
            out.append(
                LocationSearchResult(
                    service=it.get("service", ""),
                    building=it.get("building", ""),
                    floor=it.get("floor", ""),
                    room=it.get("room", ""),
                    hours=it.get("hours"),
                    directions=it.get("directions"),
                    score=float(score),
                )
            )
        return out


def get_department(age: int, gender: str, symptom: str) -> str:
    """
    基于 mock_data/triage_rules.json 的简单规则匹配，返回推荐科室名称。
    gender: "male" | "female" | "any"（输入也可为中文：男/女）
    """
    gender_n = _normalize_text(gender)
    if gender_n in {"男", "male", "m"}:
        gender_n = "male"
    elif gender_n in {"女", "female", "f"}:
        gender_n = "female"
    else:
        gender_n = "any"

    symptom_n = symptom.strip()
    raw = _load_triage_rules()

    # Support two formats:
    # 1) {"rules":[{"symptom_keywords":..., "department":...}], "default_department":...}
    # 2) [{"symptoms":[...], "recommended_department":...}, ...]
    if isinstance(raw, dict):
        default_dep = str(raw.get("default_department", "全科/内科（综合门诊）"))
        iterable = raw.get("rules", []) or []
        for rule in iterable:
            age_min = int(rule.get("age_min", 0))
            age_max = int(rule.get("age_max", 200))
            rule_gender = _normalize_text(str(rule.get("gender", "any")))
            if not (age_min <= int(age) <= age_max):
                continue
            if rule_gender not in {"any", gender_n}:
                continue
            kws = rule.get("symptom_keywords", []) or []
            if any(str(kw) in symptom_n for kw in kws):
                return str(rule.get("department", default_dep))
        return default_dep

    if isinstance(raw, list):
        default_dep = "全科/内科（综合门诊）"
        for rule in raw:
            if not isinstance(rule, dict):
                continue
            age_min = int(rule.get("age_min", 0))
            age_max = int(rule.get("age_max", 200))
            rule_gender = _normalize_text(str(rule.get("gender", "any")))
            if not (age_min <= int(age) <= age_max):
                continue
            if rule_gender not in {"any", gender_n}:
                continue
            # IMPORTANT: ignore any triage_advice / diagnosis-like content. Only use symptoms + department name.
            kws = rule.get("symptoms", []) or []
            if any(str(kw) in symptom_n for kw in kws):
                return str(rule.get("recommended_department", default_dep))
        return default_dep

    return "全科/内科（综合门诊）"


def get_doctor_schedule(department: str, day_of_week: str) -> Dict[str, Any]:
    """
    返回某科室在某星期的排班信息。
    day_of_week: Mon/Tue/Wed/Thu/Fri/Sat/Sun 或 中文（周一..周日）
    """
    day_n = _normalize_text(day_of_week)
    cn_map = {
        "周一": "Mon",
        "星期一": "Mon",
        "礼拜一": "Mon",
        "周二": "Tue",
        "星期二": "Tue",
        "周三": "Wed",
        "星期三": "Wed",
        "周四": "Thu",
        "星期四": "Thu",
        "周五": "Fri",
        "星期五": "Fri",
        "周六": "Sat",
        "星期六": "Sat",
        "周日": "Sun",
        "星期日": "Sun",
        "周天": "Sun",
        "星期天": "Sun",
    }
    day_key = cn_map.get(day_of_week.strip(), None) or cn_map.get(day_of_week, None)
    if not day_key:
        # accept short forms
        day_key = day_n[:3].capitalize()
    if day_key not in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}:
        day_key = "Mon"

    data = _load_doctor_schedules()

    # Support two formats:
    # 1) {"departments": {"神经内科": {"doctors":[{"name":..,"schedule":{"Mon":"08:30-12:00"}}]}}}
    # 2) [{"department":"神经内科","doctor_name":"...","title":"...","schedule":{"Monday_AM":{"status":"有号","fee":80}}}, ...]
    if isinstance(data, dict):
        dept = data.get("departments", {}).get(department)
        if not dept:
            return {"department": department, "day": day_key, "available": [], "note": "未找到该科室排班（mock 数据）。"}

        available = []
        for d in dept.get("doctors", []) or []:
            slot = (d.get("schedule") or {}).get(day_key, "休")
            if slot and slot != "休":
                available.append({"name": d.get("name"), "title": d.get("title"), "time": slot})

        return {"department": department, "day": day_key, "available": available}

    if isinstance(data, list):
        day_full = {
            "Mon": "Monday",
            "Tue": "Tuesday",
            "Wed": "Wednesday",
            "Thu": "Thursday",
            "Fri": "Friday",
            "Sat": "Saturday",
            "Sun": "Sunday",
        }[day_key]
        slots_for_day = [f"{day_full}_AM", f"{day_full}_PM", f"{day_full}_Night"]

        available = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if str(row.get("department", "")).strip() != str(department).strip():
                continue
            sched = row.get("schedule") or {}
            for slot_key in slots_for_day:
                info = sched.get(slot_key)
                if not isinstance(info, dict):
                    continue
                status = str(info.get("status", "")).strip()
                fee = info.get("fee", None)
                if not status or status.lower() == "off" or status in {"停诊", "已约满", "无号", "暂无号源"}:
                    continue
                fee_text = f"（¥{fee}）" if isinstance(fee, (int, float)) and fee else ""
                # e.g. Monday_AM -> AM
                time_label = slot_key.split("_", 1)[1]
                available.append(
                    {
                        "name": row.get("doctor_name"),
                        "title": row.get("title"),
                        "time": f"{time_label}：{status}{fee_text}",
                        "sub_specialty": row.get("sub_specialty"),
                    }
                )

        if not available:
            return {"department": department, "day": day_key, "available": [], "note": "当天暂无可用号源/排班信息（mock 数据）。"}
        return {"department": department, "day": day_key, "available": available}

    return {"department": department, "day": day_key, "available": [], "note": "排班数据格式无法识别（mock 数据）。"}


def humanize_schedule(schedule: Dict[str, Any]) -> str:
    dept = schedule.get("department", "")
    day = schedule.get("day", "")
    available = schedule.get("available", []) or []
    if not available:
        note = schedule.get("note") or "当天暂无排班信息（mock 数据）。"
        return f"{dept}（{day}）\n{note}"
    lines = [f"{dept}（{day}）值班医生："]
    for x in available:
        sub = x.get("sub_specialty")
        sub_txt = f"｜{sub}" if sub else ""
        lines.append(f"- {x.get('name')}（{x.get('title')}）{sub_txt}：{x.get('time')}")
    return "\n".join(lines)
