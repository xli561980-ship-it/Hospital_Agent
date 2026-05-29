from __future__ import annotations

import re
from typing import Optional


def strip_forbidden_tech(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\b(mock|debug|json|\.json)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"（[^）]*(mock|debug|json|数据源|调试)[^）]*）", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\([^)]*(mock|debug|json|data source)[^)]*\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"signature['\"]?\s*[:=]\s*['\"][A-Za-z0-9+/=\\-_]{80,}['\"]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t


def contains_forbidden_medical(text: str) -> bool:
    t = (text or "").replace("呼吸与危重症医学科", "").replace("危重症医学科", "")
    if re.search(r"(诊断|考虑为|怀疑|可能是|高度怀疑|确诊|你这是|属于|病因|并发症)", t):
        return True
    if re.search(r"(用药|吃药|药物|剂量|口服|注射|输液|激素|抗生素|抗病毒|镇痛|手术|穿刺|治疗|处方)", t):
        return True
    if re.search(r"(严重|极高危|致命|凶险|必须立即|随时可能|黄金期|危重)", t):
        return True
    return False


def compliant_replacement(department: Optional[str]) -> str:
    dep = department or "相关专科门诊"
    return (
        "我可以协助你完成导诊（科室推荐、院内指引与挂号/号源查询）。\n"
        "但我不能进行疾病诊断或提供用药/治疗建议。\n"
        f"建议你前往 **{dep}** 由线下医生进一步评估。"
    )

