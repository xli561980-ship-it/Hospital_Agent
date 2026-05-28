from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage


def _load_env() -> None:
    # Optional: load .env if present (safe: usually gitignored)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


_load_env()


@dataclass
class CaseResult:
    idx: int
    query: str
    expected_intent: Optional[str]
    expected_department: Optional[str]
    expected_location: Optional[str]
    is_emergency_expected: Optional[bool]

    prompt: str
    predicted_intent: Optional[str]
    predicted_department: Optional[str]
    predicted_locations: list[str]
    predicted_reply: str
    state: dict[str, Any]

    intent_ok: Optional[bool]
    dept_ok: Optional[bool]
    location_ok: Optional[bool]
    emergency_ok: Optional[bool]
    error: Optional[str] = None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm_space(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("—", "-").replace("－", "-").replace("–", "-")
    return s.strip()


def _normalize_intent(intent_from_state: Optional[str]) -> Optional[str]:
    """
    agent.py uses: location_query / triage_consult / other
    eval_dataset uses: location / triage / other
    """
    if not intent_from_state:
        return None
    m = str(intent_from_state).strip().lower()
    if m == "location_query":
        return "location"
    if m == "triage_consult":
        return "triage"
    if m == "other":
        return "other"
    return m


def _format_prompt(sample: dict[str, Any], include_profile: bool) -> str:
    q = str(sample.get("query") or "").strip()
    if not include_profile:
        return q

    age = sample.get("age")
    gender = str(sample.get("gender") or "").strip().lower()
    pregnancy = str(sample.get("pregnancy_status") or "").strip().lower()

    gender_cn = ""
    if gender in {"male", "m", "男"}:
        gender_cn = "男"
    elif gender in {"female", "f", "女"}:
        gender_cn = "女"

    profile_bits: list[str] = []
    if isinstance(age, int) and 0 <= age <= 120:
        profile_bits.append(f"{age}岁")
    if gender_cn:
        profile_bits.append(gender_cn)
    if pregnancy in {"yes", "y", "true", "pregnant"}:
        profile_bits.append("我怀孕了")
    elif pregnancy in {"no", "n", "false"}:
        # keep minimal; no need to say "not pregnant"
        pass

    if not profile_bits:
        return q
    # Put profile as a natural user statement to reduce pollution of "symptom" extraction.
    profile = " ".join(profile_bits).strip()
    return f"我{profile}。{q}".strip()


def _extract_last_ai_text(state: dict[str, Any]) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        # LangChain BaseMessage: .type == "ai"
        if getattr(m, "type", "") == "ai":
            return str(getattr(m, "content", "") or "").strip()
    return ""


def _predict_emergency(
    department: Optional[str],
    reply_text: str,
    state: Optional[dict[str, Any]] = None,
) -> bool:
    if isinstance(state, dict) and state.get("is_emergency") is True:
        return True
    t = _norm_space(f"{department or ''}\n{reply_text or ''}")
    if not t:
        return False
    # conservative keywords: "急诊/抢救/红区/胸痛中心/卒中中心/创伤/重症/大血管"
    return bool(re.search(r"(急诊|抢救|红区|胸痛中心|卒中中心|创伤|重症|大血管)", t))


def _extract_location_services(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in state.get("location_results") or []:
        if not isinstance(item, dict):
            continue
        service = str(item.get("service") or "").strip()
        building = str(item.get("building") or "").strip()
        floor = str(item.get("floor") or "").strip()
        room = str(item.get("room") or "").strip()
        text = " ".join([x for x in [service, building, floor, room] if x]).strip()
        if text:
            out.append(text)
    return out


def _judge_location(expected_location: str, state: dict[str, Any], reply_text: str) -> bool:
    exp = _norm_space(expected_location)
    if not exp:
        return False
    hay_parts: list[str] = [reply_text or ""]
    for item in state.get("location_results") or []:
        if isinstance(item, dict):
            hay_parts.extend(str(item.get(k) or "") for k in ("service", "building", "floor", "room", "directions"))
    hay = _norm_space("\n".join(hay_parts))
    return exp in hay


class DepartmentJudge:
    def __init__(self, enabled: bool = True, model_hint: str | None = None, sleep_s: float = 0.0):
        self.enabled = enabled
        self.model_hint = model_hint
        self.sleep_s = sleep_s
        self._llm = None
        self._cache: dict[tuple[str, str], bool] = {}

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            # Reuse agent's provider selection & env checks
            from agent import _get_llm  # type: ignore

            self._llm = _get_llm()
            return self._llm
        except Exception:
            self._llm = None
            return None

    def _heuristic(self, expected_department: str, actual_text: str, actual_department: Optional[str]) -> bool:
        exp = (expected_department or "").strip()
        if not exp:
            return False
        hay = "\n".join([actual_department or "", actual_text or ""]).strip()
        if not hay:
            return False

        def clean_dep(s: str) -> str:
            s = (s or "").strip()
            s = s.replace("（", "(").replace("）", ")")
            # drop parenthetical qualifiers like "(绿色通道)"
            s = re.sub(r"\([^)]*\)", "", s)
            s = re.sub(r"\s+", "", s)
            s = s.replace("—", "-").replace("－", "-").replace("–", "-")
            return s.strip()

        exp_n = clean_dep(exp)
        hay_n = clean_dep(hay)

        # Treat common "same-family" emergency departments as equivalent.
        # e.g. 妇产科急诊 vs 产科急诊(绿色通道) should be OK.
        def family(dep: str) -> Optional[str]:
            d = clean_dep(dep)
            if not d:
                return None
            if re.search(r"(妇产科急诊|妇科急诊|产科急诊)", d):
                return "obgyn_er"
            if re.search(r"(儿科急诊|小儿.*急诊)", d):
                return "peds_er"
            return None

        fam_exp = family(exp_n)
        fam_hay = family(hay_n)
        if fam_exp and fam_exp == fam_hay:
            return True

        # handle expected alternatives like "A/B"
        alts = [x.strip() for x in re.split(r"[／/]|或", exp) if x.strip()]
        if not alts:
            alts = [exp]

        for a in alts:
            a_n = _norm_space(a)
            if a_n and a_n in hay_n:
                return True

        # weak fuzzy: compare after removing common prefixes
        def simplify(s: str) -> str:
            s = clean_dep(s)
            s = re.sub(r"^(急诊科-?)", "", s)
            return s

        exp_s = simplify(exp)
        hay_s = simplify(hay)
        return bool(exp_s) and exp_s in hay_s

    def judge(self, expected_department: str, actual_text: str, actual_department: Optional[str]) -> bool:
        key = (expected_department or "", actual_text or "", actual_department or "")
        if key in self._cache:
            return self._cache[key]

        # If disabled, fall back to heuristic
        if not self.enabled:
            ok = self._heuristic(expected_department, actual_text, actual_department)
            self._cache[key] = ok
            return ok

        llm = self._get_llm()
        if llm is None:
            ok = self._heuristic(expected_department, actual_text, actual_department)
            self._cache[key] = ok
            return ok

        prompt = (
            "你是导诊评测裁判。你的任务是判断：下面这条导诊回复是否“推荐到了期望科室”。\n"
            "注意：允许同义/别名/更细分或更上位的科室表述，只要对患者导向是正确的。\n"
            "如果明显推荐错误科室、或没有推荐任何科室，请判 False。\n\n"
            f"期望科室：{expected_department}\n"
            f"Agent提取的department字段：{actual_department or ''}\n"
            f"Agent最终回复内容：{actual_text}\n\n"
            "请只输出一个 JSON，对象包含一个布尔字段 ok。\n"
            "示例：{\"ok\": true}\n"
        )

        try:
            msg = llm.invoke(prompt)  # type: ignore[call-arg]
            content = str(getattr(msg, "content", "") or "").strip()
            m = re.search(r"\{[\s\S]*\}", content)
            obj = json.loads(m.group(0) if m else content)
            ok = bool(obj.get("ok"))
        except Exception:
            ok = self._heuristic(expected_department, actual_text, actual_department)

        self._cache[key] = ok
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        return ok


def _color(s: str, c: str, enable: bool) -> str:
    if not enable:
        return s
    codes = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1"}
    code = codes.get(c)
    return f"\x1b[{code}m{s}\x1b[0m" if code else s


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate triage/location agent on eval_dataset.json")
    p.add_argument("--dataset", type=str, default="eval_dataset.json", help="Path to eval_dataset.json")
    p.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 means all)")
    p.add_argument("--include-profile", action="store_true", help="Append age/gender/pregnancy into prompt")
    p.add_argument("--judge-llm", action="store_true", help="Use LLM-as-a-Judge for department correctness")
    p.add_argument("--judge-sleep", type=float, default=0.0, help="Sleep seconds between judge calls")
    p.add_argument("--timeout-s", type=int, default=90, help="Per-case timeout seconds (graph invoke + judge)")
    p.add_argument(
        "--no-llm-run",
        action="store_true",
        help="Skip graph's llm_responder (faster; evaluates intent/department from extracted state only)",
    )
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = p.parse_args()

    dataset_path = Path(args.dataset).resolve()
    samples = _read_json(dataset_path)
    if not isinstance(samples, list):
        print(f"Dataset must be a list of cases, got: {type(samples)}", file=sys.stderr)
        return 2

    use_color = (not args.no_color) and sys.stdout.isatty()

    os.environ.setdefault("HOSPITAL_AGENT_NOW", "2026-05-11T09:00:00")

    from agent import GRAPH  # import after env load
    from agent import (  # type: ignore
        classify_intent,
        extract_location,
        extract_schedule,
        extract_triage,
        ingest_and_transition,
    )

    judge = DepartmentJudge(enabled=bool(args.judge_llm), sleep_s=float(args.judge_sleep))

    results: list[CaseResult] = []
    n = len(samples) if not args.limit else min(len(samples), int(args.limit))

    for i in range(n):
        s = samples[i] if isinstance(samples[i], dict) else {}
        query = str(s.get("query") or "").strip()
        prompt_text = _format_prompt(s, include_profile=bool(args.include_profile))

        expected_intent = s.get("expected_intent")
        expected_department = s.get("expected_department")
        expected_location = s.get("expected_location")
        is_emergency_expected = s.get("is_emergency_expected")

        thread_id = f"eval-{uuid.uuid4()}"

        # progress (stderr so piping stdout keeps report clean)
        print(f"[{i+1}/{n}] running...", file=sys.stderr, flush=True)

        try:
            # Per-case timeout to avoid hanging on networked LLM calls
            def _alarm_handler(_signum, _frame):
                raise TimeoutError(f"case timeout after {int(args.timeout_s)}s")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(args.timeout_s))

            if args.no_llm_run:
                # Deterministic partial run: classify (regex only) -> ingest -> extract_* (skip llm_responder).
                state: dict[str, Any] = {"messages": [HumanMessage(content=prompt_text)]}
                prev_intent = os.environ.get("USE_INTENT_LLM")
                prev_triage = os.environ.get("USE_TRIAGE_LLM")
                os.environ["USE_INTENT_LLM"] = "false"
                os.environ["USE_TRIAGE_LLM"] = "false"
                try:
                    state = classify_intent(state)  # type: ignore[assignment]
                    state = ingest_and_transition(state)  # type: ignore[assignment]
                    action = state.get("action") or "TRIAGE"
                    if action == "LOCATION":
                        state = extract_location(state)  # type: ignore[assignment]
                    elif action == "SCHEDULE":
                        state = extract_schedule(state)  # type: ignore[assignment]
                    else:
                        state = extract_triage(state)  # type: ignore[assignment]
                finally:
                    if prev_intent is None:
                        os.environ.pop("USE_INTENT_LLM", None)
                    else:
                        os.environ["USE_INTENT_LLM"] = prev_intent
                    if prev_triage is None:
                        os.environ.pop("USE_TRIAGE_LLM", None)
                    else:
                        os.environ["USE_TRIAGE_LLM"] = prev_triage
            else:
                state = GRAPH.invoke(
                    {"messages": [("human", prompt_text)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            if not isinstance(state, dict):
                state = {"_raw": state}
            predicted_intent = _normalize_intent(state.get("intent"))
            predicted_department = state.get("department")
            predicted_reply = _extract_last_ai_text(state)
            predicted_locations = _extract_location_services(state)

            # intent correctness (normalize expected where applicable)
            exp_i = str(expected_intent).strip().lower() if expected_intent is not None else None
            intent_ok = None if exp_i is None else (predicted_intent == exp_i)

            # dept correctness
            dept_ok = None
            exp_dep = expected_department if isinstance(expected_department, str) else None
            if exp_dep is not None:
                dept_ok = judge.judge(exp_dep, predicted_reply, predicted_department)

            # location correctness (top-k containment)
            location_ok = None
            exp_loc = expected_location if isinstance(expected_location, str) else None
            if exp_loc is not None:
                location_ok = _judge_location(exp_loc, state, predicted_reply)

            # emergency interception
            emergency_ok = None
            if isinstance(is_emergency_expected, bool):
                pred_em = _predict_emergency(predicted_department, predicted_reply, state)
                emergency_ok = (pred_em is True) if is_emergency_expected else True

            results.append(
                CaseResult(
                    idx=i,
                    query=query,
                    expected_intent=exp_i,
                    expected_department=exp_dep,
                    expected_location=exp_loc,
                    is_emergency_expected=is_emergency_expected if isinstance(is_emergency_expected, bool) else None,
                    prompt=prompt_text,
                    predicted_intent=predicted_intent,
                    predicted_department=str(predicted_department).strip() if predicted_department is not None else None,
                    predicted_locations=predicted_locations,
                    predicted_reply=predicted_reply,
                    state=state,
                    intent_ok=intent_ok,
                    dept_ok=dept_ok,
                    location_ok=location_ok,
                    emergency_ok=emergency_ok,
                )
            )
        except Exception as e:
            results.append(
                CaseResult(
                    idx=i,
                    query=query,
                    expected_intent=str(expected_intent).strip().lower() if expected_intent is not None else None,
                    expected_department=expected_department if isinstance(expected_department, str) else None,
                    expected_location=expected_location if isinstance(expected_location, str) else None,
                    is_emergency_expected=is_emergency_expected if isinstance(is_emergency_expected, bool) else None,
                    prompt=prompt_text,
                    predicted_intent=None,
                    predicted_department=None,
                    predicted_locations=[],
                    predicted_reply="",
                    state={},
                    intent_ok=False if expected_intent is not None else None,
                    dept_ok=False if expected_department is not None else None,
                    location_ok=False if isinstance(expected_location, str) else None,
                    emergency_ok=False if isinstance(is_emergency_expected, bool) else None,
                    error=str(e),
                )
            )
        finally:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)  # type: ignore[arg-type]
            except Exception:
                pass

    # metrics
    intent_total = sum(1 for r in results if r.intent_ok is not None)
    intent_correct = sum(1 for r in results if r.intent_ok is True)

    dept_total = sum(1 for r in results if r.dept_ok is not None)
    dept_correct = sum(1 for r in results if r.dept_ok is True)

    location_total = sum(1 for r in results if r.location_ok is not None)
    location_correct = sum(1 for r in results if r.location_ok is True)

    emer_total = sum(1 for r in results if r.is_emergency_expected is True and r.emergency_ok is not None)
    emer_correct = sum(1 for r in results if r.is_emergency_expected is True and r.emergency_ok is True)

    print()
    print(_color("========== 导诊 Agent 量化评测 ==========", "bold", use_color))
    print(f"Dataset: {dataset_path}")
    print(f"Cases:   {len(results)}")
    print(f"Profile: {'ON' if args.include_profile else 'OFF'}  (是否把 age/gender/pregnancy 拼入 prompt)")
    print(f"Judge:   {'LLM' if args.judge_llm else 'heuristic'}")
    print()

    print(_color("### 总体指标", "cyan", use_color))
    print(f"- 意图识别准确率 (%): {_pct(intent_correct, intent_total):.2f}  ({intent_correct}/{intent_total})")
    print(f"- 科室推荐准确率 (%): {_pct(dept_correct, dept_total):.2f}  ({dept_correct}/{dept_total})")
    print(f"- 位置检索准确率 (%): {_pct(location_correct, location_total):.2f}  ({location_correct}/{location_total})")
    print(f"- 急症拦截成功率 (%): {_pct(emer_correct, emer_total):.2f}  ({emer_correct}/{emer_total})")
    print()

    # failures
    intent_fail = [r for r in results if r.intent_ok is False]
    dept_fail = [r for r in results if r.dept_ok is False]
    location_fail = [r for r in results if r.location_ok is False]
    emer_fail = [r for r in results if (r.is_emergency_expected is True and r.emergency_ok is False)]

    def show_fail(title: str, items: list[CaseResult], max_items: int = 30) -> None:
        print(_color(title, "cyan", use_color))
        if not items:
            print(_color("  (无)", "green", use_color))
            print()
            return
        for r in items[:max_items]:
            badge = _color("FAIL", "red", use_color)
            err = f" | error={r.error}" if r.error else ""
            print(f"- [{badge}] case#{r.idx}  q={r.query}{err}")
            if r.expected_intent is not None:
                print(f"  expected_intent={r.expected_intent}  predicted_intent={r.predicted_intent}")
            if r.expected_department is not None:
                print(f"  expected_department={r.expected_department}")
                print(f"  predicted_department={r.predicted_department}")
            if r.expected_location is not None:
                print(f"  expected_location={r.expected_location}")
                print(f"  predicted_locations={'; '.join(r.predicted_locations[:3])}")
            if r.is_emergency_expected is True:
                print(
                    "  is_emergency_expected=True  "
                    f"predicted_emergency={_predict_emergency(r.predicted_department, r.predicted_reply, r.state)}"
                )
            if r.predicted_reply:
                short = r.predicted_reply.replace("\n", " ").strip()
                if len(short) > 220:
                    short = short[:220] + "..."
                print(f"  reply={short}")
        if len(items) > max_items:
            print(_color(f"  ... 还有 {len(items) - max_items} 条失败用例未展示（用 --limit 或改脚本阈值）", "dim", use_color))
        print()

    show_fail("### 失败用例：意图识别", intent_fail)
    show_fail("### 失败用例：科室推荐", dept_fail)
    show_fail("### 失败用例：位置检索", location_fail)
    show_fail("### 失败用例：急症拦截（仅统计期望为急症的 case）", emer_fail)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
