from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


STATE_FIELDS = [
    "intent",
    "intent_source",
    "action",
    "current_phase",
    "triage_match_source",
    "department",
    "is_emergency",
    "safety_boundary_violation",
    "age",
    "gender",
    "symptom",
    "triage_positive_findings",
    "triage_negative_findings",
    "triage_followup_questions",
    "schedule_candidates",
    "schedule_window",
]


def _force_fallback_env() -> None:
    os.environ["USE_INTENT_LLM"] = "false"
    os.environ["USE_TRIAGE_LLM"] = "false"
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["GOOGLE_API_KEY"] = ""
    os.environ.setdefault("HOSPITAL_AGENT_NOW", "2026-05-11T09:00:00")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm(text: Any) -> str:
    t = str(text or "")
    t = t.replace("（", "(").replace("）", ")")
    t = t.replace("—", "-").replace("－", "-").replace("–", "-")
    return re.sub(r"\s+", "", t).strip()


def _last_ai_text(state: dict[str, Any]) -> str:
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "ai":
            return str(getattr(msg, "content", "") or "").strip()
    return ""


def _state_subset(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in STATE_FIELDS:
        value = state.get(field)
        if field in {"schedule_candidates", "schedule_window"} and isinstance(value, list):
            out[field] = value[:2]
        else:
            out[field] = value
    return out


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(str(p) and str(p) in text for p in phrases)


def _contains_none(text: str, phrases: list[str]) -> bool:
    return all(not str(p) or str(p) not in text for p in phrases)


def _department_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return any(_department_matches(x, actual) for x in expected)
    exp = _norm(expected)
    act = _norm(actual)
    if not exp:
        return not act
    if not act:
        return False
    return exp == act or exp in act or act in exp


def _field_matches(field: str, expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return any(_field_matches(field, item, actual) for item in expected)
    if expected is None:
        return actual is None or actual == ""
    if isinstance(expected, bool):
        return bool(actual) is expected
    if field == "department":
        return _department_matches(expected, actual)
    return str(actual) == str(expected)


def _list_contains_any(items: Any, expected: list[str]) -> bool:
    hay = " ".join(str(x) for x in (items or []))
    return _contains_any(hay, expected)


def _actual_should_ask_clarification(state: dict[str, Any]) -> bool:
    return bool(
        state.get("triage_match_source") == "clarification_required"
        or state.get("triage_followup_questions")
    )


def _check_expect(state: dict[str, Any], reply: str, expect: dict[str, Any]) -> tuple[bool, list[str], dict[str, bool]]:
    failures: list[str] = []
    signal_checks: dict[str, bool] = {}

    direct_fields = [
        "intent",
        "action",
        "current_phase",
        "triage_match_source",
        "department",
        "is_emergency",
        "safety_boundary_violation",
    ]
    for field in direct_fields:
        if field not in expect:
            continue
        if not _field_matches(field, expect.get(field), state.get(field)):
            failures.append(f"{field}: expected {expect.get(field)!r}, got {state.get(field)!r}")

    if "should_ask_clarification" in expect:
        actual = _actual_should_ask_clarification(state)
        expected = bool(expect.get("should_ask_clarification"))
        ok = actual == expected
        signal_checks["clarification"] = ok
        if not ok:
            failures.append(f"should_ask_clarification: expected {expected}, got {actual}")

    if "reply_contains_any" in expect:
        phrases = [str(x) for x in expect.get("reply_contains_any") or []]
        ok = _contains_any(reply, phrases)
        signal_checks["reply_contains_any"] = ok
        if not ok:
            failures.append(f"reply_contains_any: none of {phrases!r} found")

    if "reply_not_contains_any" in expect:
        phrases = [str(x) for x in expect.get("reply_not_contains_any") or []]
        ok = _contains_none(reply, phrases)
        signal_checks["reply_not_contains_any"] = ok
        if not ok:
            failures.append(f"reply_not_contains_any: found forbidden phrase from {phrases!r}")

    if "negative_findings_contains_any" in expect:
        phrases = [str(x) for x in expect.get("negative_findings_contains_any") or []]
        ok = _list_contains_any(state.get("triage_negative_findings"), phrases)
        signal_checks["negative_findings"] = ok
        if not ok:
            failures.append(f"negative_findings_contains_any: none of {phrases!r} found")

    if "positive_findings_contains_any" in expect:
        phrases = [str(x) for x in expect.get("positive_findings_contains_any") or []]
        ok = _list_contains_any(state.get("triage_positive_findings"), phrases)
        signal_checks["positive_findings"] = ok
        if not ok:
            failures.append(f"positive_findings_contains_any: none of {phrases!r} found")

    return not failures, failures, signal_checks


def _case_has_no_infinite_clarification(turn_states: list[dict[str, Any]]) -> bool:
    max_run = 0
    cur = 0
    for state in turn_states:
        if _actual_should_ask_clarification(state):
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run <= 2


def _pct(done: int, total: int) -> float:
    return 100.0 * done / total if total else 0.0


def _print_failure(failure: dict[str, Any]) -> None:
    print()
    print(f"[FAIL] {failure['case_id']} turn {failure['turn_index']}")
    print(f"User: {failure['user']}")
    print("Expected:")
    print(json.dumps(failure["expected"], ensure_ascii=False, indent=2))
    print("Actual state subset:")
    print(json.dumps(failure["actual_state"], ensure_ascii=False, indent=2, default=str))
    print("Reply:")
    print(failure["reply"])
    print("Reasons:")
    for reason in failure["reasons"]:
        print(f"- {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn state, clarification, and continuity behavior.")
    parser.add_argument("--dataset", default="eval_multiturn_cases.json", help="Path to multi-turn eval dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases")
    parser.add_argument("--no-color", action="store_true", help="Accepted for CLI consistency; output is plain text")
    parser.add_argument("--verbose", action="store_true", help="Print every turn state subset")
    args = parser.parse_args()

    _force_fallback_env()
    from agent import GRAPH

    dataset_path = Path(args.dataset).resolve()
    cases = _read_json(dataset_path)
    if not isinstance(cases, list):
        raise SystemExit("Dataset must be a list of cases")
    if args.limit:
        cases = cases[: max(0, int(args.limit))]

    total_cases = 0
    passed_cases = 0
    total_turns = 0
    passed_turns = 0
    failures: list[dict[str, Any]] = []

    clarification_total = 0
    clarification_pass = 0
    followup_total = 0
    followup_pass = 0
    red_flag_total = 0
    red_flag_pass = 0
    schedule_total = 0
    schedule_pass = 0
    no_infinite_total = 0
    no_infinite_pass = 0

    for case in cases:
        if not isinstance(case, dict):
            continue
        total_cases += 1
        case_id = str(case.get("id") or f"case_{total_cases}")
        thread_id = f"eval-multiturn-{case_id}-{uuid.uuid4().hex}"
        case_ok = True
        turn_states: list[dict[str, Any]] = []

        for idx, turn in enumerate(case.get("turns") or [], start=1):
            if not isinstance(turn, dict):
                continue
            total_turns += 1
            user = str(turn.get("user") or "")
            expect = turn.get("expect") or {}
            tags = set(turn.get("tags") or [])
            try:
                state = GRAPH.invoke(
                    {"messages": [("human", user)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            except Exception as exc:
                state = {}
                reply = ""
                ok = False
                reasons = [f"GRAPH.invoke raised {type(exc).__name__}: {exc}"]
                checks: dict[str, bool] = {}
            else:
                reply = _last_ai_text(state)
                turn_states.append(state)
                ok, reasons, checks = _check_expect(state, reply, expect)

            if "should_ask_clarification" in expect:
                clarification_total += 1
                clarification_pass += int(checks.get("clarification", False))
            if "followup_resolution" in tags:
                followup_total += 1
                followup_pass += int(ok)
            if "red_flag_escalation" in tags:
                red_flag_total += 1
                red_flag_pass += int(ok)
            if "schedule_continuity" in tags:
                schedule_total += 1
                schedule_pass += int(ok)

            if ok:
                passed_turns += 1
            else:
                case_ok = False
                failures.append(
                    {
                        "case_id": case_id,
                        "turn_index": idx,
                        "user": user,
                        "expected": expect,
                        "actual_state": _state_subset(state),
                        "reply": reply,
                        "reasons": reasons,
                    }
                )

            if args.verbose:
                print(f"\n[{case_id} turn {idx}] {user}")
                print(json.dumps(_state_subset(state), ensure_ascii=False, indent=2, default=str))
                print(reply)

        no_infinite_total += 1
        no_infinite_ok = _case_has_no_infinite_clarification(turn_states)
        no_infinite_pass += int(no_infinite_ok)
        case_ok = case_ok and no_infinite_ok
        if case_ok:
            passed_cases += 1

    print("========== Multi-turn Evaluation ==========")
    print(f"Dataset: {dataset_path}")
    print(f"total_cases: {total_cases}")
    print(f"total_turns: {total_turns}")
    print(f"turn_pass_rate (%): {_pct(passed_turns, total_turns):.2f}  ({passed_turns}/{total_turns})")
    print(f"case_pass_rate (%): {_pct(passed_cases, total_cases):.2f}  ({passed_cases}/{total_cases})")
    print(
        "clarification_trigger_accuracy (%): "
        f"{_pct(clarification_pass, clarification_total):.2f}  ({clarification_pass}/{clarification_total})"
    )
    print(
        "followup_resolution_accuracy (%): "
        f"{_pct(followup_pass, followup_total):.2f}  ({followup_pass}/{followup_total})"
    )
    print(
        "red_flag_escalation_accuracy (%): "
        f"{_pct(red_flag_pass, red_flag_total):.2f}  ({red_flag_pass}/{red_flag_total})"
    )
    print(
        "schedule_continuity_accuracy (%): "
        f"{_pct(schedule_pass, schedule_total):.2f}  ({schedule_pass}/{schedule_total})"
    )
    print(
        "no_infinite_clarification_rate (%): "
        f"{_pct(no_infinite_pass, no_infinite_total):.2f}  ({no_infinite_pass}/{no_infinite_total})"
    )

    if failures:
        print(f"\nFailures: {len(failures)}")
        for failure in failures:
            _print_failure(failure)
    else:
        print("\nFailures: 0")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
