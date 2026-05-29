from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm(s: str) -> str:
    s = str(s or "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("—", "-").replace("－", "-").replace("–", "-")
    return re.sub(r"\s+", "", s).strip()


def _format_prompt(sample: dict[str, Any], include_profile: bool) -> str:
    q = str(sample.get("query") or "").strip()
    if not include_profile:
        return q

    age = sample.get("age")
    gender = str(sample.get("gender") or "").strip().lower()
    pregnancy = str(sample.get("pregnancy_status") or "").strip().lower()

    bits: list[str] = []
    if isinstance(age, int) and 0 <= age <= 120:
        bits.append(f"{age}岁")
    if gender in {"male", "m", "男"}:
        bits.append("男")
    elif gender in {"female", "f", "女"}:
        bits.append("女")
    if pregnancy in {"yes", "y", "true", "pregnant"}:
        bits.append("我怀孕了")

    return f"我{' '.join(bits)}。{q}".strip() if bits else q


def _department_matches(expected: str, actual: str) -> bool:
    exp = _norm(expected)
    act = _norm(actual)
    if not exp or not act:
        return False

    for alt in [x for x in re.split(r"[／/]|或", expected) if str(x).strip()]:
        alt_n = _norm(alt)
        if alt_n and (alt_n in act or act in alt_n):
            return True

    return exp in act or act in exp


def _location_matches(expected: str, item: Any) -> bool:
    if not expected:
        return False
    parts = []
    for attr in ("service", "building", "floor", "room", "directions"):
        parts.append(str(getattr(item, attr, "") or ""))
    hay = _norm(" ".join(parts))
    exp = _norm(expected)
    return bool(exp and exp in hay)


def _expected_rule_ids(sample: dict[str, Any]) -> set[str]:
    raw = sample.get("expected_rule_id")
    if raw is None:
        raw = sample.get("expected_rule_ids")
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(x).strip() for x in raw if str(x).strip()}
    return {str(raw).strip()} if str(raw).strip() else set()


@dataclass
class RetrievalStats:
    total: int = 0
    hit1: int = 0
    hit3: int = 0
    reciprocal_sum: float = 0.0
    precision3_sum: float = 0.0

    def add(self, rank: Optional[int], relevant_in_top3: int) -> None:
        self.total += 1
        if rank == 1:
            self.hit1 += 1
        if rank is not None and rank <= 3:
            self.hit3 += 1
        if rank is not None:
            self.reciprocal_sum += 1.0 / rank
        self.precision3_sum += relevant_in_top3 / 3.0

    def pct(self, n: int) -> float:
        return 100.0 * n / self.total if self.total else 0.0

    def mrr(self) -> float:
        return self.reciprocal_sum / self.total if self.total else 0.0

    def precision3(self) -> float:
        return self.precision3_sum / self.total if self.total else 0.0


def _triage_rank(sample: dict[str, Any], prompt: str, limit: int) -> tuple[Optional[int], int]:
    from agent import _rank_triage_candidates

    expected_rules = _expected_rule_ids(sample)
    expected_department = str(sample.get("expected_department") or "").strip()
    if not expected_rules and not expected_department:
        return None, 0

    candidates = _rank_triage_candidates(
        age=sample.get("age"),
        gender=sample.get("gender"),
        pregnancy_status=sample.get("pregnancy_status"),
        symptom=prompt,
        limit=limit,
    )

    first_rank: Optional[int] = None
    relevant_top3 = 0
    for idx, candidate in enumerate(candidates[:limit], start=1):
        rule = candidate.get("rule") or {}
        dep = str(rule.get("recommended_department") or rule.get("department") or "")
        rule_id = str(candidate.get("rule_id") or "")
        is_hit = rule_id in expected_rules if expected_rules else _department_matches(expected_department, dep)
        if is_hit:
            relevant_top3 += 1 if idx <= 3 else 0
            if first_rank is None:
                first_rank = idx
    return first_rank, relevant_top3


def _location_rank(prompt: str, expected_location: str, limit: int) -> tuple[Optional[int], int]:
    from tools import search_location

    # Keep this retrieval script deterministic and local; search_location will then use its TF-IDF fallback.
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = ""
    try:
        results = search_location(prompt, k=limit)
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key

    first_rank: Optional[int] = None
    relevant_top3 = 0
    for idx, item in enumerate(results[:limit], start=1):
        is_hit = _location_matches(expected_location, item)
        if is_hit:
            relevant_top3 += 1 if idx <= 3 else 0
            if first_rank is None:
                first_rank = idx
    return first_rank, relevant_top3


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate lightweight retrieval metrics for Hospital_Agent.")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to eval dataset JSON")
    parser.add_argument("--include-profile", action="store_true", help="Append age/gender/pregnancy to query text")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases")
    parser.add_argument("--k", type=int, default=3, help="Retrieval depth; metrics report @1 and @3")
    parser.add_argument("--no-color", action="store_true", help="Accepted for CLI consistency; output is plain text")
    args = parser.parse_args()

    samples = _read_json(Path(args.dataset).resolve())
    if not isinstance(samples, list):
        raise SystemExit("Dataset must be a list of cases")

    n = len(samples) if not args.limit else min(len(samples), int(args.limit))
    rule_stats = RetrievalStats()
    location_stats = RetrievalStats()
    k = max(3, int(args.k))

    for sample in samples[:n]:
        if not isinstance(sample, dict):
            continue
        prompt = _format_prompt(sample, include_profile=bool(args.include_profile))

        if sample.get("expected_department"):
            rank, relevant_top3 = _triage_rank(sample, prompt, limit=k)
            rule_stats.add(rank, relevant_top3)

        if sample.get("expected_location"):
            rank, relevant_top3 = _location_rank(prompt, str(sample.get("expected_location") or ""), limit=k)
            location_stats.add(rank, relevant_top3)

    print("========== Retrieval Evaluation ==========")
    print(f"Dataset: {Path(args.dataset).resolve()}")
    print(f"Cases:   {n}")
    print(f"Profile: {'ON' if args.include_profile else 'OFF'}")
    print()
    print("### Triage Rule Retrieval")
    print(f"- rule_recall@1 (%): {rule_stats.pct(rule_stats.hit1):.2f}  ({rule_stats.hit1}/{rule_stats.total})")
    print(f"- rule_recall@3 (%): {rule_stats.pct(rule_stats.hit3):.2f}  ({rule_stats.hit3}/{rule_stats.total})")
    print(f"- rule_mrr: {rule_stats.mrr():.4f}")
    print(f"- rule_precision@3: {rule_stats.precision3():.4f}")
    print()
    print("### Location Retrieval")
    print(
        f"- location_recall@1 (%): {location_stats.pct(location_stats.hit1):.2f}  "
        f"({location_stats.hit1}/{location_stats.total})"
    )
    print(
        f"- location_recall@3 (%): {location_stats.pct(location_stats.hit3):.2f}  "
        f"({location_stats.hit3}/{location_stats.total})"
    )
    print(f"- location_mrr: {location_stats.mrr():.4f}")
    print(f"- location_precision@3: {location_stats.precision3():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
