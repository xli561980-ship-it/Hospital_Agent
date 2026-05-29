from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Intent = Literal["location_query", "triage_consult", "other"]
Phase = Literal["INIT", "TRIAGE", "RECOMMENDED", "SCHEDULE"]
Action = Literal["LOCATION", "TRIAGE", "SCHEDULE", "OTHER"]


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    current_phase: Phase
    intent: Intent
    action: Action
    intent_source: Optional[str]

    age: Optional[int]
    gender: Optional[str]
    pregnancy_status: Optional[str]
    symptom: Optional[str]

    department: Optional[str]
    triage_advice: Optional[str]
    is_emergency: bool
    match_confidence: Optional[float]
    matched_symptoms: list[str]
    matched_rule_id: Optional[str]
    triage_match_source: Optional[str]
    triage_candidate_rules: list[dict]
    triage_llm_reason: Optional[str]
    triage_followup_questions: list[str]
    triage_candidate_departments: list[str]
    triage_possible_conditions: list[str]
    triage_positive_findings: list[str]
    triage_negative_findings: list[str]
    triage_interview_reason: Optional[str]
    primary_symptom: Optional[str]

    location_results: list[dict]

    schedule_day: Optional[str]
    schedule_candidates: list[dict]
    schedule_window: list[dict]

    registration_steps: list[str]
    registration_location: Optional[dict]
    slot_extract_source: Optional[str]
