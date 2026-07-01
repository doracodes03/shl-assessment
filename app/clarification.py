from __future__ import annotations

from typing import List, Optional

from .schema import ConversationState

REQUIRED_FIELDS = ["role", "job_level"]

FIELD_PROMPTS = {
    "role": "Can you tell me the role or job title for the assessment?",
    "job_level": "What seniority level is this for (entry, mid, manager, director, executive)?",
    "purpose": "Is this assessment for selection/hiring or development/training?",
}


def needs_clarification(state: ConversationState) -> bool:
    missing = _missing_core_fields(state)
    if missing:
        return True
    if state.comparison_request and not state.comparison_targets:
        return True
    return False


def clarification_prompt(state: ConversationState) -> Optional[str]:
    missing = _missing_core_fields(state)
    if missing:
        return FIELD_PROMPTS[missing[0]]
    if state.comparison_request and not state.comparison_targets:
        return "Which assessments should I compare? Please provide two names or IDs."
    return None


def _missing_core_fields(state: ConversationState) -> List[str]:
    return [field for field in REQUIRED_FIELDS if getattr(state, field) is None]
