from __future__ import annotations

import re
from typing import List, Optional

from .schema import ConversationState

REQUIRED_FIELDS = ["role", "job_level"]

FIELD_PROMPTS = {
    "role": [
        "What job title or function should this assessment support?",
        "Which role are you hiring or assessing for? For example, backend engineer, senior leader, or customer service.",
        "Please describe the role or job family this assessment should be used for.",
    ],
    "job_level": [
        "What seniority level is this role? Entry-level, mid-level, manager, director, or executive.",
        "Is this for an entry-level, mid-level, managerial, director, or executive position?",
        "Can you clarify the target level of experience for this role?",
    ],
    "purpose": [
        "Should this assessment support hiring/selection or development/coaching?",
        "Is the goal to select candidates or support development and feedback?",
        "Are you looking for this assessment for recruitment, performance development, or succession planning?",
    ],
    "assessment_scope": [
        "Would you like technical assessments only, or a broader battery that also includes personality and communication assessments?",
        "Should I recommend technical assessments only, or a combined technical plus personality/communication battery?",
    ],
    "role_focus": [
        "Is this role primarily frontend, backend, or full-stack?",
        "Should I treat this as a frontend, backend, or full-stack role?",
    ],
    "language": [
        "What language should this assessment be delivered in?",
        "Please confirm the candidate language requirement so I can pick the correct SHL products.",
        "Which language do your candidates need the assessment to support?",
    ],
}

_LANGUAGE_SENSITIVE_TERMS = (
    "contact centre",
    "contact center",
    "customer service",
    "call centre",
    "call center",
    "healthcare",
    "medical",
    "bilingual",
    "accent",
    "language",
)


def needs_clarification(state: ConversationState) -> bool:
    if _needs_assessment_scope_clarification(state):
        return True
    if _needs_role_focus_clarification(state):
        return True
    missing = _missing_core_fields(state)
    if missing:
        return True
    if state.comparison_request and not state.comparison_targets:
        return True
    if _needs_language_clarification(state):
        return True
    return False


def clarification_prompt(state: ConversationState) -> Optional[str]:
    if _needs_assessment_scope_clarification(state) and _needs_role_focus_clarification(state):
        return (
            "Would you like technical assessments only, or a broader battery that also includes personality and communication assessments? "
            "Also, is this role primarily frontend, backend, or full-stack?"
        )
    if _needs_assessment_scope_clarification(state):
        options = FIELD_PROMPTS["assessment_scope"]
        return options[hash(state.role or state.purpose or "assessment_scope") % len(options)]
    if _needs_role_focus_clarification(state):
        options = FIELD_PROMPTS["role_focus"]
        return options[hash(state.role or state.purpose or "role_focus") % len(options)]
    missing = _missing_core_fields(state)
    if missing:
        options = FIELD_PROMPTS[missing[0]]
        if isinstance(options, list):
            return options[hash(state.role or state.purpose or missing[0]) % len(options)]
        return options
    if _needs_language_clarification(state):
        options = FIELD_PROMPTS["language"]
        return options[hash(state.role or state.purpose or state.language or 0) % len(options)]
    if state.comparison_request and not state.comparison_targets:
        return "Which assessments should I compare? Please provide two names or IDs."
    return None


def _needs_assessment_scope_clarification(state: ConversationState) -> bool:
    if state.assessment_scope:
        return False
    if state.assessment_types:
        return False
    text_parts = [state.role, state.role_focus, state.job_level, state.industry, state.purpose]
    if state.technical_skills:
        text_parts.append(" ".join(state.technical_skills))
    text = " ".join(filter(None, text_parts)).lower()
    if not re.search(r"\b(engineer|developer|architect|analyst|programmer|scientist|specialist|manager|lead|director|executive|full-stack|full stack|backend|frontend)\b", text):
        return False
    return any(term in text for term in ("java", "spring", "rest", "angular", "react", "sql", "aws", "docker", "microservice", "full-stack", "full stack", "backend", "frontend", "years", "experience"))


def _needs_role_focus_clarification(state: ConversationState) -> bool:
    text_parts = [state.role, state.role_focus, state.job_level, state.industry, state.purpose]
    text = " ".join(filter(None, text_parts)).lower()
    if not text:
        return False
    if "full-stack" in text or "full stack" in text:
        return True
    if re.search(r"\b(engineer|developer|architect|analyst|programmer|scientist|specialist|backend|frontend)\b", text):
        return True
    return False


def _needs_language_clarification(state: ConversationState) -> bool:
    if state.language:
        return False
    text = " ".join(filter(None, [state.role, state.industry, state.purpose])).lower()
    return any(term in text for term in _LANGUAGE_SENSITIVE_TERMS)


def _missing_core_fields(state: ConversationState) -> List[str]:
    return [field for field in REQUIRED_FIELDS if getattr(state, field) is None]
