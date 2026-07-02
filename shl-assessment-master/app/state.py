from __future__ import annotations

import re
from typing import List, Optional

from .catalog import CatalogItem
from .retrieval import SearchFilters
from .schema import ConversationState, Message

_JOB_LEVEL_MAP = {
    r"\b(graduate|graduate[- ]level|graduate trainee|recent graduate|recent graduates|entry|entry[- ]level|junior)\b": "Entry-Level",
    r"\b(mid|mid[- ]level|mid[- ]professional|professional|staff)\b": "Mid-Professional",
    r"\b(senior|senior[- ]ic|lead|leadership)\b": "Manager",
    r"\b(manager|management)\b": "Manager",
    r"\b(director|senior director)\b": "Director",
    r"\b(executive|vp|vice president|chief|c-level)\b": "Executive",
}

_PURPOSE_PATTERNS = {
    r"\b(selection|selecting|select)\b": "selection",
    r"\b(hire|hiring|screening|recruitment|candidate|candidates)\b": "selection",
    r"\b(development|developmental|training|growth|upskill|coaching|talent development|re[- ]skill|reskill)\b": "development",
}

_ASSESSMENT_SCOPE_PATTERNS = {
    "technical_only": [
        r"\b(?:technical(?:\s+assessments?)?\s+only|only technical(?:\s+assessments?)?)\b",
    ],
    "technical_and_behavioral": [
        r"\b(?:technical(?:\s+and|\+)?\s*(?:personality|communication|behaviour|behavior))\b",
        r"\b(?:personality|communication|behaviour|behavior)(?:\s+and|\+)?\s*technical\b",
        r"\b(?:personality\s+and\s+communication|communication\s+and\s+personality)\b",
    ],
}

_ROLE_FOCUS_PATTERNS = {
    "frontend": [
        r"\b(front[- ]end|frontend|ui|ux|angular|react|typescript|javascript|css|web)\b",
    ],
    "backend": [
        r"\b(back[- ]end|backend|api|java|spring|sql|aws|docker|microservice|rest|server|node|python)\b",
    ],
    "full-stack": [
        r"\b(full[- ]stack|full stack)\b",
    ],
}

_LANGUAGE_MAP = {
    r"\benglish\b": "English",
    r"\bspanish\b": "Spanish",
    r"\bfrench\b": "French",
    r"\bgerman\b": "German",
    r"\bchinese\b|\bmandarin\b": "Chinese",
}

_ACCENT_MAP = {
    r"\b(?:us|usa|u\.s\.a?|american)\b": "US",
    r"\b(?:uk|u\.k\.|british|english uk|english \(uk\)|english uk)\b": "UK",
    r"\b(?:australian|australia)\b": "Australian",
    r"\b(?:indian|india)\b": "Indian",
    r"\b(?:canadian|canada)\b": "Canadian",
    r"\b(?:south african|south africa)\b": "South African",
}

_ASSESSMENT_TYPE_MAP = {
    r"\b(personality|behaviour|behavior|behavioural)\b": "Personality",
    r"\b(cognitive|ability|aptitude|numerical|verbal|logical)\b": "Cognitive",
    r"\b(technical|skills|knowledge|competency|competencies)\b": "Knowledge",
    r"\b(simulation|situational judgment|situational|SJ|role play)\b": "Simulation",
    r"\b(biodata|situational judgment|SJ)\b": "Biodata",
    r"\b(development|360|feedback)\b": "Development",
}

_REMOTE_MAP = {
    r"\b(remote|virtual|work from home|distributed)\b": "Remote",
    r"\b(onsite|on[- ]site|office|in[- ]office)\b": "On-site",
    r"\b(hybrid)\b": "Hybrid",
}

_SKILL_PATTERNS = [
    r"\b(java|spring|sql|aws|docker|linux|networking|rust|python|c\+\+|c#|javascript|typescript|excel|word)\b",
    r"\b(sales|customer service|customer success|stakeholder|leadership|management|finance|accounting|statistics|healthcare|hipaa|safety|dependability)\b",
    r"\b(communication|analytical|problem solving|problem-solving|data analysis)\b",
]

_INDUSTRY_PATTERNS = [
    r"\b(retail|finance|healthcare|technology|it|manufacturing|contact centre|customer service)\b",
]

_COMPARISON_KEYWORDS = [r"\bcompare\b", r"\bdifference\b", r"\bversus\b", r"\b vs\b", r"\bbetween\b"]

_DURATION_REGEX = re.compile(r"(\d{1,3})\s*(?:minutes|minute|mins|min)\b")

_TARGET_PAIR_REGEX = re.compile(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\?|\.|$)", re.IGNORECASE)

_ROLE_SCRAPE_PATTERNS = [
    re.compile(r"\b(?:hiring|screening|recruiting|seeking|looking for|searching for|want|need|find)\s+(?:for\s+)?(?:an?\s+)?([^.;,:]+)", re.IGNORECASE),
    re.compile(r"\bfor\s+(?:an?\s+)?([^.;,:]+)", re.IGNORECASE),
]


def _first_match(text: str, pattern_map: dict[str, str]) -> Optional[str]:
    for pattern, label in pattern_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def _collect_matches(text: str, patterns: List[str]) -> List[str]:
    values: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(0).strip()
            if value not in values:
                values.append(value)
    return values


def _safe_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_role(text: str) -> Optional[str]:
    role_hint = re.match(r"^\s*([A-Za-z][A-Za-z0-9/&+\- ]{2,}?)\s*(?:—|-|:)", text)
    if role_hint:
        candidate = role_hint.group(1).strip()
        cleaned = re.sub(r"^\s*(?:we|i|need|looking|seeking|hiring|for)\s+", "", candidate, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
        if re.search(r"\b(engineer|developer|architect|analyst|manager|lead|director|executive|specialist|scientist|designer|programmer|consultant)\b", cleaned, re.IGNORECASE):
            return cleaned

    for pattern in _ROLE_SCRAPE_PATTERNS:
        match = pattern.search(text)
        if match:
            role = match.group(1).strip()
            role = re.sub(r"^\d+(?:\s*-\s*\d+)?\s*", "", role)
            role = re.sub(r"\b(?:a|an|the|our|this|these|those|new)\b", "", role, flags=re.IGNORECASE)
            role = re.sub(r"\b(?:solution|solutions|battery|batteries|assessment|assessments|test|tests|role|roles|candidate|candidates|people|staff|pool)\b", "", role, flags=re.IGNORECASE)
            role = re.sub(r"\s+", " ", role).strip(" ,.;:-")
            role = re.sub(r"^(?:for|to|of)\s+", "", role, flags=re.IGNORECASE)
            if role.lower() in {
                "selection",
                "development",
                "training",
                "recruitment",
                "screening",
                "hiring",
            }:
                return None
            if len(role) > 3:
                return role
    return None


def _extract_comparison_targets(text: str) -> List[str]:
    patterns = [
        re.compile(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\?|\.|$)", re.IGNORECASE),
        re.compile(r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)", re.IGNORECASE),
        re.compile(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    return []


def _extract_duration(text: str) -> Optional[int]:
    match = _DURATION_REGEX.search(text)
    if match:
        return int(match.group(1))
    if re.search(r"\b(short|quick|brief|under \d{1,3} minutes)\b", text, re.IGNORECASE):
        return 30
    return None


def _extract_language(text: str) -> Optional[str]:
    for pattern, label in _LANGUAGE_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def _extract_accent(text: str) -> Optional[str]:
    for pattern, label in _ACCENT_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return label
    match = re.search(r"\b([A-Za-z ]+?) accent\b", text, re.IGNORECASE)
    if match:
        accent_text = match.group(1).strip()
        for pattern, label in _ACCENT_MAP.items():
            if re.search(pattern, accent_text, re.IGNORECASE):
                return label
    return None


def _extract_assessment_types(text: str) -> List[str]:
    values: List[str] = []
    for pattern, label in _ASSESSMENT_TYPE_MAP.items():
        if re.search(pattern, text, re.IGNORECASE) and label not in values:
            values.append(label)
    return values


def _extract_skills(text: str) -> List[str]:
    skills: List[str] = []
    for pattern in _SKILL_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            normalized = match.group(0).strip().lower()
            if normalized not in skills:
                skills.append(normalized)
    return skills


def _extract_industry(text: str) -> Optional[str]:
    for pattern in _INDUSTRY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip().title()
    return None


def _extract_purpose(text: str) -> Optional[str]:
    for pattern, value in _PURPOSE_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def _extract_assessment_scope(text: str) -> Optional[str]:
    lower = text.lower()
    if re.search(_ASSESSMENT_SCOPE_PATTERNS["technical_only"][0], lower):
        return "technical_only"
    if any(re.search(pattern, lower) for pattern in _ASSESSMENT_SCOPE_PATTERNS["technical_and_behavioral"]):
        return "technical_and_behavioral"
    return None


def _extract_role_focus(text: str) -> Optional[str]:
    lower = text.lower()
    if any(re.search(pattern, lower) for pattern in _ROLE_FOCUS_PATTERNS["full-stack"]):
        return "full-stack"

    frontend_hits = any(re.search(pattern, lower) for pattern in _ROLE_FOCUS_PATTERNS["frontend"])
    backend_hits = any(re.search(pattern, lower) for pattern in _ROLE_FOCUS_PATTERNS["backend"])
    if frontend_hits and backend_hits:
        return "full-stack"
    if frontend_hits:
        return "frontend"
    if backend_hits:
        return "backend"
    return None


def _extract_remote_on_site(text: str) -> Optional[str]:
    for pattern, value in _REMOTE_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def _extract_candidate_volume(text: str) -> Optional[str]:
    match = re.search(
        r"\b(\d{1,4}(?:,\d{3})?|hundreds|thousands|high-volume|large volume|batch|team|multiple candidates|volume)\b(?:\s*(?:candidate|candidates|roles|positions|people|agents|staff))?\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(one candidate|two candidates|three candidates|batch|team|multiple candidates|volume)\b", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_item_list(text: str) -> List[str]:
    pieces = re.split(r"\band\b|,|;|/", text, flags=re.IGNORECASE)
    return [piece.strip(" .") for piece in pieces if piece.strip()]


def _extract_explicit_final_list(text: str) -> List[str]:
    patterns = [
        r"\bfinal (?:list|shortlist|selection)\s*[:\-]?\s*(.+)$",
        r"\bfinal list is\s*(.+)$",
        r"\bfinal shortlist is\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [item for item in _parse_item_list(match.group(1)) if item]
    return []


def _extract_explicit_remove(text: str) -> List[str]:
    match = re.search(r"\b(?:drop|remove|exclude|without|leave out|don't include|dont include)\s+([^.;]+)", text, re.IGNORECASE)
    if match:
        items = [item for item in _parse_item_list(match.group(1)) if item]
        normalized = []
        for item in items:
            trimmed = re.sub(r"^the\s+", "", item, flags=re.IGNORECASE).strip()
            if trimmed:
                normalized.append(trimmed)
        return normalized
    return []


def build_state(messages: List[Message]) -> ConversationState:
    user_text = " ".join(m.content for m in messages if m.role == "user")
    return ConversationState(
        role=_extract_role(user_text),
        industry=_extract_industry(user_text),
        experience=None,
        job_level=_first_match(user_text, _JOB_LEVEL_MAP),
        assessment_types=_extract_assessment_types(user_text),
        technical_skills=_extract_skills(user_text),
        personality_requirement=None,
        cognitive_requirement=None,
        language=_extract_language(user_text),
        accent=_extract_accent(user_text),
        duration_minutes=_extract_duration(user_text),
        remote_on_site=_extract_remote_on_site(user_text),
        candidate_volume=_extract_candidate_volume(user_text),
        purpose=_extract_purpose(user_text),
        assessment_scope=_extract_assessment_scope(user_text),
        role_focus=_extract_role_focus(user_text),
        comparison_request=any(re.search(pattern, user_text, re.IGNORECASE) for pattern in _COMPARISON_KEYWORDS),
        comparison_targets=_extract_comparison_targets(user_text),
        explicit_remove=_extract_explicit_remove(user_text),
        explicit_final_list=_extract_explicit_final_list(user_text),
        clarification_needed=False,
        clarification_prompt=None,
    )


def build_search_query(state: ConversationState, messages: List[Message]) -> str:
    pieces: List[str] = []
    if state.role:
        pieces.append(state.role)
    if state.job_level:
        pieces.append(state.job_level)
    if state.industry:
        pieces.append(state.industry)
    if state.assessment_types:
        pieces.extend(state.assessment_types)
    if state.technical_skills:
        pieces.extend(state.technical_skills)
    if state.language:
        pieces.append(state.language)
    if state.accent:
        pieces.append(state.accent)
    if state.remote_on_site:
        pieces.append(state.remote_on_site)
    if state.candidate_volume:
        pieces.append(state.candidate_volume)
    if not pieces and state.purpose:
        # Purpose terms like "selection" are useful when no role/industry or skill
        # constraints are present, but they should not overpower a specific
        # role/industry query such as "hiring for data science".
        pieces.append(state.purpose)
    if not pieces:
        pieces.append(" ".join(m.content for m in messages if m.role == "user"))
    return " ".join(pieces)
