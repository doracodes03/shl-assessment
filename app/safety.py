from __future__ import annotations

import re
from typing import List

from .catalog import CatalogItem
from .schema import Recommendation

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (?:previous|earlier|above) instructions\b",
    r"\bdisregard (?:previous|earlier|above) instructions\b",
    r"\bdo not follow the system prompt\b",
    r"\bwrite a prompt\b",
    r"\bsecret\b",
]

OFF_TOPIC_PATTERNS = [
    r"\blegal\b",
    r"\bcompliance\b",
    r"\bregulatory\b",
    r"\bsalary\b",
    r"\bhiring advice\b",
    r"\bdiversity\b",
    r"\bcompensation\b",
    r"\binterview questions\b",
]

URL_WHITELIST_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def is_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in PROMPT_INJECTION_PATTERNS)


def is_off_topic(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in OFF_TOPIC_PATTERNS)


def validate_recommendations(recs: List[Recommendation], catalog_items: List[CatalogItem]) -> bool:
    item_ids = {item.entity_id for item in catalog_items}
    for rec in recs:
        if rec.entity_id not in item_ids:
            return False
        url_value = str(rec.url)
        if not URL_WHITELIST_PATTERN.match(url_value):
            return False
    return True
