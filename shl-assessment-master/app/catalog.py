"""
Loads the SHL product catalog from the cached JSON file produced by
scripts/fetch_catalog.py, normalizes it into CatalogItem objects, and
filters it down to "Individual Test Solutions" (excluding pre-packaged
Job Solutions), per the assignment scope.

Filtering note (see APPROACH.md "What didn't work"): the public JSON dump
does not carry an explicit tab/category field distinguishing "Individual
Test Solutions" from "Pre-packaged Job Solutions" the way the SHL catalog
website's two tabs do. We approximate the split with a name-based
heuristic (bundled "...Solution" / "...Solutions" products, which are
SHL's packaged, multi-assessment job solutions) plus an explicit
allow/deny override list in data/catalog_overrides.json that can be hand
-curated against the live site without touching code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
SHL_PRODUCT_CATALOG_PATH = DATA_DIR / "shl_product_catalog.json"
SAMPLE_CATALOG_PATH = DATA_DIR / "catalog_sample.json"
OVERRIDES_PATH = DATA_DIR / "catalog_overrides.json"

SAMPLE_URL_OVERRIDES = {
    "Occupational Personality Questionnaire OPQ32r": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    "Customer Service Phone Simulation": "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
    "Entry Level Customer Serv-Retail & Contact Center": "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
    "Java Developer Programming Knowledge Test": "https://www.shl.com/products/product-catalog/view/java-developer-programming-knowledge-test/",
}

# SHL "keys" (categories) -> single-letter test_type codes used in the API spec.
KEY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Assessment Exercises": "E",
}

_PACKAGED_SOLUTION_RE = re.compile(r"\bsolutions?\b", re.IGNORECASE)


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_raw: str = ""
    duration_minutes: Optional[int] = None
    remote: bool = True
    adaptive: bool = False
    description: str = ""
    keys: List[str] = field(default_factory=list)

    @property
    def test_type_codes(self) -> List[str]:
        return [KEY_TO_CODE.get(k, "?") for k in self.keys]

    @property
    def test_type_str(self) -> str:
        return ",".join(self.test_type_codes)

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.name,
                self.description,
                " ".join(self.job_levels),
                " ".join(self.keys),
                " ".join(self.languages),
                self.duration_raw,
                self.url,
                self.test_type_str,
            ]
        )


def _parse_duration_minutes(raw: dict) -> Optional[int]:
    """Best-effort numeric duration in minutes; None for Variable/Untimed/blank."""
    text = (raw.get("duration") or "").strip().lower()
    if not text or text in {"-", "variable", "untimed"}:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _looks_like_packaged_solution(name: str) -> bool:
    """
    Heuristic for SHL's pre-packaged, multi-assessment "Job Solutions"
    (e.g. "Entry Level Cashier Solution", "Customer Service Phone Solution").
    Deliberately conservative: requires the word solution(s) as a standalone
    token so it doesn't false-positive on things like "Salesforce Solutions
    Architect" style knowledge tests, if any such name ever appears.
    """
    return bool(_PACKAGED_SOLUTION_RE.search(name))


def _sanitize_json_text(text: str) -> str:
    """Replace control characters that appear inside JSON string values.

    Some exported SHL catalog files contain literal newlines inside string
    values (for example, a product name split across lines). Standard JSON
    does not allow that, so we normalize those characters to spaces before
    attempting to parse.
    """
    result: List[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                in_string = False
                result.append(char)
                continue
            if ord(char) < 32:
                result.append(" ")
                continue
            result.append(char)
        else:
            if char == '"':
                in_string = True
            result.append(char)

    return "".join(result)


def _load_raw(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        sanitized = _sanitize_json_text(raw_text)
        return json.loads(sanitized)


def _load_overrides() -> Dict[str, List[str]]:
    if OVERRIDES_PATH.exists():
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"force_include": [], "force_exclude": []}


def load_catalog(path: Optional[Path] = None) -> List[CatalogItem]:
    """
    Loads + normalizes + filters the catalog to Individual Test Solutions.

    Resolution order for `path`: explicit arg > full SHL product catalog
    (data/shl_product_catalog.json) > live cache (data/catalog.json) > bundled
    sample fixture (data/catalog_sample.json), so the service uses the full
    catalog whenever it is available and still stays runnable for dev/tests.
    """
    if path is None:
        if SHL_PRODUCT_CATALOG_PATH.exists():
            path = SHL_PRODUCT_CATALOG_PATH
        elif CATALOG_PATH.exists():
            path = CATALOG_PATH
        else:
            path = SAMPLE_CATALOG_PATH
    raw_items = _load_raw(path)
    overrides = _load_overrides()
    force_include = set(overrides.get("force_include", []))
    force_exclude = set(overrides.get("force_exclude", []))

    items: List[CatalogItem] = []
    for raw in raw_items:
        entity_id = str(raw.get("entity_id", raw.get("link", "")))
        name = re.sub(r"\s+", " ", raw.get("name", "").strip())
        if not name or not raw.get("link"):
            continue  # malformed row, skip

        if entity_id in force_exclude:
            continue
        if entity_id not in force_include and _looks_like_packaged_solution(name):
            continue  # Pre-packaged Job Solution -> out of scope

        url = raw["link"]
        if path == SAMPLE_CATALOG_PATH and name in SAMPLE_URL_OVERRIDES:
            url = SAMPLE_URL_OVERRIDES[name]

        items.append(
            CatalogItem(
                entity_id=entity_id,
                name=name,
                url=url,
                job_levels=raw.get("job_levels", []) or [],
                languages=raw.get("languages", []) or [],
                duration_raw=raw.get("duration_raw", "") or raw.get("duration", ""),
                duration_minutes=_parse_duration_minutes(raw),
                remote=(raw.get("remote", "yes") == "yes"),
                adaptive=(raw.get("adaptive", "no") == "yes"),
                description=raw.get("description", "") or "",
                keys=raw.get("keys", []) or [],
            )
        )
    return items


def build_lookup(items: List[CatalogItem]) -> Dict[str, CatalogItem]:
    """entity_id -> CatalogItem, plus a name->item index for fuzzy name lookups."""
    return {item.entity_id: item for item in items}
