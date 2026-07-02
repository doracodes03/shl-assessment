from __future__ import annotations

from typing import List

from .catalog import CatalogItem

COMPARISON_COLUMNS = [
    "name",
    "purpose",
    "duration",
    "job_levels",
    "languages",
    "test_type",
    "adaptive",
    "remote",
]


def comparison_table(items: List[CatalogItem]) -> str:
    rows: List[str] = []
    header = "| Assessment | Purpose | Duration | Job Levels | Languages | Category | Adaptive | Remote |"
    separator = "|---|---|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(separator)
    for item in items:
        purpose = _infer_purpose(item)
        duration = item.duration_raw or "TBD"
        job_levels = ", ".join(item.job_levels) if item.job_levels else "Any"
        languages = ", ".join(item.languages[:3]) if item.languages else "Any"
        category = item.test_type_str
        adaptive = "Yes" if item.adaptive else "No"
        remote = "Yes" if item.remote else "No"
        rows.append(f"| {item.name} | {purpose} | {duration} | {job_levels} | {languages} | {category} | {adaptive} | {remote} |")
    return "\n".join(rows)


def _infer_purpose(item: CatalogItem) -> str:
    if any(k.lower() in {"development", "360"} for k in item.keys):
        return "Development"
    return "Selection"
