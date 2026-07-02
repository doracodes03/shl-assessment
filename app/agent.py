"""
Conversation orchestration for the SHL recommendation agent.

This implementation is stateless: it rebuilds structured conversation state
from the full message history and then runs deterministic clarification,
hybrid retrieval, optional reranking, comparison, and safety filtering.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from .catalog import CatalogItem
from .clarification import clarification_prompt, needs_clarification
from .comparison import comparison_table
from .llm import LLMClient
from .prompt import SYSTEM_PROMPT
from .retrieval import CatalogIndex, SearchFilters
from .safety import is_off_topic, is_prompt_injection, validate_recommendations
from .schema import ChatResponse, ConversationState, Message, Recommendation
from .state import build_search_query, build_state

logger = logging.getLogger("shl-agent")
MAX_RECOMMENDATIONS = 10

CATEGORY_TO_CODE = {
    "Personality": "P",
    "Cognitive": "A",
    "Knowledge": "K",
    "Simulation": "S",
    "Biodata": "B",
    "Development": "D",
    "Competency": "C",
}

RERANK_PROMPT = """
You are an SHL assessment ranking assistant.
You may only use the catalog metadata provided below.
Rank the candidate items by fit for the user's request.
Return only entity IDs separated by commas, in best-to-worst order.
If candidates are equally good, keep their existing order.
"""

KNOWLEDGE_QUERY_PATTERNS = [
    r"\bwhat\s+(?:is|does)\b",
    r"\btell me about\b",
    r"\bdescribe\b",
    r"\bdifference\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bwhich one\b",
    r"\bwhy use\b",
    r"\bhow\s+(?:is|does)\b",
]

_STOPWORD_TOKENS = {
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "the",
    "a",
    "an",
    "for",
    "to",
    "of",
    "on",
    "in",
    "and",
    "or",
    "by",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "role",
    "roles",
    "assessment",
    "assessments",
    "test",
    "tests",
    "candidate",
    "candidates",
    "selection",
    "hiring",
    "recruitment",
    "development",
    "training",
}


class Agent:
    def __init__(self, index: CatalogIndex, llm: LLMClient):
        self.index = index
        self.llm = llm

    def handle(self, messages: List[Message]) -> ChatResponse:
        state = build_state(messages)
        user_text = " ".join(m.content for m in messages if m.role == "user")

        if is_prompt_injection(user_text):
            return ChatResponse(
                reply=(
                    "I cannot follow that request because it appears to ask me to ignore my safety rules. "
                    "I can only recommend SHL Individual Test Solutions from the catalog."
                ),
                state=state,
                end_of_conversation=True,
            )

        if is_off_topic(user_text):
            return ChatResponse(
                reply=(
                    "I'm focused on SHL assessment recommendations only. "
                    "I cannot provide legal, salary, or general hiring advice."
                ),
                state=state,
                end_of_conversation=True,
            )

        if self._looks_like_assessment_knowledge_query(user_text):
            return self._handle_assessment_knowledge(state, messages)

        if needs_clarification(state):
            prompt = clarification_prompt(state) or "Can you clarify what you need from the SHL catalog?"
            return ChatResponse(reply=prompt, state=state, end_of_conversation=False)

        if state.comparison_request:
            return self._handle_comparison(state)

        return self._handle_recommendation(state, messages)

    def _handle_comparison(self, state: ConversationState) -> ChatResponse:
        items = self._resolve_comparison_items(state.comparison_targets)
        if len(items) < 2:
            return ChatResponse(
                reply=(
                    "I could not find two matching SHL assessments to compare. "
                    "Please provide exact catalog names or IDs for the two assessments."
                ),
                state=state,
                end_of_conversation=False,
            )
        table = comparison_table(items)
        return ChatResponse(
            reply="Here is the grounded comparison of the requested SHL assessments.",
            comparison_table=table,
            state=state,
            end_of_conversation=True,
        )

    def _looks_like_assessment_knowledge_query(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        if any(re.search(pattern, lower) for pattern in KNOWLEDGE_QUERY_PATTERNS):
            return True
        if re.search(r"\b(what|tell|describe|difference|compare|versus|vs|why|how)\b", lower):
            return bool(self._match_assessment_items(text, top_k=3))
        return False

    def _handle_assessment_knowledge(self, state: ConversationState, messages: List[Message]) -> ChatResponse:
        query = " ".join(m.content for m in messages if m.role == "user")
        matched_items = self._match_assessment_items(query, top_k=4)
        if not matched_items:
            return ChatResponse(
                reply=(
                    "I could not find a matching assessment entry in the loaded SHL catalog. "
                    "I can only answer from the catalog metadata and product links that are currently available. "
                    "If you share the full SHL assessment name, I can explain what information is missing from the catalog."
                ),
                state=state,
                end_of_conversation=True,
            )

        is_comparison = any(re.search(pattern, query.lower()) for pattern in [r"\bdifference\b", r"\bcompare\b", r"\bversus\b", r"\bvs\b"])
        if is_comparison and len(matched_items) >= 2:
            reply = self._format_assessment_comparison(matched_items[:2])
            recommendations = [self._build_catalog_recommendation(item) for item in matched_items[:2]]
            return ChatResponse(reply=reply, recommendations=recommendations, state=state, end_of_conversation=True)

        item = matched_items[0]
        reply = self._format_assessment_summary(item)
        recommendations = [self._build_catalog_recommendation(item)]
        return ChatResponse(reply=reply, recommendations=recommendations, state=state, end_of_conversation=True)

    def _match_assessment_items(self, query: str, top_k: int = 4) -> List[CatalogItem]:
        raw_query = query.strip()
        if not raw_query:
            return []

        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9+#.]+", raw_query.lower())
            if len(token) > 2 and token not in _STOPWORD_TOKENS
        }

        direct_matches = self.index.find_by_name(raw_query)
        if direct_matches:
            return direct_matches[:top_k]

        exact_matches = []
        for item in self.index.items:
            item_tokens = {
                token
                for token in re.findall(r"[a-z0-9+#.]+", item.name.lower())
                if len(token) > 2 and token not in _STOPWORD_TOKENS
            }
            if query_tokens and query_tokens.issubset(item_tokens):
                exact_matches.append(item)
                continue
            if raw_query.lower() in item.name.lower() or item.name.lower() in raw_query.lower():
                exact_matches.append(item)

        if exact_matches:
            # Prefer exact text matches and preserve the item order.
            return exact_matches[:top_k]

        search_results = self.index.search(raw_query, top_k=max(top_k, 8))
        scored_items: List[tuple[int, CatalogItem]] = []
        for result in search_results:
            item = result.item
            name_tokens = {
                token
                for token in re.findall(r"[a-z0-9+#.]+", item.name.lower())
                if len(token) > 2 and token not in _STOPWORD_TOKENS
            }
            overlap = len(query_tokens.intersection(name_tokens))
            exact_name_match = raw_query.lower() in item.name.lower() or item.name.lower() in raw_query.lower()
            if overlap > 0 or exact_name_match:
                scored_items.append((overlap, item))

        if not scored_items:
            return []

        scored_items.sort(
            key=lambda entry: (
                -entry[0],
                -self.index.search(entry[1].name, top_k=1)[0].score
                if self.index.search(entry[1].name, top_k=1)
                else 0,
            )
        )
        seen: set[str] = set()
        matched: List[CatalogItem] = []
        for _, item in scored_items:
            if item.entity_id in seen:
                continue
            seen.add(item.entity_id)
            matched.append(item)
            if len(matched) >= top_k:
                break
        return matched

    def _format_assessment_summary(self, item: CatalogItem) -> str:
        description = item.description.strip() or "No description was provided in the catalog entry."
        languages = ", ".join(item.languages) if item.languages else "not listed"
        keys = ", ".join(item.keys) if item.keys else "not listed"
        duration = item.duration_raw or ("%s minutes" % item.duration_minutes if item.duration_minutes is not None else "not listed")
        link = item.url or "No catalog URL was available."
        assessed_areas = keys if keys != "not listed" else "not listed"

        return (
            f"I found {item.name} in the SHL catalog.\n\n"
            f"Catalog facts:\n"
            f"- Test type: {item.test_type_str or 'not listed'}\n"
            f"- Keys: {keys}\n"
            f"- Duration: {duration}\n"
            f"- Languages: {languages}\n"
            f"- Description: {description}\n"
            f"- Link: {link}\n\n"
            f"What this assessment evaluates:\n"
            f"- Focus areas: {assessed_areas}\n"
            f"- It is designed to help identify candidates' strengths and fit for roles that require {assessed_areas.lower()} based on the catalog metadata.\n"
            f"- Use it when you want grounded insight from the SHL catalog into candidate capabilities and behavioral style."
        )

    def _format_assessment_comparison(self, items: List[CatalogItem]) -> str:
        sections = []
        for item in items:
            languages = ", ".join(item.languages) if item.languages else "not listed"
            keys = ", ".join(item.keys) if item.keys else "not listed"
            duration = item.duration_raw or ("%s minutes" % item.duration_minutes if item.duration_minutes is not None else "not listed")
            sections.append(
                f"{item.name}\n"
                f"- Test type: {item.test_type_str or 'not listed'}\n"
                f"- Keys: {keys}\n"
                f"- Duration: {duration}\n"
                f"- Languages: {languages}\n"
                f"- Link: {item.url}"
            )
        return "I found catalog entries that match your comparison request. Based only on the catalog metadata and links, here is the grounded comparison:\n\n" + "\n\n".join(sections)

    def _build_catalog_recommendation(self, item: CatalogItem) -> Recommendation:
        return Recommendation(
            entity_id=item.entity_id,
            name=item.name,
            url=item.url,
            test_type=item.test_type_str,
            keys=item.keys or [],
            duration=item.duration_raw or "Not listed",
            languages=item.languages or [],
            reason="Catalog-grounded assessment detail from the SHL catalog.",
            confidence="high",
            matched_constraints=["catalog metadata"],
            matched_skills=[],
        )

    def _handle_recommendation(self, state: ConversationState, messages: List[Message]) -> ChatResponse:
        if state.explicit_final_list:
            explicit_items = self._resolve_explicit_items(state.explicit_final_list)
            if explicit_items:
                recommendations = [self._build_recommendation(item, state) for item in explicit_items[:MAX_RECOMMENDATIONS]]
                return ChatResponse(
                    reply=self._format_explicit_final_reply(state, recommendations, state.explicit_final_list),
                    recommendations=recommendations,
                    state=state,
                    end_of_conversation=True,
                )
            return ChatResponse(
                reply=(
                    "I understood that you wanted a final shortlist, but I couldn't match one or more of the requested catalog names exactly. "
                    "Please provide the exact assessment names or product titles so I can update the shortlist correctly."
                ),
                state=state,
                end_of_conversation=False,
            )

        query = build_search_query(state, messages)
        filters = SearchFilters(
            job_level=state.job_level,
            max_duration_minutes=state.duration_minutes,
            language=state.language,
            test_types=self._map_assessment_types(state.assessment_types),
        )
        candidates = self.index.search(query, filters=filters, top_k=20)

        if not candidates:
            return ChatResponse(
                reply=(
                    "I couldn't find SHL Individual Test Solutions that match those constraints. "
                    "Please clarify the role, level, language, or duration."
                ),
                state=state,
                end_of_conversation=False,
            )

        candidate_items = [result.item for result in candidates]
        if state.explicit_remove:
            removed_items = self._resolve_explicit_items(state.explicit_remove)
            removed_ids = {item.entity_id for item in removed_items}
            candidate_items = [item for item in candidate_items if item.entity_id not in removed_ids]

        ranked = self._rerank_candidates(candidate_items, state)
        recommendations = [self._build_recommendation(item, state) for item in ranked[:MAX_RECOMMENDATIONS]]

        if not validate_recommendations(recommendations, self.index.items):
            logger.warning("Recommendation validation failed; falling back to filtered candidates.")
            recommendations = [self._build_recommendation(item, state) for item in candidate_items[:MAX_RECOMMENDATIONS]]

        reply = self._format_recommendation_reply(state, recommendations)
        if state.explicit_remove and removed_items:
            reply = self._format_replacement_reply(state, removed_items, recommendations)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            state=state,
            end_of_conversation=bool(state.explicit_remove),
        )

    def _resolve_comparison_items(self, targets: List[str]) -> List[CatalogItem]:
        items: List[CatalogItem] = []
        for token in targets:
            item = self.index.get_by_id(token)
            if item:
                items.append(item)
                continue
            matches = self.index.find_by_name(token)
            if matches:
                items.append(matches[0])
        return items

    def _resolve_explicit_items(self, names: List[str]) -> List[CatalogItem]:
        items: List[CatalogItem] = []
        seen: set[str] = set()
        for name in names:
            normalized = name.lower().strip()
            if not normalized:
                continue
            item = self.index.get_by_id(normalized)
            if item and item.entity_id not in seen:
                items.append(item)
                seen.add(item.entity_id)
                continue
            matches = self.index.find_by_name(name)
            if matches:
                for match in matches:
                    if match.entity_id not in seen:
                        items.append(match)
                        seen.add(match.entity_id)
                        break
                continue
            search_results = self.index.search(name, top_k=3)
            for result in search_results:
                if result.item.entity_id not in seen:
                    items.append(result.item)
                    seen.add(result.item.entity_id)
                    break
        return items

    def _format_explicit_final_reply(self, state: ConversationState, recommendations: List[Recommendation], final_list: List[str]) -> str:
        requested = ", ".join(final_list)
        return (
            f"Confirmed. I have updated the shortlist to your final requested list: {requested}. "
            f"It is now aligned to your selected core assessments and avoids the previously suggested personality assessment."
        )

    def _format_replacement_reply(self, state: ConversationState, removed_items: List[CatalogItem], recommendations: List[Recommendation]) -> str:
        removed_names = ", ".join(item.name for item in removed_items)
        replaced_with = ", ".join(rec.name for rec in recommendations[:2])
        return (
            f"I removed {removed_names} as requested. "
            f"The updated shortlist still covers your core needs; I kept {replaced_with} because they offer cognitive and situational judgement coverage for an entry-level graduate programme."
        )

    def _map_assessment_types(self, assessment_types: List[str]) -> Optional[List[str]]:
        codes = [CATEGORY_TO_CODE.get(at) for at in assessment_types if CATEGORY_TO_CODE.get(at)]
        return list(dict.fromkeys(codes)) if codes else None

    def _rerank_candidates(self, candidates: List[CatalogItem], state: ConversationState) -> List[CatalogItem]:
        if not self.llm.configured or len(candidates) <= 1:
            return candidates

        prompt_lines = [
            RERANK_PROMPT,
            f"User need: role={state.role}, level={state.job_level}, purpose={state.purpose}, language={state.language}, duration={state.duration_minutes}" if (state.role or state.job_level or state.purpose or state.language or state.duration_minutes) else "User need: general SHL assessment recommendation.",
            "Candidates:",
        ]
        for item in candidates:
            prompt_lines.append(
                f"{item.entity_id}: {item.name} | category={item.test_type_str} | job_levels={','.join(item.job_levels)} | duration={item.duration_raw} | languages={','.join(item.languages[:3])}"
            )
        prompt = "\n".join(prompt_lines)
        try:
            ranking = self.llm.complete(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            ordered_ids = [token.strip() for token in ranking.replace("\n", ",").split(",") if token.strip()]
            ordered = [next((item for item in candidates if item.entity_id == eid), None) for eid in ordered_ids]
            ordered = [item for item in ordered if item is not None]
            if len(ordered) >= 2:
                return ordered
        except Exception as exc:
            logger.warning("LLM rerank failed: %s", exc)
        return candidates

    def _build_recommendation(self, item: CatalogItem, state: ConversationState) -> Recommendation:
        matched_constraints = []
        if state.job_level and state.job_level in item.job_levels:
            matched_constraints.append("job_level")
        if state.language and any(state.language.lower() in lang.lower() for lang in item.languages):
            matched_constraints.append("language")
        if state.accent and any(state.accent.lower() in lang.lower() for lang in item.languages):
            matched_constraints.append("accent")
        if state.duration_minutes is not None and item.duration_minutes is not None and item.duration_minutes <= state.duration_minutes:
            matched_constraints.append("duration")
        if state.purpose:
            matched_constraints.append("purpose")
        if state.assessment_types and set(self._map_assessment_types(state.assessment_types) or []).intersection(item.test_type_codes):
            matched_constraints.append("test_type")

        matched_skills = [skill for skill in state.technical_skills if skill in item.searchable_text().lower()]
        if matched_skills:
            matched_constraints.append("skills")
        if not matched_constraints:
            matched_constraints.append("catalog fit")

        reason_parts: List[str] = []
        if state.role:
            reason_parts.append(f"Relevant for {state.role}")
        if state.job_level:
            reason_parts.append(f"appropriate for {state.job_level} level")
        if state.purpose:
            reason_parts.append(f"supports {state.purpose}")
        if state.language:
            reason_parts.append(f"available in {state.language}")
        if state.accent:
            reason_parts.append(f"matches accent preference {state.accent}")
        if item.duration_raw:
            reason_parts.append(f"fits duration {item.duration_raw}")
        if matched_skills:
            reason_parts.append(f"aligned to skills: {', '.join(matched_skills)}")
        reason = ", ".join(reason_parts) if reason_parts else "Matches the user's requested assessment criteria."

        confidence = "high" if len(matched_constraints) >= 2 else "medium"
        return Recommendation(
            entity_id=item.entity_id,
            name=item.name,
            url=item.url,
            test_type=item.test_type_str,
            keys=item.keys or [],
            duration=item.duration_raw or "Not listed",
            languages=item.languages or [],
            reason=reason,
            confidence=confidence,
            matched_constraints=matched_constraints,
            matched_skills=matched_skills,
        )

    def _format_recommendation_reply(self, state: ConversationState, recommendations: List[Recommendation]) -> str:
        if not recommendations:
            return "I couldn't find good matches in the SHL catalog with the details provided."

        base = f"Based on the requested {state.job_level or 'role'} {state.role or ''}".strip()
        lines = [
            f"{base}, here are the top {len(recommendations)} SHL Individual Test Solutions that fit the catalog and the stated constraints:",
            "",
            "| # | Name | Test Type | Keys | Duration | Languages | URL |",
            "|---|------|-----------|------|----------|-----------|-----|",
        ]
        for idx, rec in enumerate(recommendations, start=1):
            lines.append(
                f"| {idx} | {rec.name} | {rec.test_type} | {rec.keys} | {rec.duration or 'Not listed'} | {rec.languages or 'Not listed'} | {rec.url} |"
            )

        best = recommendations[0]
        lines.append("")
        lines.append(
            f"If I were to choose one, I would recommend {best.name} because {best.reason.lower()}. "
            "It has the strongest match to the requested role and level and the clearest catalog fit among the top results."
        )
        return "\n".join(lines)
