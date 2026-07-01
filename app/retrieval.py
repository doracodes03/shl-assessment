"""
Hybrid retrieval for the SHL catalog.

This pipeline blends lexical BM25 search, semantic embeddings with a
free model, and structured catalog filtering. The result is strong recall
for exact SHL terminology plus better coverage of experience / skill
queries that use related language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback for environments without the model stack
    SentenceTransformer = None

from .catalog import CatalogItem

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BM25_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.45


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if scores is None:
        return []
    values = list(scores)
    if not values:
        return []
    max_score = max(values)
    if max_score <= 0:
        return [0.0] * len(values)
    return [float(score) / float(max_score) for score in values]


@dataclass
class SearchFilters:
    job_level: Optional[str] = None
    max_duration_minutes: Optional[int] = None
    language: Optional[str] = None
    test_types: Optional[List[str]] = None


@dataclass
class SearchResult:
    item: CatalogItem
    score: float


class CatalogIndex:
    def __init__(self, items: List[CatalogItem]):
        self.items = items
        self._corpus_tokens = [_tokenize(item.searchable_text()) for item in items]
        self._bm25 = BM25Okapi(self._corpus_tokens) if items else None
        self._embedder = None
        self._item_embeddings = np.zeros((0, 0), dtype=np.float32)
        if items:
            try:
                if SentenceTransformer is not None:
                    self._embedder = SentenceTransformer(EMBEDDING_MODEL)
                    self._item_embeddings = self._compute_embeddings(items)
            except Exception:
                self._embedder = None
                self._item_embeddings = np.zeros((0, 0), dtype=np.float32)

    def _compute_embeddings(self, items: List[CatalogItem]) -> np.ndarray:
        if not self._embedder or not items:
            return np.zeros((0, 0), dtype=np.float32)
        texts = [item.searchable_text() for item in items]
        embeddings = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        if not self._embedder or not query:
            return None
        embedding = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].astype(np.float32)

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        if not self.items:
            return []

        query_tokens = _tokenize(query) if query else []
        lexical_scores = self._bm25.get_scores(query_tokens) if query_tokens and self._bm25 else [0.0] * len(self.items)
        semantic_scores = self._semantic_scores(query)

        lexical_norm = _normalize_scores(lexical_scores)
        semantic_norm = _normalize_scores(semantic_scores) if semantic_scores else [0.0] * len(self.items)

        candidate_indices = self._candidate_indices(lexical_norm, semantic_norm, top_k=top_k)
        results: List[SearchResult] = []
        for idx in candidate_indices:
            item = self.items[idx]
            if filters and not self._passes_filters(item, filters):
                continue
            score = BM25_WEIGHT * lexical_norm[idx] + SEMANTIC_WEIGHT * semantic_norm[idx]
            results.append(SearchResult(item=item, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _semantic_scores(self, query: str) -> List[float]:
        if not query or self._item_embeddings.size == 0:
            return [0.0] * len(self.items)
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return [0.0] * len(self.items)
        similarities = np.dot(self._item_embeddings, query_embedding)
        return similarities.tolist()

    def _candidate_indices(self, lexical_norm: List[float], semantic_norm: List[float], top_k: int) -> List[int]:
        scores = [BM25_WEIGHT * l + SEMANTIC_WEIGHT * s for l, s in zip(lexical_norm, semantic_norm)]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[: max(top_k * 2, 20)]

    @staticmethod
    def _passes_filters(item: CatalogItem, filters: SearchFilters) -> bool:
        if filters.job_level and item.job_levels:
            if filters.job_level not in item.job_levels:
                return False
        if filters.max_duration_minutes is not None and item.duration_minutes is not None:
            if item.duration_minutes > filters.max_duration_minutes:
                return False
        if filters.language and item.languages:
            if not any(filters.language.lower() in lang.lower() for lang in item.languages):
                return False
        if filters.test_types:
            wanted = set(filters.test_types)
            if not wanted.intersection(item.test_type_codes):
                return False
        return True

    def get_by_id(self, entity_id: str) -> Optional[CatalogItem]:
        for item in self.items:
            if item.entity_id == entity_id:
                return item
        return None

    def find_by_name(self, name: str) -> List[CatalogItem]:
        name_lower = name.lower().strip()
        exact = [i for i in self.items if i.name.lower() == name_lower]
        if exact:
            return exact
        return [i for i in self.items if name_lower in i.name.lower()]

    def search_by_names(self, names: List[str]) -> List[CatalogItem]:
        found: List[CatalogItem] = []
        for name in names:
            found.extend(self.find_by_name(name))
        return found
