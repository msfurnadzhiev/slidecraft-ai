"""Main ContentAnalyzer: orchestrates analysis and produces AnalyzedContent."""

from collections import defaultdict
from typing import Dict, List, Set

from app.schemas.analysis import AnalyzedContent, PassageAnalysis
from app.schemas.context import Passage, RetrievalContext

from .corpus import build_corpus, fit_tfidf
from .passage_scoring import analyze_single_passage, compute_page_range
from .relationships import discover_relationships
from .themes import discover_themes


class ContentAnalyzer:
    """Analyse a RetrievalContext to extract themes, rank passages, categorise
    content by document role, and discover inter-passage relationships.

    The result is an AnalyzedContent object designed for downstream consumption
    by the presentation planner.
    """

    def __init__(
        self,
        *,
        max_themes: int = 5,
        relationship_threshold: float = 0.15,
        key_passage_fraction: float = 0.4,
        max_relationships: int = 50,
    ):
        self._max_themes = max_themes
        self._relationship_threshold = relationship_threshold
        self._key_passage_fraction = key_passage_fraction
        self._max_relationships = max_relationships

    def analyze(self, context: RetrievalContext) -> AnalyzedContent:
        """Perform full content analysis on a RetrievalContext.

        Steps:
            1. Build TF-IDF representation of passage texts.
            2. Cluster passages into thematic groups (KMeans).
            3. Score each passage: retrieval quality, topic centrality,
               content category, and media metadata.
            4. Discover thematic, same-page, and sequential relationships
               between passages.
            5. Select key passages by combined score.
        """
        passages = context.passages
        if not passages:
            return self._empty_result(context)

        corpus, corpus_indices = build_corpus(passages)
        tfidf_matrix, vectorizer = fit_tfidf(corpus)

        passage_to_row = {
            passage_idx: row for row, passage_idx in enumerate(corpus_indices)
        }

        themes = discover_themes(
            tfidf_matrix, vectorizer, corpus_indices, passages,
            self._max_themes,
        )

        page_range = compute_page_range(passages)
        passage_analyses = [
            analyze_single_passage(
                i, p, themes, tfidf_matrix, passage_to_row, page_range,
            )
            for i, p in enumerate(passages)
        ]

        relationships = discover_relationships(
            passages, tfidf_matrix, passage_to_row, corpus_indices,
            self._relationship_threshold,
            self._max_relationships,
        )

        key_indices = self._select_key_passages(passage_analyses)
        for pa in passage_analyses:
            pa.is_key_passage = pa.passage_index in key_indices

        category_groups = self._build_category_groups(passage_analyses)
        total_images = sum(len(p.images) for p in passages)

        return AnalyzedContent(
            document_id=context.document_id,
            query=context.query,
            themes=themes,
            passage_analyses=passage_analyses,
            relationships=relationships,
            key_passage_indices=sorted(key_indices),
            category_groups=category_groups,
            total_passages=len(passages),
            total_images=total_images,
        )

    def _select_key_passages(
        self,
        analyses: List[PassageAnalysis],
    ) -> Set[int]:
        """Top passages by weighted retrieval + topic relevance score."""
        if not analyses:
            return set()

        scored = [
            (
                pa.passage_index,
                pa.retrieval_score * 0.6 + pa.topic_relevance * 0.4,
            )
            for pa in analyses
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        count = max(1, int(len(scored) * self._key_passage_fraction))
        return {idx for idx, _ in scored[:count]}

    @staticmethod
    def _build_category_groups(
        analyses: List[PassageAnalysis],
    ) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for pa in analyses:
            groups[pa.primary_category.value].append(pa.passage_index)
        return dict(groups)

    @staticmethod
    def _empty_result(context: RetrievalContext) -> AnalyzedContent:
        return AnalyzedContent(
            document_id=context.document_id,
            query=context.query,
            themes=[],
            passage_analyses=[],
            relationships=[],
            key_passage_indices=[],
            category_groups={},
            total_passages=0,
            total_images=0,
        )
