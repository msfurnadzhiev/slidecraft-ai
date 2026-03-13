""" This module contains the ContentAnalyzer class which performs semantic and
structural analysis over a RetrievedContext.

The analysis pipeline extracts high-level document themes, evaluates passage 
importance, detects relationships between passages, and groups content by 
semantic category.
"""

from collections import defaultdict
from typing import Dict, List, Set

from src.schemas.analysis import ContextAnalysis, PassageAnalysis
from src.schemas.context import RetrievedContext

from src.services.context_analyzer.constants import (
    DEFAULT_MAX_THEMES,
    DEFAULT_RELATIONSHIP_THRESHOLD,
    DEFAULT_KEY_PASSAGE_FRACTION,
    DEFAULT_MAX_RELATIONSHIPS,
    KEY_PASSAGE_RETRIEVAL_WEIGHT, 
    KEY_PASSAGE_TOPIC_RELEVANCE_WEIGHT
)
from src.services.context_analyzer.corpus import build_corpus, fit_tfidf
from src.services.context_analyzer.passage_scoring import analyze_passages
from src.services.context_analyzer.relationships import discover_relationships
from src.services.context_analyzer.themes import discover_themes


# Type aliases
CategoryGroups = Dict[str, List[int]]

class ContentAnalyzer:
    """
    Performs semantic analysis of a RetrievedContext.

    The analyzer processes retrieved document passages and produces a
    structured ContextAnalysis object describing:

    - Major document themes discovered via TF-IDF + clustering
    - Passage-level relevance and metadata
    - Relationships between passages (semantic or structural)
    - Key passages most representative of the document
    - Grouping of passages by semantic content category
    """

    def __init__(
        self,
        *,
        max_themes: int = DEFAULT_MAX_THEMES,
        relationship_threshold: float = DEFAULT_RELATIONSHIP_THRESHOLD,
        key_passage_fraction: float = DEFAULT_KEY_PASSAGE_FRACTION,
        max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
    ):
        """Initialize the ContentAnalyzer with configuration parameters."""
        self._max_themes = max_themes
        self._relationship_threshold = relationship_threshold
        self._key_passage_fraction = key_passage_fraction
        self._max_relationships = max_relationships


    def analyze(self, context: RetrievedContext) -> ContextAnalysis:
        """Run the full content analysis pipeline on a RetrievedContext.
        
        The method orchestrates several analysis stages to transform raw
        retrieved chunks into structured semantic insights.
        """
        chunks = context.chunks

        if not chunks:
            return self._empty_result(context)

        corpus, corpus_indices = build_corpus(chunks)
        tfidf_matrix, vectorizer = fit_tfidf(corpus)

        passage_to_row = {
            p_idx: row for row, p_idx in enumerate(corpus_indices)
        }

        themes = discover_themes(tfidf_matrix, vectorizer, corpus_indices, self._max_themes)

        passage_analyses = analyze_passages(chunks, themes, tfidf_matrix, passage_to_row)

        relationships = discover_relationships(
            chunks,
            tfidf_matrix,
            corpus_indices,
            self._relationship_threshold,
            self._max_relationships,
        )

        key_indices = self._select_key_passages(passage_analyses)

        category_groups = self._build_category_groups(passage_analyses)

        result = self._build_result(
            context,
            themes,
            passage_analyses,
            relationships,
            key_indices,
            category_groups,
        )

        return result

    def _select_key_passages(
        self,
        analyses: List[PassageAnalysis],
    ) -> Set[int]:
        """Find and mark the most important passages in the document.

        A combined importance score is computed for each passage using a
        weighted mixture of retrieval score and topic relevance.

        Args:
            analyses: The list of PassageAnalysis objects to score.

        Returns:
            A set of passage indices that are considered key passages.
        """
        if not analyses:
            return set()

        # Compute importance score for each passage
        scored = [
            (
                pa.passage_index,
                pa.retrieval_score * KEY_PASSAGE_RETRIEVAL_WEIGHT + \
                pa.topic_relevance * KEY_PASSAGE_TOPIC_RELEVANCE_WEIGHT,
            )
            for pa in analyses
        ]

        # Sort passages by importance score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select the top key passages
        count = max(1, int(len(scored) * self._key_passage_fraction))
        key_indices = {idx for idx, _ in scored[:count]}
        
        # Mark key passages
        for pa in analyses:
            pa.is_key_passage = pa.passage_index in key_indices
        
        return key_indices


    @staticmethod
    def _build_category_groups(
        analyses: List[PassageAnalysis],
    ) -> CategoryGroups:
        """Group passages by their primary semantic category.

        Each PassageAnalysis contains a primary_category attribute describing
        the type of content contained in the passage (e.g. introduction,
        background, data, analysis, key_finding, conclusion).

        Args:
            analyses: The list of PassageAnalysis objects to group.

        Returns:
            A dictionary mapping primary category names to lists of passage indices.
        """
        groups: CategoryGroups = defaultdict(list)

        for pa in analyses:
            groups[pa.primary_category.value].append(pa.passage_index)

        return dict(groups)


    def _build_result(
        self,
        context: RetrievedContext,
        themes,
        passage_analyses,
        relationships,
        key_indices,
        category_groups,
    ) -> ContextAnalysis:
        """Assemble the final ContextAnalysis object."""
        return ContextAnalysis(
            document_id=context.document_id,
            options=context.options,
            themes=themes,
            passage_analyses=passage_analyses,
            relationships=relationships,
            key_passage_indices=sorted(key_indices),
            category_groups=category_groups,
            total_passages=len(context.chunks),
        )


    @staticmethod
    def _empty_result(context: RetrievedContext) -> ContextAnalysis:
        """Return empty analysis when there are no chunks."""
        return ContextAnalysis(
            document_id=context.document_id,
            options=context.options,
            themes=[],
            passage_analyses=[],
            relationships=[],
            key_passage_indices=[],
            category_groups={},
            total_passages=0,
        )