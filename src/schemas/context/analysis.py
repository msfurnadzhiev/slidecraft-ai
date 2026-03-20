"""Schemas for content analysis results produced by ContentAnalyzer."""

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from src.schemas.base import BaseSchema
from src.schemas.context.context import ContextRetrievalOptions


class ContentCategory(str, Enum):
    """High-level content role a passage plays in the source document."""

    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    DATA = "data"
    ANALYSIS = "analysis"
    KEY_FINDING = "key_finding"
    CONCLUSION = "conclusion"


class Theme(BaseSchema):
    """A thematic cluster discovered across passages."""

    theme_id: int
    label: str
    keywords: List[str]
    passage_indices: List[int]


class PassageAnalysis(BaseSchema):
    """Per-passage analysis combining retrieval quality, topic centrality,
    and content categorisation."""

    passage_index: int
    page_number: int
    summary: Optional[str] = None
    retrieval_score: float
    topic_relevance: float
    category_scores: Dict[str, float]
    primary_category: ContentCategory
    theme_ids: List[int]
    word_count: int
    is_key_passage: bool


class ContentRelationship(BaseSchema):
    """A discovered relationship between two passages."""

    source_index: int
    target_index: int
    similarity: float
    relationship_type: str


class ContextAnalysis(BaseSchema):
    """Analysis of the context for a document."""
    document_id: UUID
    options: ContextRetrievalOptions | None = None
    themes: List[Theme]
    passage_analyses: List[PassageAnalysis]
    relationships: List[ContentRelationship]
    key_passage_indices: List[int]
    category_groups: Dict[str, List[int]]
    total_passages: int


