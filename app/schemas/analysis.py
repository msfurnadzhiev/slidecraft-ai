"""Schemas for content analysis results produced by ContentAnalyzer."""

from enum import Enum
from typing import Dict, List

from app.schemas.base import BaseSchema
from app.schemas.context import RetrievalContext


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
    content categorisation, and media metadata."""

    passage_index: int
    page_number: int
    retrieval_score: float
    topic_relevance: float
    category_scores: Dict[str, float]
    primary_category: ContentCategory
    theme_ids: List[int]
    has_images: bool
    image_count: int
    word_count: int
    is_key_passage: bool


class ContentRelationship(BaseSchema):
    """A discovered relationship between two passages."""

    source_index: int
    target_index: int
    similarity: float
    relationship_type: str


class AnalyzedContent(BaseSchema):
    """Complete content analysis of a RetrievalContext, ready for
    consumption by the PresentationPlanner."""

    document_id: str
    query: str | None = None
    themes: List[Theme]
    passage_analyses: List[PassageAnalysis]
    relationships: List[ContentRelationship]
    key_passage_indices: List[int]
    category_groups: Dict[str, List[int]]
    total_passages: int
    total_images: int


class AnalysisResponse(BaseSchema):
    """Search endpoint response: assembled context enriched with content analysis."""

    context: RetrievalContext | None = None
    analysis: AnalyzedContent
