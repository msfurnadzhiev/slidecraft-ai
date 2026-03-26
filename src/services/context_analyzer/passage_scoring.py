"""Per-passage scoring and categorization of retrieved content."""

from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas.analysis import ContentCategory, PassageAnalysis, Theme
from src.schemas.chunk import ChunkSearchResult

from src.services.context_analyzer.constants import (
    CATEGORY_KEYWORDS,
    EARLY_CATEGORIES,
    LATE_CATEGORIES,
    NUMERIC_BONUS_CATEGORIES,
    NUMBER_PATTERN,
    RETRIEVAL_BONUS_CATEGORIES,
    W_KEYWORD,
    W_NUMERIC,
    W_POSITION,
    W_RETRIEVAL,
)


def passage_topic_relevance(
    index: int,
    tfidf_matrix,
    passage_to_row: Dict[int, int],
) -> float:
    """Mean cosine similarity to all other passages in the TF-IDF space.

    High values indicate the passage is thematically central;
    low values indicate an outlier or niche topic.
    """
    if tfidf_matrix is None or index not in passage_to_row:
        return 0.0

    row = passage_to_row[index]
    sims = cosine_similarity(tfidf_matrix[row], tfidf_matrix).flatten()
    sims[row] = 0.0
    n = len(sims) - 1
    if n <= 0:
        return 0.0
    return float(np.clip(sims.sum() / n, 0.0, 1.0))


def relative_position(
    page_number: int,
    page_range: Tuple[int, int],
) -> float:
    """Page position normalised to 0..1 within the retrieved range."""
    min_page, max_page = page_range
    span = max_page - min_page
    if span == 0:
        return 0.5
    return (page_number - min_page) / span


def compute_page_range(chunks: List[ChunkSearchResult]) -> Tuple[int, int]:
    """Compute the range of page numbers across the retrieved chunks."""
    if not chunks:
        return (0, 0)
    pages = [c.page_number for c in chunks]
    return min(pages), max(pages)


def score_categories(
    words: Set[str],
    relative_position: float,
    retrieval_score: float,
    raw_text: str,
) -> Dict[ContentCategory, float]:
    """Score a passage against all content categories using multiple signals.

    Signals include:
    - Keyword overlap with category-specific vocabularies
    - Relative document position (early/late bias)
    - Numeric density (bonus for DATA passages)
    - Retrieval score bonus (for KEY_FINDING passages)

    Scores are normalized to sum to 1.0 across categories.

    Args:
        words: A set of unique words from the passage text.
        relative_position: The normalized position of the passage (0.0-1.0).
        retrieval_score: The semantic-search score of the chunk.
        raw_text: The raw text content of the passage.

    Returns:
        A dictionary mapping ContentCategory enum values to normalized scores.
    """
    scores: Dict[ContentCategory, float] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        kw_score = min(1.0, len(words & keywords) / 3.0)

        pos_score = 0.0
        if category in EARLY_CATEGORIES:
            pos_score = max(0.0, 1.0 - relative_position * 2.5)
        elif category in LATE_CATEGORIES:
            pos_score = max(0.0, relative_position * 2.5 - 1.5)

        numeric_score = 0.0
        if category in NUMERIC_BONUS_CATEGORIES:
            num_count = len(NUMBER_PATTERN.findall(raw_text))
            word_count = max(1, len(raw_text.split()))
            numeric_score = min(1.0, num_count / word_count * 15)

        ret_bonus = 0.0
        if category in RETRIEVAL_BONUS_CATEGORIES:
            ret_bonus = retrieval_score

        combined = (
            kw_score * W_KEYWORD
            + pos_score * W_POSITION
            + numeric_score * W_NUMERIC
            + ret_bonus * W_RETRIEVAL
        )
        scores[category] = max(combined, 0.01)

    total = sum(scores.values())
    return {cat: score / total for cat, score in scores.items()}


def analyze_single_passage(
    index: int,
    chunk: ChunkSearchResult,
    themes: List[Theme],
    tfidf_matrix,
    passage_to_row: Dict[int, int],
    page_range: Tuple[int, int],
) -> PassageAnalysis:
    """Compute a comprehensive analysis for a single chunk.

    Computes retrieval score, topic relevance, relative position, category
    scores, primary category, theme membership, and word count.

    Args:
        index: The index of the chunk in the retrieved list.
        chunk: The ChunkSearchResult to analyze.
        themes: The list of themes discovered in the document.
        tfidf_matrix: The TF-IDF matrix of the document chunks.
        passage_to_row: Maps chunk indices to their TF-IDF row indices.
        page_range: The range of page numbers across retrieved chunks.

    Returns:
        A PassageAnalysis object containing the analysis of the chunk.
    """
    retrieval_score = chunk.score
    topic_relevance = passage_topic_relevance(
        index, tfidf_matrix, passage_to_row,
    )

    analysis_text = chunk.text
    if chunk.summary:
        analysis_text = f"{analysis_text} {chunk.summary}"

    words = set(analysis_text.lower().split())
    word_count = len(chunk.text.split())
    rel_pos = relative_position(chunk.page_number, page_range)

    category_scores = score_categories(
        words, rel_pos, retrieval_score, analysis_text,
    )
    primary = max(category_scores, key=category_scores.get)

    theme_ids = [
        t.theme_id for t in themes if index in t.passage_indices
    ]

    return PassageAnalysis(
        passage_index=index,
        page_number=chunk.page_number,
        summary=chunk.summary,
        retrieval_score=round(retrieval_score, 4),
        topic_relevance=round(topic_relevance, 4),
        category_scores={k.value: round(v, 4) for k, v in category_scores.items()},
        primary_category=primary,
        theme_ids=theme_ids,
        word_count=word_count,
        is_key_passage=False,
    )

def analyze_passages(
    chunks: List[ChunkSearchResult],
    themes: List[Theme],
    tfidf_matrix,
    passage_to_row: Dict[int, int],
) -> List[PassageAnalysis]:
    """Analyze all chunks in the retrieved context."""
    page_range = compute_page_range(chunks)

    passage_analyses = [
        analyze_single_passage(
            i,
            chunk,
            themes,
            tfidf_matrix,
            passage_to_row,
            page_range,
        )
        for i, chunk in enumerate(chunks)
    ]

    return passage_analyses
