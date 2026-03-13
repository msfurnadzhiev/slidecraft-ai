"""Corpus building and TF-IDF for chunk text."""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas.chunk import ChunkSearchResult


def build_corpus(
    chunks: List[ChunkSearchResult],
) -> Tuple[List[str], List[int]]:
    """Return (texts, chunk_indices) for chunks with non-empty content.

    When a chunk has an LLM-generated summary it is appended to the raw
    text so that TF-IDF captures both the original vocabulary and the
    distilled key concepts produced during ingestion.
    """
    corpus: List[str] = []
    indices: List[int] = []
    for i, c in enumerate(chunks):
        text = c.text.strip()
        summary = (c.summary or "").strip()
        combined = f"{text} {summary}".strip() if summary else text
        if combined:
            corpus.append(combined)
            indices.append(i)
    return corpus, indices


def fit_tfidf(
    corpus: List[str],
) -> Tuple:
    """Fit TF-IDF on the corpus.

    Returns ``(sparse_matrix, vectorizer)`` or ``(None, None)``
    when the corpus is empty or fitting fails.
    """
    if not corpus:
        return None, None
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        matrix = vectorizer.fit_transform(corpus)
        return matrix, vectorizer
    except ValueError:
        return None, None
