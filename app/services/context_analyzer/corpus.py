"""Corpus building and TF-IDF for passage text."""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from app.schemas.context import Passage


def build_corpus(
    passages: List[Passage],
) -> Tuple[List[str], List[int]]:
    """Return (texts, passage_indices) for passages with non-empty text."""
    corpus: List[str] = []
    indices: List[int] = []
    for i, p in enumerate(passages):
        text = p.text.strip()
        if text:
            corpus.append(text)
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
