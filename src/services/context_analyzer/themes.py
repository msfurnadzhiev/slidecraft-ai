"""Theme discovery via KMeans clustering on TF-IDF."""

from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas.analysis import Theme

from src.services.context_analyzer.constants import MIN_PASSAGES_FOR_CLUSTERING


def discover_themes(
    tfidf_matrix,
    vectorizer: TfidfVectorizer | None,
    corpus_indices: List[int],
    max_themes: int,
) -> List[Theme]:
    """Cluster passages into thematic groups via KMeans on TF-IDF.
    
    The function clusters TF-IDF passage vectors using KMeans to identify
    coherent thematic groups. Each cluster represents a shared topic
    across a subset of passages.

    Args:
        tfidf_matrix: The TF-IDF matrix of the document passages.
        vectorizer: The TF-IDF vectorizer used to transform text into vectors.
        corpus_indices: The indices of the document passages.
        max_themes: The maximum number of themes to discover.

    Returns:
        A list of Theme objects, each representing a thematic cluster.
    """
    if tfidf_matrix is None or vectorizer is None:
        return []

    # If there are too few passages, use a single theme fallback
    if len(corpus_indices) < MIN_PASSAGES_FOR_CLUSTERING:
        return single_theme_fallback(
            vectorizer, tfidf_matrix, corpus_indices,
        )

    n_clusters = min(max_themes, len(corpus_indices))
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(tfidf_matrix)
    except Exception:
        return single_theme_fallback(
            vectorizer, tfidf_matrix, corpus_indices,
        )

    feature_names = vectorizer.get_feature_names_out()
    themes: List[Theme] = []

    for cluster_id in range(n_clusters):
        member_mask = labels == cluster_id
        if not member_mask.any():
            continue

        center = kmeans.cluster_centers_[cluster_id]
        top_term_indices = center.argsort()[::-1][:7]
        keywords = [feature_names[idx] for idx in top_term_indices]

        member_passage_indices = [
            corpus_indices[j]
            for j, is_member in enumerate(member_mask)
            if is_member
        ]

        themes.append(
            Theme(
                theme_id=cluster_id,
                label=", ".join(keywords[:3]),
                keywords=keywords,
                passage_indices=member_passage_indices,
            )
        )

    return themes


def single_theme_fallback(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    corpus_indices: List[int],
) -> List[Theme]:
    """Produce a single catch-all theme when there are too few passages.
    
    When the document contains too few passages to perform meaningful
    clustering, the function creates a single theme that encompasses all
    passages. This theme is characterized by the average TF-IDF vector of
    all passages.
    """
    feature_names = vectorizer.get_feature_names_out()
    mean_vector = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_vector.argsort()[::-1][:7]
    keywords = [feature_names[idx] for idx in top_indices]
    return [
        Theme(
            theme_id=0,
            label=", ".join(keywords[:3]),
            keywords=keywords,
            passage_indices=list(corpus_indices),
        )
    ]
