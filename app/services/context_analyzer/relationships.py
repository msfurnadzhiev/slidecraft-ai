"""Discovery of thematic, same-page, and sequential relationships between passages."""

from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.analysis import ContentRelationship
from app.schemas.context import Passage


def discover_relationships(
    passages: List[Passage],
    tfidf_matrix,
    passage_to_row: Dict[int, int],
    corpus_indices: List[int],
    relationship_threshold: float,
    max_relationships: int,
) -> List[ContentRelationship]:
    """Find thematic, same-page, and sequential relationships."""
    relationships: List[ContentRelationship] = []

    if tfidf_matrix is not None and tfidf_matrix.shape[0] > 1:
        sim_matrix = cosine_similarity(tfidf_matrix)
        for i in range(sim_matrix.shape[0]):
            for j in range(i + 1, sim_matrix.shape[1]):
                if sim_matrix[i, j] >= relationship_threshold:
                    relationships.append(
                        ContentRelationship(
                            source_index=corpus_indices[i],
                            target_index=corpus_indices[j],
                            similarity=round(float(sim_matrix[i, j]), 4),
                            relationship_type="thematic",
                        )
                    )

    thematic_pairs = {
        (r.source_index, r.target_index) for r in relationships
    }

    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            if (i, j) in thematic_pairs:
                continue
            pi, pj = passages[i].page_number, passages[j].page_number
            if pi == pj:
                relationships.append(
                    ContentRelationship(
                        source_index=i,
                        target_index=j,
                        similarity=1.0,
                        relationship_type="same_page",
                    )
                )
            elif abs(pi - pj) == 1:
                relationships.append(
                    ContentRelationship(
                        source_index=i,
                        target_index=j,
                        similarity=0.8,
                        relationship_type="sequential",
                    )
                )

    if len(relationships) > max_relationships:
        relationships.sort(key=lambda r: r.similarity, reverse=True)
        relationships = relationships[:max_relationships]

    return relationships
