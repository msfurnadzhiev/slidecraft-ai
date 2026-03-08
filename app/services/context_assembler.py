"""Deterministic context assembly from search results."""

from collections import defaultdict

from sqlalchemy.orm import Session

from app.schemas.context import Passage, RetrievalContext
from app.schemas.search import SearchResponse, SearchResultItem

_MIN_OVERLAP = 20
_OVERLAP_WINDOW = 200


class ContextAssembler:
    """Transform raw SearchResponse into document-ordered RetrievalContext."""

    def __init__(self, db: Session):
        pass

    def assemble(self, search_response: SearchResponse) -> RetrievalContext:
        """Deduplicate, reorder, merge chunks and attach page images."""
        if search_response.results:
            grouped = self._group_by_page(search_response.results)
            passages = self._merge_passages(grouped)
        else:
            passages = []

        return RetrievalContext(
            document_id=search_response.document_id,
            query=search_response.query,
            passages=passages,
        )

    @staticmethod
    def _group_by_page(
        items: list[SearchResultItem],
    ) -> dict[int, list[SearchResultItem]]:
        """Group results by page_number; sort each group by chunk_index."""
        by_page: dict[int, list[SearchResultItem]] = defaultdict(list)
        for item in items:
            by_page[item.page_number].append(item)
        for page_items in by_page.values():
            page_items.sort(key=lambda it: it.chunk_index)
        return by_page

    def _merge_passages(
        self, grouped: dict[int, list[SearchResultItem]],
    ) -> list[Passage]:
        """Merge overlapping chunks per page into passages (document order)."""
        passages: list[Passage] = []
        for page_number in sorted(grouped.keys()):
            passages.extend(self._merge_page_chunks(page_number, grouped[page_number]))
        return passages

    @staticmethod
    def _find_overlap(a: str, b: str) -> int:
        """Return length of the longest suffix of *a* that matches a prefix of *b*.

        Only the last/first ``_OVERLAP_WINDOW`` chars are examined, and
        overlaps shorter than ``_MIN_OVERLAP`` are ignored (returns 0).
        """
        a_tail = a[-_OVERLAP_WINDOW:]
        b_head = b[:_OVERLAP_WINDOW]
        for size in range(min(len(a_tail), len(b_head)), _MIN_OVERLAP - 1, -1):
            if a_tail.endswith(b_head[:size]):
                return size
        return 0

    @staticmethod
    def _try_join(current: str, new_text: str, adjacent: bool) -> str | None:
        """Try to merge two chunk texts by overlap or adjacency.

        Returns the merged string, or ``None`` if the chunks are disjoint.
        """
        overlap = ContextAssembler._find_overlap(current, new_text)
        if overlap >= _MIN_OVERLAP:
            return current + new_text[overlap:]
        if adjacent:
            sep = "" if current.endswith(" ") or new_text.startswith(" ") else " "
            return current + sep + new_text
        return None

    @staticmethod
    def _merge_page_chunks(
        page_number: int, page_items: list[SearchResultItem],
    ) -> list[Passage]:
        """Merge consecutive or overlapping chunks on one page into passages."""
        if not page_items:
            return []

        first = page_items[0]
        run_text = first.text
        run_ids: list[str] = [first.chunk_id]
        run_score = first.score
        run_end = first.chunk_index
        passages: list[Passage] = []

        for item in page_items[1:]:
            merged = ContextAssembler._try_join(
                run_text, item.text, adjacent=(item.chunk_index == run_end + 1),
            )
            if merged is not None:
                run_text = merged
                run_ids.append(item.chunk_id)
                run_score = max(run_score, item.score)
                run_end = item.chunk_index
            else:
                passages.append(Passage(
                    page_number=page_number,
                    text=run_text,
                    chunk_ids=run_ids,
                    score=run_score,
                ))
                run_text = item.text
                run_ids = [item.chunk_id]
                run_score = item.score
                run_end = item.chunk_index

        passages.append(Passage(
            page_number=page_number,
            text=run_text,
            chunk_ids=run_ids,
            score=run_score,
        ))
        return passages
