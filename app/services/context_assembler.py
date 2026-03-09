"""Deterministic context assembly from search results."""

from collections import defaultdict
from typing import List, Dict

from app.schemas.context import Passage, PassageChunk, PassageImage, RetrievalContext
from app.schemas.search import SearchResponse, SearchResultItem, ImageResultItem

# Minimum overlap in characters to merge chunks
_MIN_OVERLAP = 20
# Maximum window of characters to consider for overlaps
_OVERLAP_WINDOW = 200

# Type aliases for clarity
GroupedChunks = Dict[int, List[SearchResultItem]]


class ContextAssembler:
    """Transform raw SearchResponse into document-ordered RetrievalContext."""

    def assemble(self, search_response: SearchResponse) -> RetrievalContext:
        """
        Deduplicate, reorder, merge chunks, and attach semantic image matches.

        Args:
            search_response: A SearchResponse object containing chunk and image results.

        Returns:
            RetrievalContext containing ordered passages with associated images.
        """
        passages: List[Passage] = []

        if search_response.results:
            grouped: GroupedChunks = self._group_by_page(search_response.results)
            passages = self._merge_passages(grouped)

        self._attach_images(passages, search_response)

        return RetrievalContext(
            document_id=search_response.document_id,
            query=search_response.query,
            passages=passages,
        )

    @staticmethod
    def _attach_images(
        passages: List[Passage],
        search_response: SearchResponse,
    ) -> None:
        """
        Attach semantic image results to passages and update passage scores.

        Args:
            passages: List of Passage objects to attach images to.
            search_response: The SearchResponse containing image results.

        Returns:
            None. Passages are modified in place.
        """
        if not search_response.image_results:
            return

        images_by_page: Dict[int, List[ImageResultItem]] = defaultdict(list)
        for img in search_response.image_results:
            images_by_page[img.page_number].append(img)

        def to_passage_images(items: List[ImageResultItem]) -> List[PassageImage]:
            return [PassageImage(image_id=img.image_id, score=img.score) for img in items]

        assigned_pages = set()
        for passage in passages:
            if passage.page_number in images_by_page:
                passage.images = to_passage_images(images_by_page[passage.page_number])
                assigned_pages.add(passage.page_number)

        for page_number in sorted(images_by_page.keys() - assigned_pages):
            passages.append(
                Passage(
                    page_number=page_number,
                    text="",
                    chunks=[],
                    images=to_passage_images(images_by_page[page_number]),
                )
            )

        passages.sort(key=lambda p: p.page_number)

    @staticmethod
    def _group_by_page(items: List[SearchResultItem]) -> GroupedChunks:
        """
        Group chunk results by page number and sort by chunk_index.

        Args:
            items: List of SearchResultItem objects.

        Returns:
            Dictionary mapping page_number -> sorted list of SearchResultItem.
        """
        by_page: GroupedChunks = defaultdict(list)
        for item in items:
            by_page[item.page_number].append(item)
        for page_items in by_page.values():
            page_items.sort(key=lambda it: it.chunk_index)
        return by_page

    def _merge_passages(self, grouped: GroupedChunks) -> List[Passage]:
        """
        Merge overlapping chunks per page into ordered passages.

        Args:
            grouped: Dictionary mapping page_number -> list of SearchResultItem.

        Returns:
            List of Passage objects with merged text and scores.
        """
        passages: List[Passage] = []
        for page_number in sorted(grouped.keys()):
            passages.extend(self._merge_page_chunks(page_number, grouped[page_number]))
        return passages

    @staticmethod
    def _find_overlap(a: str, b: str) -> int:
        """
        Compute the length of the longest suffix of a matching a prefix of b.

        Args:
            a: First string (existing passage).
            b: Second string (new chunk).

        Returns:
            Length of overlap. Returns 0 if below _MIN_OVERLAP.
        """
        a_tail = a[-_OVERLAP_WINDOW:]
        b_head = b[:_OVERLAP_WINDOW]
        for size in range(min(len(a_tail), len(b_head)), _MIN_OVERLAP - 1, -1):
            if a_tail.endswith(b_head[:size]):
                return size
        return 0

    @staticmethod
    def _try_join(current: str, new_text: str, adjacent: bool) -> str | None:
        """
        Attempt to merge two chunk texts using overlap or adjacency.

        Args:
            current: Existing text of the passage.
            new_text: Text of the new chunk.
            adjacent: Whether chunks are consecutive indices.

        Returns:
            Merged string if overlap or adjacency allows, else None.
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
        page_number: int,
        page_items: List[SearchResultItem],
    ) -> List[Passage]:
        """
        Merge consecutive or overlapping chunks on a single page into passages.

        Args:
            page_number: The page number of the chunks.
            page_items: List of SearchResultItem sorted by chunk_index.

        Returns:
            List of merged Passage objects for the page.
        """
        if not page_items:
            return []

        first = page_items[0]
        run_text = first.text
        run_items: List[SearchResultItem] = [first]
        run_end = first.chunk_index
        passages: List[Passage] = []

        for item in page_items[1:]:
            merged = ContextAssembler._try_join(
                run_text, item.text, adjacent=(item.chunk_index == run_end + 1)
            )
            if merged is not None:
                run_text = merged
                run_items.append(item)
                run_end = item.chunk_index
            else:
                passages.append(
                    Passage(
                        page_number=page_number,
                        text=run_text,
                        chunks=[
                            PassageChunk(
                                chunk_id=c.chunk_id,
                                chunk_index=c.chunk_index,
                                score=c.score,
                            )
                            for c in run_items
                        ],
                    )
                )
                run_text = item.text
                run_items = [item]
                run_end = item.chunk_index

        passages.append(
            Passage(
                page_number=page_number,
                text=run_text,
                chunks=[
                    PassageChunk(
                        chunk_id=c.chunk_id,
                        chunk_index=c.chunk_index,
                        score=c.score,
                    )
                    for c in run_items
                ],
            )
        )
        return passages