"""Assemble retrieval context from raw chunk/image search outputs."""

from collections import defaultdict
from typing import Dict, List

from app.schemas.context import (
    ChunkReference,
    ImageReference,
    Passage,
    RawContext,
    RetrievalContext,
)
from app.schemas.chunk import ChunkSearchResult
from app.schemas.image import ImageSearchResult

# Minimum overlap in characters to merge chunks
_MIN_OVERLAP = 20
# Maximum window of characters to consider for overlaps
_OVERLAP_WINDOW = 200

GroupedChunks = Dict[int, List[ChunkSearchResult]]
GroupedImages = Dict[int, List[ImageSearchResult]]


class ContextAssembler:
    """Transform raw search results into a document-ordered retrieval context."""

    def assemble(self, raw_context: RawContext) -> RetrievalContext:
        """Deduplicate, reorder, merge chunks, and attach semantic image matches."""
        passages: List[Passage] = []

        if raw_context.chunks:
            grouped: GroupedChunks = self._group_by_page(raw_context.chunks)
            passages = self._merge_passages(grouped)

        if raw_context.images:
            self._attach_images(passages, raw_context.images)

        return RetrievalContext(
            document_id=raw_context.document_id,
            query=raw_context.query,
            passages=passages,
        )

    @staticmethod
    def _attach_images(
        passages: List[Passage],
        context_images: List[ImageSearchResult],
    ) -> None:
        """Attach image results to matching passages by page number."""
        images_by_page: GroupedImages = defaultdict(list)
        for image in context_images:
            images_by_page[image.page_number].append(image)

        def to_image_refs(items: List[ImageSearchResult]) -> List[ImageReference]:
            return [
                ImageReference(
                    image_id=img.image_id,
                    page_number=img.page_number,
                    storage_path=img.storage_path,
                    file_name=img.file_name,
                    score=img.score,
                )
                for img in items
            ]

        assigned_pages = set()
        for passage in passages:
            if passage.page_number in images_by_page:
                passage.images = to_image_refs(images_by_page[passage.page_number])
                assigned_pages.add(passage.page_number)

        for page_number in sorted(images_by_page.keys() - assigned_pages):
            passages.append(
                Passage(
                    page_number=page_number,
                    text="",
                    chunks=[],
                    images=to_image_refs(images_by_page[page_number]),
                )
            )

        passages.sort(key=lambda p: p.page_number)

    @staticmethod
    def _group_by_page(items: List[ChunkSearchResult]) -> GroupedChunks:
        """Group chunk results by page number and sort by chunk index."""
        by_page: GroupedChunks = defaultdict(list)
        for item in items:
            by_page[item.page_number].append(item)
        for page_items in by_page.values():
            page_items.sort(key=lambda it: it.chunk_index)
        return by_page

    def _merge_passages(self, grouped: GroupedChunks) -> List[Passage]:
        """Merge overlapping chunks per page into ordered passages."""
        passages: List[Passage] = []
        for page_number in sorted(grouped.keys()):
            passages.extend(self._merge_page_chunks(page_number, grouped[page_number]))
        return passages

    @staticmethod
    def _find_overlap(a: str, b: str) -> int:
        """Compute longest suffix-prefix overlap from a -> b."""
        a_tail = a[-_OVERLAP_WINDOW:]
        b_head = b[:_OVERLAP_WINDOW]
        for size in range(min(len(a_tail), len(b_head)), _MIN_OVERLAP - 1, -1):
            if a_tail.endswith(b_head[:size]):
                return size
        return 0

    @staticmethod
    def _try_join(current: str, new_text: str, adjacent: bool) -> str | None:
        """Join chunk texts by overlap or adjacency."""
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
        page_chunks: List[ChunkSearchResult],
    ) -> List[Passage]:
        """Merge consecutive or overlapping chunks on a single page."""
        if not page_chunks:
            return []

        first_chunk = page_chunks[0]
        run_text = first_chunk.text
        run_items: List[ChunkReference] = [
            ChunkReference(
                chunk_id=first_chunk.chunk_id,
                page_number=first_chunk.page_number,
                chunk_index=first_chunk.chunk_index,
                score=first_chunk.score,
            )
        ]
        run_end = first_chunk.chunk_index
        passages: List[Passage] = []

        for chunk in page_chunks[1:]:
            merged = ContextAssembler._try_join(
                run_text, chunk.text, adjacent=(chunk.chunk_index == run_end + 1)
            )
            chunk_ref = ChunkReference(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
            if merged is not None:
                run_text = merged
                run_items.append(chunk_ref)
                run_end = chunk.chunk_index
            else:
                passages.append(
                    Passage(
                        page_number=page_number,
                        text=run_text,
                        chunks=run_items,
                    )
                )
                run_text = chunk.text
                run_items = [chunk_ref]
                run_end = chunk.chunk_index

        passages.append(
            Passage(
                page_number=page_number,
                text=run_text,
                chunks=run_items,
            )
        )
        return passages
