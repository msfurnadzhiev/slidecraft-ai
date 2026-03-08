"""Search router -- testing endpoint for semantic chunk retrieval."""

from fastapi import APIRouter, HTTPException

from app.api.dependencies import SearchService, ContextAssembler
from app.schemas.context import RetrievalContext
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=RetrievalContext)
def search_with_context(
    request: SearchRequest,
    search_service: SearchService,
    context_assembler: ContextAssembler,
):
    """Search and assemble into document-ordered context with images.

    Runs semantic search, then deduplicates, reorders by document position,
    merges overlapping chunks, and attaches page images.
    """
    try:
        search_response = search_service.search(
            document_id=request.document_id,
            query=request.query,
            chunk_limit=request.chunk_limit,
            image_limit=request.image_limit,
            threshold=request.threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return context_assembler.assemble(search_response)
