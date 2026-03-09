"""Search router: endpoint for semantic chunk retrieval."""

from fastapi import APIRouter, HTTPException

from app.api.dependencies import SearchService, ContextAssembler
from app.schemas.context import RetrievalContext
from app.schemas.search import SearchRequest

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=RetrievalContext)
def search_with_context(
    request: SearchRequest,
    search_service: SearchService,
    context_assembler: ContextAssembler,
) -> RetrievalContext:
    """Perform semantic search and assemble results into a document-ordered context.

    Args:
        request: SearchRequest schema containing query, document ID, limits, and thresholds.
        search_service: Injected SearchService dependency.
        context_assembler: Injected ContextAssembler dependency.

    Raises:
        HTTPException: If the document is not found or search fails.

    Returns:
        RetrievalContext containing ordered chunks and associated images.
    """
    try:
        search_response = search_service.search(
            request=request,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return context_assembler.assemble(search_response)