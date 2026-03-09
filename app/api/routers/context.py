"""Context router: retrieve context for a document."""

from fastapi import APIRouter, Body, HTTPException

from app.api.dependencies import ContextAssembler, ContextRetriever
from app.schemas.context import ContextRequest, RetrievalContext

router = APIRouter(prefix="/context", tags=["context"])


@router.post("/{document_id}", response_model=RetrievalContext)
def search_with_context(
    document_id: str,
    context_retriever: ContextRetriever,
    context_assembler: ContextAssembler,
    request: ContextRequest | None = Body(None),
) -> RetrievalContext:
    """Retrieve document content with optional semantic search.

    Args:
        document_id: Target document (path parameter).
        request: Optional SearchRequest body with query, limits, and thresholds.

    Raises:
        HTTPException: If the document is not found.

    Returns:
        ContextWithAnalysis containing ordered passages and content analysis.
    """
    try:
        raw_context = context_retriever.retrieve_context(document_id, request)

        return context_assembler.assemble(raw_context)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# @router.get("/{document_id}/analysis", response_model=RetrievalContext)
# def get_context_analysis(
#     document_id: str,
#     context_assembler: ContextAssembler,
#     context_retriever: ContextRetriever,
#     content_analyzer: ContentAnalyzer,
# ) -> RetrievalContext:
#     """Get the analysis of the context for a document."""
#     try:
#         raw_context = context_retriever.retrieve_context(document_id)
#         return context_assembler.assemble(raw_context)
#     except ValueError as e:
#         raise HTTPException(status_code=404, detail=str(e))