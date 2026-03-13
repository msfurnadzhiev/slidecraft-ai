"""Context router: retrieve context for a document."""

from uuid import UUID

from fastapi import APIRouter, Body, HTTPException

from src.api.dependencies import ContentAnalyzer, ContextRetriever
from src.schemas.analysis import ContextAnalysis
from src.schemas.context import ContextRetrievalOptions, RetrievedContext

router = APIRouter(prefix="/context", tags=["context"])



@router.post("/{document_id}", response_model=RetrievedContext)
def retrieve_context(
    document_id: UUID,
    context_retriever: ContextRetriever,
    options: ContextRetrievalOptions | None = Body(None),
) -> RetrievedContext:
    """Retrieve document content with optional semantic search.

    Args:
        document_id: Target document (path parameter).
        context_retriever: Retriever to retrieve the context.
        options: Optional ContextRetrievalOptions body with query, limits, and thresholds.

    Raises:
        HTTPException: If the document is not found.

    Returns:
        ContextWithAnalysis containing ordered passages and content analysis.
    """
    try:
        retrieved_context: RetrievedContext = \
            context_retriever.retrieve_context(document_id, options)

        return retrieved_context
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{document_id}/analysis", response_model=ContextAnalysis)
def analyze_context(
    document_id: str,
    context_retriever: ContextRetriever,
    content_analyzer: ContentAnalyzer,
    options: ContextRetrievalOptions | None = Body(None),
) -> ContextAnalysis:
    """Analyze the context for a document and return the analysis.
    
    Args:
        document_id: Target document (path parameter).
        context_retriever: Retriever to retrieve the context.
        content_analyzer: Analyzer to analyze the context.
        options: Optional ContextRetrievalOptions body with query, limits, and thresholds.

    Returns:
        ContextAnalysis containing the analysis.
    """
    try:
        retrieved_context: RetrievedContext = \
            context_retriever.retrieve_context(document_id, options)

        analyzed_context: ContextAnalysis = \
            content_analyzer.analyze(retrieved_context)

        return analyzed_context
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))