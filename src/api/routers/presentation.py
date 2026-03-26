"""Presentation router: generate presentation structure, content, and .pptx files."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.dependencies import (
    DocumentRepository,
    PresentationWorkflow,
    TemplateRepository,
)
from src.schemas.presentation.presentation import (
    PresentationWorkflowRequest,
    PresentationWorkflowResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/presentation", tags=["presentation"])


@router.post(
    "/generate",
    response_model=PresentationWorkflowResponse,
)
def run_presentation_workflow(
    request: PresentationWorkflowRequest,
    workflow: PresentationWorkflow,
    document_repository: DocumentRepository,
    template_repository: TemplateRepository,
):
    """Run the full end-to-end presentation generation pipeline in one call.

    Raises:
        HTTPException 404: Document or template not found.
        HTTPException 500: Any unrecoverable error during pipeline execution.

    Returns:
        :class:`PresentationWorkflowResponse` with the storage path, total
        slide count, and total number of quality revision attempts.
    """
    document = document_repository.get_document(request.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    template = template_repository.get_template(request.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")

    log.info(
        "Workflow endpoint — document_id=%s, template_id=%s, request='%s'",
        request.document_id,
        request.template_id,
        request.user_request[:120],
    )

    try:
        response = workflow.run(
            document_id=request.document_id,
            user_request=request.user_request,
            template_id=request.template_id,
            template_file_path=template.file_path,
            template_layouts=template.layouts,
        )
    except Exception as exc:
        log.exception("Workflow failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Presentation workflow failed: {exc}",
        ) from exc

    return response
