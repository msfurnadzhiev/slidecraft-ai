"""Template API endpoints."""

import os
import tempfile
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile

from src.api.dependencies import TemplateIngestionPipeline, TemplateRepository
from src.schemas.template import TemplateResponse, TemplateWithLayoutsResponse

router = APIRouter(prefix="/templates", tags=["templates"])


@router.post("/upload", response_model=TemplateWithLayoutsResponse)
async def upload_template(
    file: UploadFile,
    pipeline: TemplateIngestionPipeline,
):
    """Upload a .pptx template file and extract its slide layout metadata.

    The file is parsed immediately and all layout/element information is
    persisted to the database.

    Raises:
        HTTPException 400: If the uploaded file is not a .pptx.

    Returns:
        Fully nested template response including all layouts and elements.
    """
    if not file.filename or not file.filename.lower().endswith(".pptx"):
        raise HTTPException(status_code=400, detail="Only .pptx files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return pipeline.ingest(
            tmp_path=tmp_path,
            original_filename=file.filename,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/all", response_model=List[TemplateResponse])
def list_templates(repository: TemplateRepository):
    """List all stored templates (without layout details)."""
    return repository.list_templates()


@router.get("/{template_id}", response_model=TemplateWithLayoutsResponse)
def get_template(template_id: UUID, repository: TemplateRepository):
    """Retrieve a single template with all its slide layouts and elements.

    Raises:
        HTTPException 404: If template not found.
    """
    template = repository.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    return template


@router.delete("/{template_id}")
def delete_template(template_id: UUID, repository: TemplateRepository):
    """Delete a template and all its associated layouts and elements.

    Raises:
        HTTPException 404: If template not found.
    """
    success = repository.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found.")
    return {"message": "Template deleted successfully.", "template_id": template_id}
