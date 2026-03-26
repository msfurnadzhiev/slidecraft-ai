"""Document API endpoints."""

import os
import tempfile
from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.dependencies import DocumentIngestionPipeline, DocumentRepository
from src.schemas.document import DocumentIngestResponse, DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    pipeline: DocumentIngestionPipeline,
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
):
    """Upload a PDF document and process it through the ingestion pipeline.

    Send **multipart/form-data** with one file part named **file** (e.g. curl
    `-F "file=@/path/to/doc.pdf"`). JSON bodies or wrong field names return 422.

    Raises:
        HTTPException 400: If file is not a PDF.

    Returns:
        Document ingestion response.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        return pipeline.ingest(
            file_path=tmp_file_path,
            original_filename=file.filename,
        )
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@router.get("/all", response_model=List[DocumentResponse])
def list_documents(repository: DocumentRepository):
    """List all documents stored in the database."""
    return repository.list_documents()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID, repository: DocumentRepository):
    """Get a document by its ID.

    Raises:
        HTTPException 404: If document not found.
    """
    document = repository.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}")
async def delete_document(document_id: UUID, repository: DocumentRepository):
    """Delete a document and all associated chunks and embeddings.

    Raises:
        HTTPException 404: If document not found.
    """
    success = repository.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully", "document_id": document_id}
