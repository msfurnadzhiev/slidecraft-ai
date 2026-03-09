"""Document API endpoints."""

import os
import tempfile
from typing import List

from fastapi import APIRouter, UploadFile, HTTPException

from app.api.dependencies import DocumentService
from app.schemas.document import DocumentIngestResponse, DocumentResponse
from app.schemas.image import ImageResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/all", response_model=List[DocumentResponse])
def list_documents(document_service: DocumentService):
    """
    List all documents stored in the database.

    Args:
        document_service: Injected DocumentService dependency.

    Returns:
        List of documents.
    """
    return document_service.list_documents()


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(file: UploadFile, document_service: DocumentService):
    """
    Upload a PDF document and process it through the ingestion pipeline.

    Args:
        file: The uploaded PDF file.
        document_service: Injected DocumentService dependency.

    Raises:
        HTTPException: If file is not a PDF.

    Returns:
        Document ingestion response.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        result = document_service.ingest_document(
            file_path=tmp_file_path,
            original_filename=file.filename,
        )
        return result
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@router.get("/{document_id}/images", response_model=List[ImageResponse])
def get_document_images(document_id: str, document_service: DocumentService):
    """
    List all images for a document.

    Args:
        document_id: The ID of the document.
        document_service: Injected DocumentService dependency.

    Raises:
        HTTPException: If document not found.

    Returns:
        List of images associated with the document.
    """
    doc = document_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_service.get_document_images(document_id)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, document_service: DocumentService):
    """
    Get a document by its ID.

    Args:
        document_id: The ID of the document.
        document_service: Injected DocumentService dependency.

    Raises:
        HTTPException: If document not found.

    Returns:
        Document response.
    """
    document = document_service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}")
async def delete_document(document_id: str, document_service: DocumentService):
    """
    Delete a document and all associated chunks and embeddings.

    Args:
        document_id: The ID of the document.
        document_service: Injected DocumentService dependency.

    Raises:
        HTTPException: If document not found.

    Returns:
        Success message and deleted document ID.
    """
    success = document_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully", "document_id": document_id}
