"""Template ingestion pipeline: parse → store file → persist metadata."""

import logging
import os
from uuid import uuid4

from sqlalchemy.orm import Session

from src.db.crud import template_crud
from src.infrastructure.loaders.ppt_loader import PPTLoader
from src.infrastructure.storage.template_storage import TemplateStorage
from src.schemas.template import (
    LayoutElementResponse,
    SlideLayoutResponse,
    TemplateCreate,
    TemplateWithLayoutsResponse,
)

log = logging.getLogger(__name__)


class TemplateIngestionPipeline:
    """Orchestrates the full template ingestion flow."""

    def __init__(
        self,
        db: Session,
        ppt_loader: PPTLoader,
        template_storage: TemplateStorage,
    ) -> None:
        """Initialize the template ingestion pipeline."""
        self.db = db
        self.loader = ppt_loader
        self.storage = template_storage

    # ---------------------------------------------------------------------------
    # Ingestion pipeline
    # ---------------------------------------------------------------------------

    def ingest(self, tmp_path: str, original_filename: str) -> TemplateWithLayoutsResponse:
        """Run the full ingestion pipeline for a single .pptx template.

        Args:
            tmp_path: Absolute path to the temporary upload on disk.
            original_filename: Original filename supplied by the client.

        Returns:
            Fully nested TemplateWithLayoutsResponse.
        """
        log.info("Ingesting template '%s'.", original_filename)

        content = self._parse_template(tmp_path, original_filename)
        storage_path = self._store_file(tmp_path, original_filename)

        db_template = self._persist_template(content, storage_path, original_filename)

        log.info(
            "Template '%s' (id=%s) persisted with %d layout(s)",
            db_template.name,
            db_template.template_id,
            len(db_template.layouts),
        )

        return self._build_full_response(db_template)

    def _parse_template(self, path: str, filename: str):
        content = self.loader.load_template(path)

        log.info(
            "Parsed '%s' - %d layout(s)",
            filename,
            len(content.layouts),
        )

        return content

    def _store_file(self, tmp_path: str, filename: str) -> str:
        storage_name = self._generate_storage_name(filename)

        storage_path = self.storage.save(
            source_path=tmp_path,
            storage_name=storage_name,
        )

        log.info("Stored template file as '%s'", storage_path)
        return storage_path
        
    def _persist_template(self, content, storage_path: str, filename: str):
        template_create = TemplateCreate(
            name=self._extract_display_name(filename),
            file_path=storage_path,
            layouts=content.layouts,
        )

        db_template = template_crud.create_template(self.db, template_create)
        self.db.flush()

        return db_template

    # ---------------------------------------------------------------------------
    # Response mapping
    # ---------------------------------------------------------------------------

    def _build_response(self, db_template) -> TemplateWithLayoutsResponse:
        """Build a fully nested TemplateWithLayoutsResponse from an ORM object."""
        return TemplateWithLayoutsResponse(
            template_id=db_template.template_id,
            name=db_template.name,
            file_path=db_template.file_path,
            layouts=[
                self._map_layout(layout)
                for layout in db_template.layouts
            ],
        )

    def _map_layout(self, layout) -> SlideLayoutResponse:
        """Map a SlideLayout to a SlideLayoutResponse."""
        return SlideLayoutResponse(
            id=layout.id,
            template_id=layout.template_id,
            layout_index=layout.layout_index,
            name=layout.name,
            elements=[
                self._map_element(el)
                for el in layout.elements
            ],
        )

    @staticmethod
    def _map_element(el) -> LayoutElementResponse:
        """Map a LayoutElement to a LayoutElementResponse."""
        return LayoutElementResponse(
            id=el.id,
            layout_id=el.layout_id,
            placeholder_idx=el.placeholder_idx,
            role=el.role,
            x=el.x,
            y=el.y,
            width=el.width,
            height=el.height,
        )

    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------

    @staticmethod
    def _generate_storage_name(filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        return f"{uuid4()}{ext}"

    @staticmethod
    def _extract_display_name(filename: str) -> str:
        return os.path.splitext(filename)[0]
