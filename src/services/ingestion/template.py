"""Repository for template read and delete operations."""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.crud import template_crud
from src.infrastructure.storage.template_storage import TemplateStorage
from src.schemas.template import (
    LayoutElementResponse,
    SlideLayoutResponse,
    TemplateResponse,
    TemplateWithLayoutsResponse,
)

log = logging.getLogger(__name__)

class TemplateRepository:
    """Thin façade over template CRUD and associated file cleanup."""

    def __init__(self, db: Session, template_storage: TemplateStorage) -> None:
        self.db = db
        self.storage = template_storage

    def get_template(self, template_id: UUID) -> Optional[TemplateWithLayoutsResponse]:
        """Retrieve a single template with all its layouts and elements."""
        db_template = template_crud.get_template_with_layouts(self.db, template_id)
        if db_template is None:
            return None
        return self._build_full_response(db_template)

    def get_template_absolute_path(self, template_id: UUID) -> Optional[str]:
        """Return the absolute filesystem path of a stored template file."""
        db_template = template_crud.get_template(self.db, template_id)
        if db_template is None:
            return None
        return self.storage.get_absolute_path(db_template.file_path)

    def list_templates(self, skip: int = 0, limit: int = 100) -> List[TemplateResponse]:
        """List all templates with pagination (layouts not included)."""
        db_templates = template_crud.get_all_templates(self.db, skip=skip, limit=limit)
        return [
            TemplateResponse(
                template_id=t.template_id,
                name=t.name,
                file_path=t.file_path,
            )
            for t in db_templates
        ]

    def delete_template(self, template_id: UUID) -> bool:
        """Delete a template, its layouts/elements, and the stored .pptx file."""
        db_template = template_crud.get_template(self.db, template_id)
        if db_template is None:
            return False

        storage_path = db_template.file_path

        template_crud.delete_template(self.db, template_id)
        self.db.flush()

        deleted_file = self.storage.delete(storage_path)
        if not deleted_file:
            log.warning(
                "Template template_id=%s DB record deleted but file '%s' was not found on disk.",
                template_id,
                storage_path,
            )

        log.info("Template template_id=%s deleted.", template_id)
        return True

    @staticmethod
    def _build_full_response(db_template) -> TemplateWithLayoutsResponse:
        """Build a fully nested TemplateWithLayoutsResponse from an ORM object."""
        layouts = []
        for layout in db_template.layouts:
            elements = [
                LayoutElementResponse(
                    id=el.id,
                    layout_id=el.layout_id,
                    placeholder_idx=el.placeholder_idx,
                    role=el.role,
                    x=el.x,
                    y=el.y,
                    width=el.width,
                    height=el.height,
                )
                for el in layout.elements
            ]
            layouts.append(
                SlideLayoutResponse(
                    id=layout.id,
                    template_id=layout.template_id,
                    layout_index=layout.layout_index,
                    name=layout.name,
                    elements=elements,
                )
            )

        return TemplateWithLayoutsResponse(
            template_id=db_template.template_id,
            name=db_template.name,
            file_path=db_template.file_path,
            layouts=layouts,
        )
