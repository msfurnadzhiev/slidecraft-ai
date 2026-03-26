"""CRUD operations for Template, SlideLayout, and LayoutElement objects."""

from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.db.models.template import LayoutElement, SlideLayout, Template
from src.schemas.template import SlideLayoutCreate, TemplateCreate

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def create_template(db: Session, template: TemplateCreate) -> Template:
    """Stage a new template (with its layouts and elements) in the session.

    Args:
        db: SQLAlchemy session.
        template: TemplateCreate schema.

    Returns:
        Template instance (not yet committed).
    """
    db_template = Template(
        template_id=uuid4(),
        name=template.name,
        file_path=template.file_path,
    )
    db.add(db_template)
    db.flush()

    for layout_data in template.layouts:
        _create_slide_layout(db, db_template.template_id, layout_data)

    return db_template


def get_template(db: Session, template_id: UUID) -> Optional[Template]:
    """Get a template by its ID (layouts and elements are not eagerly loaded).

    Args:
        db: SQLAlchemy session.
        template_id: Template primary key.

    Returns:
        Template instance if found, else None.
    """
    stmt = select(Template).where(Template.template_id == template_id)
    return db.execute(stmt).scalars().first()


def get_template_with_layouts(db: Session, template_id: UUID) -> Optional[Template]:
    """Get a template with all its slide layouts and layout elements loaded.

    Args:
        db: SQLAlchemy session.
        template_id: Template primary key.

    Returns:
        Template instance with eagerly loaded layouts/elements, or None.
    """
    stmt = (
        select(Template)
        .where(Template.template_id == template_id)
        .options(
            selectinload(Template.layouts).selectinload(SlideLayout.elements)
        )
    )
    return db.execute(stmt).scalars().first()


def get_all_templates(
    db: Session, skip: int = 0, limit: int = 100
) -> List[Template]:
    """Get all templates with pagination (layouts not eagerly loaded).

    Args:
        db: SQLAlchemy session.
        skip: Number of rows to skip.
        limit: Maximum number of rows to return.

    Returns:
        List of Template instances.
    """
    stmt = select(Template).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def delete_template(db: Session, template_id: UUID) -> bool:
    """Stage a template deletion in the session (cascades to layouts/elements).

    Args:
        db: SQLAlchemy session.
        template_id: Template primary key.

    Returns:
        True if the template was found and staged for deletion, False otherwise.
    """
    template = get_template(db, template_id)
    if template:
        db.delete(template)
        return True
    return False


# ---------------------------------------------------------------------------
# SlideLayout
# ---------------------------------------------------------------------------


def _create_slide_layout(
    db: Session, template_id: UUID, layout: SlideLayoutCreate
) -> SlideLayout:
    """Stage a slide layout (and its elements) attached to a template."""
    db_layout = SlideLayout(
        template_id=template_id,
        layout_index=layout.layout_index,
        name=layout.name,
    )
    db.add(db_layout)
    db.flush()

    for element_data in layout.elements:
        db_element = LayoutElement(
            layout_id=db_layout.id,
            **element_data.model_dump(),
        )
        db.add(db_element)

    return db_layout


def get_layouts_by_template(db: Session, template_id: UUID) -> List[SlideLayout]:
    """Return all slide layouts for a template, ordered by layout_index.

    Args:
        db: SQLAlchemy session.
        template_id: Parent template primary key.

    Returns:
        List of SlideLayout instances (elements not eagerly loaded).
    """
    stmt = (
        select(SlideLayout)
        .where(SlideLayout.template_id == template_id)
        .order_by(SlideLayout.layout_index)
    )
    return list(db.execute(stmt).scalars().all())


def get_layout_with_elements(
    db: Session, layout_id: int
) -> Optional[SlideLayout]:
    """Get a single slide layout with its elements eagerly loaded.

    Args:
        db: SQLAlchemy session.
        layout_id: SlideLayout primary key.

    Returns:
        SlideLayout instance with elements loaded, or None.
    """
    stmt = (
        select(SlideLayout)
        .where(SlideLayout.id == layout_id)
        .options(selectinload(SlideLayout.elements))
    )
    return db.execute(stmt).scalars().first()


def delete_layout(db: Session, layout_id: int) -> bool:
    """Stage a layout deletion in the session (cascades to elements).

    Args:
        db: SQLAlchemy session.
        layout_id: SlideLayout primary key.

    Returns:
        True if the layout was found and staged for deletion, False otherwise.
    """
    stmt = select(SlideLayout).where(SlideLayout.id == layout_id)
    layout = db.execute(stmt).scalars().first()
    if layout:
        db.delete(layout)
        return True
    return False
