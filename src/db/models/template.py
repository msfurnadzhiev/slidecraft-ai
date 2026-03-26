"""This module defines the Template, SlideLayout, and LayoutElement SQLAlchemy models."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models import Base as BaseModel


class Template(BaseModel):
    __tablename__ = "templates"

    template_id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)

    layouts: Mapped[List["SlideLayout"]] = relationship(
        back_populates="template",
        cascade="all, delete-orphan"
    )


class SlideLayout(BaseModel):
    __tablename__ = "slide_layouts"

    id: Mapped[int] = mapped_column(primary_key=True)

    template_id: Mapped[UUID] = mapped_column(
        ForeignKey("templates.template_id", ondelete="CASCADE"),
        nullable=False,
    )

    layout_index: Mapped[int] = mapped_column()
    name: Mapped[Optional[str]] = mapped_column(String)

    template: Mapped["Template"] = relationship(back_populates="layouts")

    elements: Mapped[List["LayoutElement"]] = relationship(
        back_populates="layout",
        cascade="all, delete-orphan"
    )


class LayoutElement(BaseModel):
    __tablename__ = "layout_elements"

    id: Mapped[int] = mapped_column(primary_key=True)

    layout_id: Mapped[int] = mapped_column(
        ForeignKey("slide_layouts.id"),
        nullable=False
    )

    placeholder_idx: Mapped[int] = mapped_column()

    # Semantic role used by the agent: "title", "content", "image", etc.
    role: Mapped[Optional[str]] = mapped_column(String)

    # Normalised geometry (0–1 range)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)

    layout: Mapped["SlideLayout"] = relationship(back_populates="elements")
   