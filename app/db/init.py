"""Database initialization: extensions and table creation."""

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    """Initialize the database by creating the vector extension and all tables."""
    _create_vector_extension(engine)
    Base.metadata.create_all(bind=engine)


def _create_vector_extension(engine: Engine) -> None:
    """
    Ensure the PostgreSQL vector extension is installed.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
