"""Database initialization: extensions and table creation."""

from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    """Create vector extension and all tables. Safe to call at application startup."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
