"""Database session management for SQLAlchemy."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


# Read database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")


# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, future=True)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
    expire_on_commit=False,
)


def get_db():
    """Yield a database session scoped to a single request.

    Commits automatically when the request handler returns successfully.
    Rolls back on any unhandled exception, ensuring atomicity across the
    entire request without services or CRUD functions needing to manage
    transaction boundaries themselves.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
