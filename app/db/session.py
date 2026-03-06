import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", None)

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

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
