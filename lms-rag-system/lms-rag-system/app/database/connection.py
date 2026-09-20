"""
Database Connection Manager
----------------------------
Manages PostgreSQL connections (with pgvector support) via SQLAlchemy.
Credentials are read from the DATABASE_URL environment variable so that
docker-compose, .env files, and CI all work without code changes.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# ── connection ──────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:my_password@localhost:5433/lms_db",  # FIX Bug 7: was lms_db
)

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,          # keep it simple; swap for QueuePool in prod
    echo=False,
    connect_args={
        "connect_timeout": 10,
        "application_name": "rag-system",
    },
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── session helper ───────────────────────────────────────────────────────────
@contextmanager
def get_db() -> Generator:
    """Yield a SQLAlchemy session and commit/rollback automatically."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Database error: %s", exc, exc_info=True)
        raise
    finally:
        db.close()


# ── convenience query runner ─────────────────────────────────────────────────
def execute_query(query: str, params: dict | None = None) -> list[dict]:
    """
    Run a raw SQL query and return results as a list of dicts.

    NOTE: This is synchronous; the async wrapper in the tutorial was a bug
    (execute_query was not actually async).  Call it normally.
    """
    with get_db() as db:
        result = db.execute(text(query), params or {})
        columns = list(result.keys())
        rows = []
        for row in result.fetchall():
            row_dict: dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # serialise datetime objects for JSON compatibility
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                row_dict[col] = value
            rows.append(row_dict)
        return rows