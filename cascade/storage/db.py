"""
Database engine/session management. Works with SQLite out of the box for
local development; point DATABASE_URL at Postgres for production use.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cascade.storage.models import Base

_DEFAULT_URL = "sqlite:///./cascade.db"


def get_engine(database_url: str | None = None):
    url = database_url or os.getenv("DATABASE_URL", _DEFAULT_URL)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(database_url: str | None = None):
    """Create all tables if they don't exist yet. Call once at startup."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


_SessionLocal = None


def get_session_factory(database_url: str | None = None):
    global _SessionLocal
    if _SessionLocal is None:
        engine = init_db(database_url)
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return _SessionLocal


@contextmanager
def session_scope(database_url: str | None = None):
    """Usage: with session_scope() as session: session.add(obj)"""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
