"""
Read-only queries over RequestLog for the dashboard / CLI reporting:
fallback rate, estimated cost saved, accuracy trend. Nothing here writes
to the database -- pure reporting.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from cascade.storage.models import Handler, RequestLog


def fallback_rate(session: Session, task_id: str, window: int = 500) -> float | None:
    """Fraction of the most recent `window` requests that had to escalate to the LLM."""
    rows = (
        session.query(RequestLog.handler)
        .filter(RequestLog.task_id == task_id)
        .order_by(RequestLog.created_at.desc())
        .limit(window)
        .all()
    )
    if not rows:
        return None
    llm_calls = sum(1 for (h,) in rows if h == Handler.LLM)
    return llm_calls / len(rows)


def estimated_calls_saved(session: Session, task_id: str) -> int:
    """Total count of requests served by the student model instead of the LLM."""
    return (
        session.query(func.count(RequestLog.id))
        .filter(RequestLog.task_id == task_id, RequestLog.handler == Handler.STUDENT)
        .scalar()
    ) or 0


def task_summary(session: Session, task_id: str) -> dict:
    total = (
        session.query(func.count(RequestLog.id))
        .filter(RequestLog.task_id == task_id)
        .scalar()
    ) or 0
    student_served = estimated_calls_saved(session, task_id)
    return {
        "task_id": task_id,
        "total_requests": total,
        "served_by_student": student_served,
        "served_by_llm": total - student_served,
        "student_share": (student_served / total) if total else 0.0,
        "recent_fallback_rate": fallback_rate(session, task_id),
    }
