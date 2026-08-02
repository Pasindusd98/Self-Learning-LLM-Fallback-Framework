"""
Pulls training data out of RequestLog for a given task. The "label" for
each example is the LLM's historical output during shadow/fallback
requests (or human-verified `was_correct` ground truth, when available) --
this is the core of the distillation loop: the teacher's past answers
become the student's training labels.
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from cascade.storage.models import Handler, RequestLog


def load_training_examples(session: Session, task_id: str, limit: int | None = None):
    """
    Returns (examples, labels) where examples are the deserialized input
    dicts and labels are the corresponding correct outputs -- sourced from
    whichever the LLM produced (since LLM is the ground-truth teacher) on
    every request where the LLM was actually called.
    """
    query = (
        session.query(RequestLog)
        .filter(RequestLog.task_id == task_id, RequestLog.handler == Handler.LLM)
        .order_by(RequestLog.created_at.asc())
    )
    if limit:
        query = query.limit(limit)

    examples, labels = [], []
    for row in query.all():
        examples.append(json.loads(row.input_payload))
        labels.append(json.loads(row.llm_output))
    return examples, labels


def load_calibration_pairs(session: Session, task_id: str):
    """
    Returns (raw_confidences, was_correct) pairs for fitting the confidence
    calibrator -- pulled from requests where the student model made a
    prediction AND we later found out (via was_correct) whether it was right.
    Shadow mode is the main source: student prediction vs LLM output
    agreement can auto-populate was_correct without needing manual review.
    """
    rows = (
        session.query(RequestLog)
        .filter(
            RequestLog.task_id == task_id,
            RequestLog.student_confidence.isnot(None),
            RequestLog.was_correct.isnot(None),
        )
        .all()
    )
    confidences = [r.student_confidence for r in rows]
    correctness = [r.was_correct for r in rows]
    return confidences, correctness


def auto_label_shadow_agreement(session: Session, task_id: str) -> int:
    """
    In shadow mode, both the student prediction and the LLM's real output
    are logged for the same request. This auto-fills `was_correct` by
    comparing them, so you don't need manual review just to bootstrap
    calibration data. Returns the number of rows updated.

    Note: exact-match agreement is a reasonable proxy for classification-
    style tasks; for open-ended generative tasks, replace this with a
    similarity threshold or a human review step instead.
    """
    rows = (
        session.query(RequestLog)
        .filter(
            RequestLog.task_id == task_id,
            RequestLog.student_output.isnot(None),
            RequestLog.llm_output.isnot(None),
            RequestLog.was_correct.is_(None),
        )
        .all()
    )
    updated = 0
    for row in rows:
        row.was_correct = row.student_output == row.llm_output
        updated += 1
    session.commit()
    return updated
