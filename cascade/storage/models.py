"""
Persistence models.

Three tables carry the whole system:
  - RequestLog: every single request that passed through the interceptor,
    what handled it (student or LLM), and whether it was later verified correct.
  - StageState: current lifecycle stage per task (shadow/assisted/autonomous)
    plus the rolling metrics used to decide promotion/demotion.
  - TrainingRun: history of retrain jobs, so you can audit what the student
    model learned from and when (governance-relevant).
"""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Column, DateTime, Enum as SAEnum, Float, Integer, String, Text, Boolean
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Stage(str, enum.Enum):
    SHADOW = "shadow"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


class Handler(str, enum.Enum):
    STUDENT = "student"
    LLM = "llm"


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    input_payload = Column(Text, nullable=False)      # JSON-serialized input
    student_output = Column(Text, nullable=True)       # JSON-serialized, if student ran
    student_confidence = Column(Float, nullable=True)
    llm_output = Column(Text, nullable=True)            # JSON-serialized, if LLM ran

    handler = Column(SAEnum(Handler), nullable=False)   # who actually served the response
    stage_at_request_time = Column(SAEnum(Stage), nullable=False)

    # filled in later, either by comparing student vs LLM (shadow mode)
    # or by explicit feedback (thumbs up/down, human review, downstream signal)
    was_correct = Column(Boolean, nullable=True)
    verified_at = Column(DateTime, nullable=True)


class StageState(Base):
    __tablename__ = "stage_states"

    task_id = Column(String, primary_key=True)
    current_stage = Column(SAEnum(Stage), default=Stage.SHADOW, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    # rolling metrics used by promotion_rules.py
    samples_seen = Column(Integer, default=0)
    recent_accuracy = Column(Float, nullable=True)
    consecutive_stable_cycles = Column(Integer, default=0)
    last_retrain_at = Column(DateTime, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, index=True, nullable=False)
    started_at = Column(DateTime, default=dt.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    num_training_examples = Column(Integer, nullable=True)
    resulting_accuracy = Column(Float, nullable=True)   # measured on held-out log data
    model_version_path = Column(String, nullable=True)  # where the trained artifact was saved
    notes = Column(Text, nullable=True)
