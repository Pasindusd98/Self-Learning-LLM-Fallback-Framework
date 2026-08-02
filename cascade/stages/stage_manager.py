"""
Owns the per-task Stage lifecycle (Shadow -> Assisted -> Autonomous, with
possible demotion). Reads/writes StageState in the database and calls out
to promotion_rules.py for the actual decision logic.

This runs on a slower cadence than the router (per-retrain-cycle, not
per-request) -- see training/scheduler.py for how it's triggered.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func
from sqlalchemy.orm import Session

from cascade.config import TaskConfig
from cascade.stages.promotion_rules import check_demotion, check_promotion
from cascade.storage.models import Handler, RequestLog, Stage, StageState


class StageManager:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create_state(self, task_id: str) -> StageState:
        state = self.session.get(StageState, task_id)
        if state is None:
            state = StageState(task_id=task_id, current_stage=Stage.SHADOW)
            self.session.add(state)
            self.session.commit()
        return state

    def current_stage(self, task_id: str) -> Stage:
        return self.get_or_create_state(task_id).current_stage

    def compute_recent_accuracy(self, task_id: str, window: int) -> float | None:
        """Accuracy over the most recent `window` verified requests."""
        rows = (
            self.session.query(RequestLog)
            .filter(RequestLog.task_id == task_id, RequestLog.was_correct.isnot(None))
            .order_by(RequestLog.created_at.desc())
            .limit(window)
            .all()
        )
        if not rows:
            return None
        correct = sum(1 for r in rows if r.was_correct)
        return correct / len(rows)

    def evaluate_and_update(self, config: TaskConfig, this_cycle_accuracy: float | None) -> Stage:
        """
        Call this once per retrain cycle (see training/scheduler.py).
        Updates stability counters, checks promotion, then checks demotion
        against the rolling window -- demotion always takes priority since
        drift safety matters more than growth.
        """
        state = self.get_or_create_state(config.task_id)

        samples_seen = (
            self.session.query(func.count(RequestLog.id))
            .filter(RequestLog.task_id == config.task_id)
            .scalar()
        ) or 0
        state.samples_seen = samples_seen

        if this_cycle_accuracy is not None:
            if state.recent_accuracy is not None and this_cycle_accuracy >= (
                config.thresholds.assisted_mode
                if state.current_stage == Stage.SHADOW
                else config.thresholds.autonomous_mode
            ):
                state.consecutive_stable_cycles += 1
            else:
                state.consecutive_stable_cycles = 0
            state.recent_accuracy = this_cycle_accuracy

        new_stage = check_promotion(
            current_stage=state.current_stage,
            recent_accuracy=state.recent_accuracy,
            samples_seen=state.samples_seen,
            consecutive_stable_cycles=state.consecutive_stable_cycles,
            rules=config.promotion,
            thresholds=config.thresholds,
        )

        # Demotion guard: check rolling window accuracy regardless of the
        # promotion outcome above. This is what protects against drift
        # after a task has already been promoted.
        window_accuracy = self.compute_recent_accuracy(
            config.task_id, config.promotion.demotion_window_size
        )
        if window_accuracy is not None:
            new_stage = check_demotion(new_stage, window_accuracy, config.promotion)

        if new_stage != state.current_stage:
            state.current_stage = new_stage
        state.last_retrain_at = dt.datetime.utcnow()
        state.updated_at = dt.datetime.utcnow()
        self.session.commit()
        return new_stage
