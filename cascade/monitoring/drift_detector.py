"""
Lightweight drift check you can run more frequently than the full
training cycle (e.g. hourly) to catch fast degradation early, without
waiting for the next scheduled retrain. Wraps the same rolling-window
accuracy check used in stage_manager, but as a standalone alertable
function so it can be wired into monitoring/alerting separately.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from cascade.config import TaskConfig
from cascade.stages.stage_manager import StageManager


def check_drift(session: Session, config: TaskConfig) -> dict:
    manager = StageManager(session)
    window_accuracy = manager.compute_recent_accuracy(
        config.task_id, config.promotion.demotion_window_size
    )
    if window_accuracy is None:
        return {"task_id": config.task_id, "status": "insufficient_data"}

    is_drifting = window_accuracy < config.promotion.demotion_accuracy_floor
    return {
        "task_id": config.task_id,
        "status": "drifting" if is_drifting else "healthy",
        "rolling_accuracy": window_accuracy,
        "floor": config.promotion.demotion_accuracy_floor,
    }
