"""
Entry point for a scheduled retrain job (cron, Airflow, GitHub Actions
scheduled workflow -- whatever you have available). Run this on the
cadence set in each task's `retrain_schedule` config.

Example cron usage (daily at 2am):
    0 2 * * * cd /path/to/repo && python -m cascade.training.scheduler
"""
from __future__ import annotations

import logging
from pathlib import Path

from cascade.config import load_all_task_configs
from cascade.router.calibration import ConfidenceCalibrator
from cascade.stages.stage_manager import StageManager
from cascade.storage.db import session_scope
from cascade.student_models.registry import get_student_model
from cascade.training.trainer import run_training_cycle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cascade.scheduler")

_calibrators: dict[str, ConfidenceCalibrator] = {}


def run_all(tasks_dir: str = "configs/tasks", model_save_dir: str = "./models"):
    configs = load_all_task_configs(tasks_dir)
    if not configs:
        logger.warning("No task configs found in %s", tasks_dir)
        return

    with session_scope() as session:
        for task_id, config in configs.items():
            logger.info("Running retrain cycle for task: %s", task_id)
            student = get_student_model(config)
            calibrator = _calibrators.setdefault(task_id, ConfidenceCalibrator())

            accuracy = run_training_cycle(
                session=session,
                config=config,
                student_model=student,
                calibrator=calibrator,
                model_save_dir=model_save_dir,
            )
            if accuracy is None:
                logger.info("  -> not enough data yet, skipped training")
                continue

            manager = StageManager(session)
            new_stage = manager.evaluate_and_update(config, accuracy)
            logger.info(
                "  -> accuracy=%.3f, stage=%s", accuracy, new_stage.value
            )


if __name__ == "__main__":
    run_all()
