"""
Runs a single retrain cycle for one task: pull data, fit the student
model, fit the calibrator, evaluate held-out accuracy, save artifacts,
and record a TrainingRun row for auditability.
"""
from __future__ import annotations

import datetime as dt
import random

from sqlalchemy.orm import Session

from cascade.config import TaskConfig
from cascade.router.calibration import ConfidenceCalibrator
from cascade.storage.models import TrainingRun
from cascade.student_models.base import StudentModel
from cascade.training.data_loader import (
    auto_label_shadow_agreement,
    load_calibration_pairs,
    load_training_examples,
)


def run_training_cycle(
    session: Session,
    config: TaskConfig,
    student_model: StudentModel,
    calibrator: ConfidenceCalibrator,
    model_save_dir: str = "./models",
    holdout_fraction: float = 0.2,
) -> float | None:
    """
    Returns the held-out accuracy achieved this cycle, or None if there
    wasn't enough data to train yet.
    """
    run = TrainingRun(task_id=config.task_id, started_at=dt.datetime.utcnow())
    session.add(run)
    session.commit()

    examples, labels = load_training_examples(session, config.task_id)
    if len(examples) < config.promotion.min_samples_before_promotion // 5:
        # Not worth a training pass yet -- avoid overfitting on scraps.
        run.finished_at = dt.datetime.utcnow()
        run.notes = f"Skipped: only {len(examples)} examples logged so far."
        session.commit()
        return None

    # simple holdout split for evaluating this cycle's accuracy
    combined = list(zip(examples, labels))
    random.Random(42).shuffle(combined)
    split = int(len(combined) * (1 - holdout_fraction))
    train_set, holdout_set = combined[:split], combined[split:]

    train_examples, train_labels = zip(*train_set) if train_set else ([], [])
    student_model.fit(list(train_examples), list(train_labels))

    accuracy = None
    if holdout_set:
        correct = 0
        for ex, true_label in holdout_set:
            pred = student_model.predict(ex)
            if pred.is_trained and pred.output == true_label:
                correct += 1
        accuracy = correct / len(holdout_set)

    student_model.save(f"{model_save_dir}/{config.task_id}.joblib")

    # Refresh calibration data too -- auto-label shadow-mode agreements
    # first so there's more (confidence, correctness) signal to fit on.
    auto_label_shadow_agreement(session, config.task_id)
    confidences, correctness = load_calibration_pairs(session, config.task_id)
    if len(confidences) >= 20:
        calibrator.fit(confidences, correctness)

    run.finished_at = dt.datetime.utcnow()
    run.num_training_examples = len(train_examples)
    run.resulting_accuracy = accuracy
    run.model_version_path = f"{model_save_dir}/{config.task_id}.joblib"
    session.commit()

    return accuracy
