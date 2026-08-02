"""
The core routing decision: given a student model's prediction, decide
whether to serve it directly or escalate to the LLM.

Combines three signals, as discussed in the architecture:
  1. Calibrated model-intrinsic confidence (from calibration.py)
  2. Novelty / OOD score (from novelty_detector.py) -- inverted and blended in
  3. (Optional) Agreement-based confidence via a learned meta-model
     (meta_model.py) once enough labeled data exists

Early on (little data), this falls back to a simple weighted combination.
Once enough (signal -> was_correct) pairs are logged, `meta_model.py` can
replace the weighted combination with a learned router for a better fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cascade.router.calibration import ConfidenceCalibrator
from cascade.router.novelty_detector import NoveltyDetector
from cascade.student_models.base import Prediction


@dataclass
class RoutingDecision:
    should_use_student: bool
    combined_confidence: float
    raw_confidence: float
    novelty_score: Optional[float]
    reason: str


class ConfidenceRouter:
    """
    Weighted combination is intentionally simple and transparent by
    default -- swap `combine_fn` for `meta_model.MetaRouter.predict_proba`
    once you have enough labeled routing outcomes to train it (see
    docs/confidence_scoring.md).
    """

    def __init__(
        self,
        calibrator: ConfidenceCalibrator,
        novelty_detector: Optional[NoveltyDetector] = None,
        novelty_weight: float = 0.3,
    ):
        self.calibrator = calibrator
        self.novelty_detector = novelty_detector
        self.novelty_weight = novelty_weight

    def decide(
        self,
        task_id: str,
        prediction: Prediction,
        input_text: Optional[str] = None,
        threshold: float = 0.85,
    ) -> RoutingDecision:
        if not prediction.is_trained:
            return RoutingDecision(
                should_use_student=False,
                combined_confidence=0.0,
                raw_confidence=0.0,
                novelty_score=None,
                reason="student model not yet trained (still in shadow mode)",
            )

        calibrated = self.calibrator.calibrate(prediction.raw_confidence)

        novelty = None
        combined = calibrated
        if self.novelty_detector is not None and input_text is not None:
            novelty = self.novelty_detector.novelty_score(task_id, input_text)
            # High novelty pulls combined confidence down, even if the
            # model itself claims to be sure -- this is what catches
            # "confidently wrong on unfamiliar input" failures.
            combined = calibrated * (1 - self.novelty_weight) + (1 - novelty) * self.novelty_weight

        should_use_student = combined >= threshold
        reason = (
            f"combined confidence {combined:.3f} {'>=' if should_use_student else '<'} "
            f"threshold {threshold:.3f}"
        )
        return RoutingDecision(
            should_use_student=should_use_student,
            combined_confidence=combined,
            raw_confidence=prediction.raw_confidence,
            novelty_score=novelty,
            reason=reason,
        )
