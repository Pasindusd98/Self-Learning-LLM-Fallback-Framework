"""
Confidence calibration. Raw model confidence (softmax probability, logit
margin, etc.) is notoriously overconfident -- a model can output 0.98
"confidence" and still be wrong 30% of the time. This module fits a small
correction so confidence scores are trustworthy enough to gate real
decisions on.

Two approaches are provided:
  - Temperature scaling: for generative/neural models with logits.
  - Isotonic/Platt scaling: for classifiers (also handled inline by
    sklearn's CalibratedClassifierCV in classifier_student.py -- this
    module exists for cases where you're not using sklearn's classifier).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ConfidenceCalibrator:
    """
    Fits a monotonic mapping from raw_confidence -> P(correct), using
    logged (raw_confidence, was_correct) pairs. Isotonic regression is
    used because it makes no assumption about the shape of the
    miscalibration curve.
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._is_fit = False

    def fit(self, raw_confidences: list[float], was_correct: list[bool]) -> None:
        if len(raw_confidences) < 20:
            raise ValueError(
                "Need at least 20 labeled (confidence, correctness) pairs "
                "to fit a calibration curve reliably."
            )
        X = np.array(raw_confidences)
        y = np.array([1.0 if c else 0.0 for c in was_correct])
        self._iso.fit(X, y)
        self._is_fit = True

    def calibrate(self, raw_confidence: float) -> float:
        """Returns a calibrated confidence. Falls back to the raw value
        (with a conservative haircut) if calibration hasn't been fit yet."""
        if not self._is_fit:
            return raw_confidence * 0.8  # conservative default before enough data exists
        return float(self._iso.predict([raw_confidence])[0])
