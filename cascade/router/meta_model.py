"""
Optional upgrade path from ConfidenceRouter's fixed weighted combination:
a small learned model (gradient-boosted trees) that takes
(raw_confidence, novelty_score, ...) as features and predicts
P(student_output_is_correct). This is the same idea RouteLLM uses for
its router.

Only worth switching to once you have a few hundred labeled
(signals -> was_correct) examples logged -- before that, the simple
weighted combination in confidence.py is more robust (less prone to
overfitting on tiny samples).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class MetaRouter:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        self._is_fit = False

    def fit(self, raw_confidences: list[float], novelty_scores: list[float], was_correct: list[bool]):
        if len(raw_confidences) < 200:
            raise ValueError(
                "Need at least 200 labeled examples to train a meta-router "
                "reliably. Use ConfidenceRouter's weighted combination until then."
            )
        X = np.column_stack([raw_confidences, novelty_scores])
        y = np.array([1 if c else 0 for c in was_correct])
        self.model.fit(X, y)
        self._is_fit = True

    def predict_proba(self, raw_confidence: float, novelty_score: float) -> float:
        if not self._is_fit:
            raise RuntimeError("MetaRouter not fit yet -- call fit() first.")
        X = np.array([[raw_confidence, novelty_score]])
        return float(self.model.predict_proba(X)[0][1])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self._is_fit = True
