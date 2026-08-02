"""
Classifier-type student model: the right fit for classification, routing,
tagging, and extraction-style tasks (the majority of "repetitive daily
task" use cases). Uses TF-IDF features + a calibrated linear classifier so
raw confidence scores are meaningful probabilities, not arbitrary logits.

For a different feature representation (e.g. sentence embeddings instead
of TF-IDF), swap `_vectorize` — the rest of the class is agnostic to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from cascade.student_models.base import Prediction, StudentModel


class ClassifierStudent(StudentModel):
    def __init__(self, task_id: str, text_field: str = "text"):
        self.task_id = task_id
        self.text_field = text_field
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self._base_clf = SGDClassifier(loss="log_loss", random_state=42)
        self.clf: CalibratedClassifierCV | None = None
        self._is_trained = False

    def _vectorize(self, input_data: dict):
        text = input_data.get(self.text_field, "")
        return self.vectorizer.transform([text])

    def predict(self, input_data: dict) -> Prediction:
        if not self._is_trained:
            return Prediction(output=None, raw_confidence=0.0, is_trained=False)

        X = self._vectorize(input_data)
        probs = self.clf.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))
        label = self.clf.classes_[top_idx]
        confidence = float(probs[top_idx])
        return Prediction(output=label, raw_confidence=confidence, is_trained=True)

    def fit(self, examples: list[dict], labels: list[Any]) -> None:
        """
        examples: list of input dicts, e.g. [{"text": "..."}, ...]
        labels:   list of correct outputs (the LLM's historical answers,
                  or human-verified ground truth), same length as examples
        """
        if len(examples) < 10:
            raise ValueError(
                f"Need at least 10 training examples, got {len(examples)}. "
                "Let shadow mode run longer before training."
            )
        texts = [ex.get(self.text_field, "") for ex in examples]
        X = self.vectorizer.fit_transform(texts)

        # CalibratedClassifierCV wraps the base classifier so predict_proba
        # outputs are genuinely calibrated probabilities (Platt scaling),
        # not raw uncalibrated logits -- this matters a lot for the router.
        self.clf = CalibratedClassifierCV(self._base_clf, cv=min(3, len(set(labels))))
        self.clf.fit(X, labels)
        self._is_trained = True

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "clf": self.clf, "text_field": self.text_field},
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.clf = data["clf"]
        self.text_field = data["text_field"]
        self._is_trained = self.clf is not None
