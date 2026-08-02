"""
Abstract interface every student model type must implement. Keeping this
minimal is what lets the router and stage manager stay agnostic to whether
the underlying model is a classifier, a LoRA-tuned generative model, or a
rule-based hybrid.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    output: Any
    raw_confidence: float          # model-intrinsic confidence, e.g. softmax prob
    is_trained: bool = True         # False if the model hasn't been fit yet


class StudentModel(ABC):
    task_id: str

    @abstractmethod
    def predict(self, input_data: dict) -> Prediction:
        """Run inference and return an output plus a raw (uncalibrated) confidence."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, examples: list[dict], labels: list[Any]) -> None:
        """Train (or retrain) on logged (input, correct_output) pairs."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError
