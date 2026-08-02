"""Abstract interface so the fallback LLM is swappable without touching router logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Send `prompt` to the LLM and return its raw text response."""
        raise NotImplementedError
