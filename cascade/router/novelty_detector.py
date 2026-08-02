"""
Novelty (out-of-distribution) detection: even if the student model reports
high confidence, an input that looks nothing like anything it was trained
on should still be treated with suspicion. This embeds the incoming input
and checks its similarity against everything logged for this task so far.
"""
from __future__ import annotations

import numpy as np

from cascade.storage.vector_store import VectorStore

_EMBEDDER = None


def _get_embedder():
    """Lazy-loaded so importing this module doesn't require sentence-transformers
    unless novelty detection is actually used."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


class NoveltyDetector:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def embed(self, text: str) -> np.ndarray:
        model = _get_embedder()
        return model.encode(text, convert_to_numpy=True)

    def novelty_score(self, task_id: str, text: str) -> float:
        """
        Returns a score in [0, 1] where 1 = completely novel (never seen
        anything similar) and 0 = near-identical to something already
        handled. This is (1 - max_similarity).
        """
        vector = self.embed(text)
        max_sim = self.vector_store.max_similarity(task_id, vector)
        max_sim = max(0.0, min(1.0, max_sim))  # clamp, cosine sim can dip slightly below 0
        return 1.0 - max_sim

    def record(self, task_id: str, text: str, metadata: dict | None = None) -> None:
        """Add this input to the task's seen-examples index."""
        vector = self.embed(text)
        self.vector_store.add(task_id, vector, metadata or {})
