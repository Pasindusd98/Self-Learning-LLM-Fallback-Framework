"""
Minimal vector store abstraction used only for novelty/OOD detection
(the "have I seen something like this before?" check in the router).

Default backend is a local in-memory/on-disk numpy store so the framework
runs with zero external services. Swap in Qdrant or pgvector for
production scale by implementing the same interface.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore:
    """Abstract interface. Implement add() and query() for a new backend."""

    def add(self, task_id: str, vector: np.ndarray, metadata: dict) -> None:
        raise NotImplementedError

    def max_similarity(self, task_id: str, vector: np.ndarray) -> float:
        """Return the highest cosine similarity between `vector` and anything
        stored for this task. Used as the novelty signal: low max similarity
        means the input is unlike anything the student has seen."""
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """
    Simple flat-file, in-memory-at-runtime store. Fine for prototyping and
    for tasks with up to tens of thousands of logged examples. For larger
    scale, swap to Qdrant/pgvector without changing calling code.
    """

    def __init__(self, persist_dir: str = "./vector_store_data"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[np.ndarray]] = {}

    def _path(self, task_id: str) -> Path:
        return self.persist_dir / f"{task_id}.npy"

    def _load(self, task_id: str) -> np.ndarray:
        if task_id in self._cache:
            return np.array(self._cache[task_id]) if self._cache[task_id] else np.empty((0,))
        path = self._path(task_id)
        if path.exists():
            arr = np.load(path)
            self._cache[task_id] = list(arr)
            return arr
        self._cache[task_id] = []
        return np.empty((0,))

    def add(self, task_id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        self._load(task_id)  # ensure cache populated
        self._cache[task_id].append(vector)
        arr = np.array(self._cache[task_id])
        np.save(self._path(task_id), arr)

    def max_similarity(self, task_id: str, vector: np.ndarray) -> float:
        arr = self._load(task_id)
        if len(arr) == 0:
            return 0.0  # nothing seen yet -> treat as maximally novel
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        normed = arr / norms
        sims = normed @ vector
        return float(np.max(sims))
