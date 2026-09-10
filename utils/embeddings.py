"""Local sentence-transformer embeddings used everywhere retrieval-like behavior is needed."""
from __future__ import annotations

import numpy as np

_MODEL = None


def get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    vectors = get_model(model_name).encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)
