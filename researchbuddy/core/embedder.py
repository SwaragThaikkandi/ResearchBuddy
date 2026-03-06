"""
embedder.py
Singleton wrapper around sentence-transformers.
Loaded once per process; all modules share the same instance.
"""

from __future__ import annotations
import numpy as np
from typing import Union

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from researchbuddy.config import EMBEDDING_MODEL
        print(f"[embedder] Loading model '{EMBEDDING_MODEL}' (first run only) ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[embedder] Model ready.")
    return _model


def embed(texts: Union[str, list[str]]) -> np.ndarray:
    """
    Encode one string or a list of strings.
    Returns shape (dim,) for a single string, (n, dim) for a list.
    """
    model  = _get_model()
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs[0] if single else vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors (already L2-normalised)."""
    return float(np.dot(a, b))


def mean_pool(vecs: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of embedding vectors, then L2-normalised."""
    if not vecs:
        raise ValueError("Cannot pool an empty list of vectors.")
    mat = np.stack(vecs)
    if weights:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        result = (mat * w[:, None]).sum(axis=0)
    else:
        result = mat.mean(axis=0)
    norm = np.linalg.norm(result)
    return result / norm if norm > 0 else result
