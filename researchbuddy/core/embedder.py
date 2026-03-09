"""
embedder.py
Singleton wrapper around sentence-transformers.
Loaded once per process; all modules share the same instance.
"""

from __future__ import annotations
import numpy as np
import sys
import types
from typing import Union

_model = None
_device = None


def _guard_torchvision():
    """
    Prevent a broken torchvision install from crashing sentence-transformers.

    sentence_transformers → transformers → torchvision.transforms (for vision
    models we never use).  If torch and torchvision versions are mismatched
    the import raises ``RuntimeError: operator torchvision::nms does not exist``.

    Fix: if torchvision fails to import, inject a minimal stub so the
    downstream ``from torchvision.transforms import InterpolationMode``
    succeeds harmlessly.  Text embeddings are unaffected.
    """
    if "torchvision" in sys.modules:
        return                              # already loaded (or already stubbed)
    try:
        import torchvision                  # noqa: F401 — test if it loads
    except (RuntimeError, ImportError, OSError):
        # ── Build a minimal stub ────────────────────────────────────────
        tv = types.ModuleType("torchvision")
        tv.__path__ = []                    # mark as package

        # torchvision.transforms + InterpolationMode (used by transformers)
        transforms = types.ModuleType("torchvision.transforms")
        transforms.InterpolationMode = type(
            "InterpolationMode", (),
            {"BILINEAR": 2, "NEAREST": 0, "BICUBIC": 3,
             "LANCZOS": 1, "BOX": 4, "HAMMING": 5},
        )()
        tv.transforms = transforms

        # torchvision.transforms.functional (imported by some transformers code)
        functional = types.ModuleType("torchvision.transforms.functional")
        tv.transforms.functional = functional

        # Register stubs so Python's import system finds them
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.functional"] = functional
        sys.modules["torchvision._meta_registrations"] = types.ModuleType(
            "torchvision._meta_registrations"
        )

        print("[embedder] Warning: torchvision is broken or missing — "
              "stubbed out (not needed for text embeddings).")
        print("[embedder] To fix permanently: pip install --upgrade torch torchvision")


def _resolve_device() -> str:
    """Pick the best device for embedding: config override → CUDA → CPU."""
    from researchbuddy.config import EMBEDDING_DEVICE
    if EMBEDDING_DEVICE == "cuda":
        return "cuda"
    if EMBEDDING_DEVICE == "cpu":
        return "cpu"
    # "auto" — try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_model():
    global _model, _device
    if _model is None:
        _guard_torchvision()                # ← prevent torchvision crash
        from sentence_transformers import SentenceTransformer
        from researchbuddy.config import EMBEDDING_MODEL
        _device = _resolve_device()
        print(f"[embedder] Loading '{EMBEDDING_MODEL}' on {_device} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=_device)
        print(f"[embedder] Model ready ({_device}).")
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
