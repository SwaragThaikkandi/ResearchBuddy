"""
embedder.py
Singleton wrapper around sentence-transformers.
Loaded once per process; all modules share the same instance.
"""

from __future__ import annotations

import importlib.machinery
import logging
import numpy as np
import sys
import types
from typing import Union

logger = logging.getLogger(__name__)

_model = None
_device = None


def _guard_torchvision():
    """
    Prevent a broken torchvision install from crashing sentence-transformers.

    sentence_transformers -> transformers -> torchvision.transforms (for vision
    models we never use). If torch and torchvision versions are mismatched,
    imports can fail during module initialization.

    If torchvision fails to import, inject a minimal stub so downstream
    imports that expect torchvision.transforms.InterpolationMode succeed.
    Text embeddings are unaffected.
    """
    existing = sys.modules.get("torchvision")
    if existing is not None:
        # If torchvision loaded cleanly (or we already injected a stub), keep it.
        if hasattr(existing, "transforms"):
            return
        # Failed imports can leave partially initialized modules behind.
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)

    try:
        import torchvision  # noqa: F401 - test if it loads
    except (Exception, ValueError):
        # Build a minimal stub.
        tv = types.ModuleType("torchvision")
        tv.__path__ = []  # mark as package
        # importlib.util.find_spec() crashes with ValueError when __spec__
        # is None.  transformers.utils.import_utils calls find_spec() during
        # is_torchvision_available() — setting a proper ModuleSpec avoids this.
        tv.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision", None, is_package=True,
        )

        transforms = types.ModuleType("torchvision.transforms")
        transforms.InterpolationMode = type(
            "InterpolationMode",
            (),
            {
                "BILINEAR": 2,
                "NEAREST": 0,
                "BICUBIC": 3,
                "LANCZOS": 1,
                "BOX": 4,
                "HAMMING": 5,
            },
        )()
        tv.transforms = transforms

        functional = types.ModuleType("torchvision.transforms.functional")
        tv.transforms.functional = functional

        # Set __spec__ on every sub-module so importlib.util.find_spec()
        # never encounters __spec__ = None.
        for mod_name, mod_obj in [
            ("torchvision", tv),
            ("torchvision.transforms", transforms),
            ("torchvision.transforms.functional", functional),
        ]:
            if mod_obj.__spec__ is None:
                mod_obj.__spec__ = importlib.machinery.ModuleSpec(
                    mod_name, None, is_package="." not in mod_name,
                )

        meta_reg = types.ModuleType("torchvision._meta_registrations")
        meta_reg.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision._meta_registrations", None,
        )

        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.functional"] = functional
        sys.modules["torchvision._meta_registrations"] = meta_reg

        logger.info(
            "torchvision is broken or missing; "
            "stubbed out (not needed for text embeddings)."
        )
        logger.info("To fix permanently: pip install --upgrade torch torchvision")


def _cuda_status() -> tuple[bool, str]:
    """
    Return (usable, reason_or_device_name) for CUDA.
    Never raises: broken torch installs are reported as text.
    """
    try:
        import torch
    except Exception as e:
        return False, f"PyTorch import failed ({e})"

    try:
        cuda_build = getattr(torch.version, "cuda", None)
        if not cuda_build:
            return False, "PyTorch is CPU-only (no CUDA build)"

        if not torch.cuda.is_available():
            return False, f"CUDA not available at runtime (torch CUDA build: {cuda_build})"

        name = torch.cuda.get_device_name(0)
        return True, name
    except Exception as e:
        return False, f"CUDA probe failed ({e})"


def _resolve_device() -> str:
    """Pick the best device for embedding: config override -> CUDA -> CPU."""
    from researchbuddy.config import EMBEDDING_DEVICE

    if EMBEDDING_DEVICE == "cpu":
        return "cpu"

    cuda_ok, detail = _cuda_status()

    if EMBEDDING_DEVICE == "cuda":
        if cuda_ok:
            return "cuda"
        logger.warning("CUDA requested but unavailable: %s", detail)
        logger.info("Falling back to CPU.")
        return "cpu"

    # "auto" - try CUDA first
    if cuda_ok:
        return "cuda"

    logger.info("CUDA not usable (%s); using CPU.", detail)
    return "cpu"


def _get_model():
    global _model, _device
    if _model is None:
        _guard_torchvision()  # prevent torchvision crash
        from sentence_transformers import SentenceTransformer
        from researchbuddy.config import EMBEDDING_MODEL

        _device = _resolve_device()
        logger.info("Loading '%s' on %s ...", EMBEDDING_MODEL, _device)

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL, device=_device)
        except Exception as e:
            if _device == "cuda":
                logger.warning("CUDA model load failed: %s", e)
                logger.info("Retrying on CPU.")
                _device = "cpu"
                _model = SentenceTransformer(EMBEDDING_MODEL, device=_device)
            else:
                raise

        logger.info("Model ready (%s).", _device)
    return _model


def embed(texts: Union[str, list[str]]) -> np.ndarray:
    """
    Encode one string or a list of strings.
    Returns shape (dim,) for a single string, (n, dim) for a list.
    """
    model = _get_model()
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs[0] if single else vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors (already L2-normalized)."""
    return float(np.dot(a, b))


def mean_pool(vecs: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of embedding vectors, then L2-normalized."""
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
