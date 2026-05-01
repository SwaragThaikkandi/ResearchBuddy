"""
embedder.py
Singleton wrapper around sentence-transformers.
Loaded once per process; all modules share the same instance.
"""

from __future__ import annotations

import gc
import importlib.machinery
import logging
import numpy as np
import sys
import types
from typing import Union

logger = logging.getLogger(__name__)

_model = None
_device = None
_force_cpu = False


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

    if _force_cpu:
        return "cpu"

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


def _needs_trust_remote_code(model_name: str) -> bool:
    return "nomic" in model_name.lower()


def _doc_prefix(model_name: str) -> str:
    """
    Some models (nomic-embed-text) use task prefixes for best performance.
    All embeddings in ResearchBuddy are document-to-document similarity,
    so we always use the document prefix.
    """
    if "nomic" in model_name.lower():
        return "search_document: "
    return ""


_auto_batch_size: int | None = None  # cached after first call to _get_model()
_auto_max_seq:    int | None = None
_auto_precision:  str | None = None  # one of "fp16", "fp32", "bf16"


def _vram_gb_for_device(device_index: int = 0) -> float:
    """Return total VRAM in GB for the given CUDA device, or 0.0 on failure."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(device_index)
        return float(props.total_memory) / (1024 ** 3)
    except Exception:
        return 0.0


def _vram_tier(vram_gb: float) -> str:
    """
    Map VRAM to a tier name. Tiers determine the auto defaults.
    Conservative cutoffs to keep room for OS, browser, other GPU users.
    """
    if vram_gb <= 0.0:
        return "cpu"
    if vram_gb < 5.0:        # 4GB cards (3050 4GB, 1650, etc.)
        return "tiny"
    if vram_gb < 9.0:        # 6-8GB cards (3050 6GB, 3060/4060 8GB, 2070, etc.)
        return "small"
    if vram_gb < 16.0:       # 12GB cards (3060 12GB, 4070 12GB)
        return "medium"
    return "large"           # 16GB+ cards


_TIER_DEFAULTS: dict[str, dict] = {
    # batch_size, max_seq_length (0 = no cap), precision
    "tiny":   {"batch_size": 2, "max_seq": 512,  "precision": "fp16"},
    "small":  {"batch_size": 4, "max_seq": 1024, "precision": "fp16"},
    "medium": {"batch_size": 8, "max_seq": 2048, "precision": "fp16"},
    "large":  {"batch_size": 8, "max_seq": 0,    "precision": "fp32"},
    "cpu":    {"batch_size": 4, "max_seq": 512,  "precision": "fp32"},
}


def _resolve_auto_settings(device: str) -> dict:
    """
    Pick batch_size / max_seq_length / precision from VRAM tier,
    honoring explicit env-var overrides where given.
    """
    from researchbuddy.config import (
        EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_SEQ_LENGTH, EMBEDDING_PRECISION,
    )

    if device == "cuda":
        vram = _vram_gb_for_device(0)
        tier = _vram_tier(vram)
    else:
        vram = 0.0
        tier = "cpu"

    defaults = _TIER_DEFAULTS[tier]

    # Env overrides win when explicitly set (>0 / non-auto)
    batch_size = int(EMBEDDING_BATCH_SIZE) if EMBEDDING_BATCH_SIZE > 0 else defaults["batch_size"]
    max_seq    = int(EMBEDDING_MAX_SEQ_LENGTH) if EMBEDDING_MAX_SEQ_LENGTH > 0 else defaults["max_seq"]
    precision  = EMBEDDING_PRECISION if EMBEDDING_PRECISION in ("fp16", "fp32", "bf16") else defaults["precision"]

    # fp16/bf16 only ever applied on CUDA
    if device != "cuda":
        precision = "fp32"

    return {
        "vram_gb": vram, "tier": tier,
        "batch_size": batch_size, "max_seq": max_seq, "precision": precision,
    }


def _parse_batch_size() -> int:
    """Return the resolved auto / overridden batch size (>=1)."""
    global _auto_batch_size
    if _auto_batch_size is not None:
        return max(1, _auto_batch_size)
    from researchbuddy.config import EMBEDDING_BATCH_SIZE
    return max(1, int(EMBEDDING_BATCH_SIZE) if EMBEDDING_BATCH_SIZE > 0 else 4)


def _parse_cpu_oom_fallback() -> bool:
    from researchbuddy.config import EMBEDDING_CPU_FALLBACK_ON_OOM

    return bool(EMBEDDING_CPU_FALLBACK_ON_OOM)


def _batch_sizes_to_try(start: int) -> list[int]:
    """
    Build a descending batch-size ladder ending in 1.
    Example: 8 -> [8, 4, 2, 1]
    """
    start = max(1, int(start))
    sizes: list[int] = []
    seen: set[int] = set()
    cur = start
    while True:
        if cur not in seen:
            sizes.append(cur)
            seen.add(cur)
        if cur == 1:
            break
        cur = max(1, cur // 2)
    return sizes


def _is_cuda_oom_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "outofmemory" in text:
        return True
    return (
        ("out of memory" in text or "cudaerrormemoryallocation" in text)
        and ("cuda" in text or "accelerator" in text)
    )


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _apply_runtime_embedding_limits(model, max_len: int) -> None:
    """Cap model.max_seq_length when the auto-tuner / config asks us to."""
    if max_len <= 0:
        return
    if not hasattr(model, "max_seq_length"):
        return
    try:
        prev = getattr(model, "max_seq_length", None)
        if prev is None or int(prev) > max_len:
            model.max_seq_length = max_len
            logger.info("Embedding max_seq_length set to %d (was %s).", max_len, prev)
    except Exception as e:
        logger.debug("Could not apply embedding max_seq_length=%s: %s", max_len, e)


def _apply_precision(model, precision: str) -> str:
    """
    Cast the model to fp16 / bf16 / fp32 in place. Returns the precision
    actually applied (in case a fallback was needed).
    """
    if precision == "fp32":
        return "fp32"
    try:
        import torch
        if precision == "fp16":
            model.half()
            return "fp16"
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            # sentence-transformers doesn't expose .bfloat16() directly;
            # walk submodules
            for m in model.modules():
                try:
                    m.to(dtype=torch.bfloat16)
                except Exception:
                    pass
            return "bf16"
    except Exception as e:
        logger.warning("Could not cast model to %s (%s); staying fp32.", precision, e)
    return "fp32"


def _encode_with_backoff(model, texts: list[str]) -> np.ndarray:
    """
    Encode with adaptive batch-size backoff on CUDA OOM.
    If OOM persists at batch_size=1, optionally move model to CPU and retry.
    """
    global _device, _force_cpu

    base_batch = _parse_batch_size()
    cpu_fallback = _parse_cpu_oom_fallback()
    last_oom: Exception | None = None

    for batch_size in _batch_sizes_to_try(base_batch):
        try:
            return model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            if _device == "cuda" and _is_cuda_oom_error(e):
                last_oom = e
                logger.warning(
                    "CUDA OOM during embedding (batch_size=%d, n_texts=%d).",
                    batch_size, len(texts),
                )
                _clear_cuda_cache()
                continue
            raise

    if _device == "cuda" and last_oom is not None and cpu_fallback:
        logger.warning("Falling back to CPU embeddings after repeated CUDA OOM.")
        try:
            model.cpu()
            _clear_cuda_cache()
            gc.collect()
            _device = "cpu"
            _force_cpu = True
            return model.encode(
                texts,
                batch_size=base_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            # Re-raise the original OOM context when CPU fallback fails too.
            raise last_oom

    if last_oom is not None:
        raise last_oom
    raise RuntimeError("Embedding failed for unknown reasons.")


def _get_model():
    global _model, _device, _auto_batch_size, _auto_max_seq, _auto_precision
    if _model is None:
        _guard_torchvision()  # prevent torchvision crash
        from sentence_transformers import SentenceTransformer
        from researchbuddy.config import EMBEDDING_MODEL

        _device = _resolve_device()
        logger.info("Loading '%s' on %s ...", EMBEDDING_MODEL, _device)

        kwargs = {"device": _device}
        if _needs_trust_remote_code(EMBEDDING_MODEL):
            kwargs["trust_remote_code"] = True

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
        except Exception as e:
            if _device == "cuda":
                logger.warning("CUDA model load failed: %s", e)
                logger.info("Retrying on CPU.")
                _device = "cpu"
                kwargs["device"] = "cpu"
                _model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
            else:
                raise

        # Resolve VRAM-aware auto settings (or honor explicit env overrides)
        settings = _resolve_auto_settings(_device)
        _auto_batch_size = settings["batch_size"]
        _auto_max_seq    = settings["max_seq"]

        applied_precision = _apply_precision(_model, settings["precision"])
        _auto_precision = applied_precision

        _apply_runtime_embedding_limits(_model, settings["max_seq"])

        logger.info(
            "Embedder ready: device=%s tier=%s vram=%.1fGB batch=%d max_seq=%d precision=%s",
            _device, settings["tier"], settings["vram_gb"],
            _auto_batch_size, _auto_max_seq, applied_precision,
        )
    return _model


def embed(texts: Union[str, list[str]]) -> np.ndarray:
    """
    Encode one string or a list of strings.
    Prepends the model's document task prefix when required (e.g. nomic-embed-text).
    Returns shape (dim,) for a single string, (n, dim) for a list.
    """
    from researchbuddy.config import EMBEDDING_MODEL
    model = _get_model()
    prefix = _doc_prefix(EMBEDDING_MODEL)
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    if prefix:
        texts = [prefix + t for t in texts]
    vecs = _encode_with_backoff(model, texts)
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
