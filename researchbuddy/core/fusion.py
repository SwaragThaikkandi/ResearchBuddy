"""
fusion.py
Similarity Network Fusion (SNF) — Wang et al., Nature Methods 2014.

Given two similarity matrices W1 (semantic) and W2 (citation), SNF iteratively
diffuses information between them, producing a fused matrix that captures both
semantic closeness and bibliographic relationship simultaneously.

This is more powerful than a simple weighted average because:
  - Each network "corrects" the other's noise via cross-diffusion
  - The iterative process amplifies consistent signals across both modalities
  - Converges to a stable representation in ~10-20 iterations

Algorithm
---------
1. Normalise W1, W2 using a KNN-kernel (keep only k nearest neighbours per row)
2. P1 ← row-normalise(W1),  P2 ← row-normalise(W2)
3. For t in 1..n_iter:
       P1 ← K1 @ P2 @ K1.T   (diffuse through W2's structure)
       P2 ← K2 @ P1 @ K2.T
       row-normalise each
4. Fused = alpha * P1 + (1-alpha) * P2  →  symmetrised  →  [0,1] scaled

Fallback
--------
If W2 has too few non-zero entries (< 5% of W1), we skip SNF and return
alpha * W1 directly. This avoids diffusion artefacts from sparse citation data.
"""

from __future__ import annotations

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row_normalise(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-stochastic normalisation (each row sums to 1)."""
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return W / row_sums


def _knn_kernel(W: np.ndarray, k: int) -> np.ndarray:
    """
    Sparse KNN-kernel: keep only the top-k entries per row, zero out the rest.
    Result is symmetrised and row-normalised.
    """
    n = W.shape[0]
    k = min(k, n - 1)
    K = np.zeros_like(W)
    for i in range(n):
        top_k_idx = np.argsort(W[i])[-k:]
        K[i, top_k_idx] = W[i, top_k_idx]
    K = (K + K.T) / 2.0       # symmetrise
    return _row_normalise(K)


def _min_max_scale(W: np.ndarray) -> np.ndarray:
    lo, hi = W.min(), W.max()
    if hi - lo < 1e-12:
        return W
    return (W - lo) / (hi - lo)


# ── Public API ─────────────────────────────────────────────────────────────────

def snf(
    W_sem: np.ndarray,
    W_cit: np.ndarray,
    alpha: float = 0.6,
    k: int = 10,
    n_iter: int = 15,
    sparse_fallback_frac: float = 0.05,
) -> np.ndarray:
    """
    Fuse a semantic similarity matrix and a citation similarity matrix using SNF.

    Parameters
    ----------
    W_sem  : (n, n) semantic cosine-similarity matrix, values in [0, 1]
    W_cit  : (n, n) citation coupling matrix, values in [0, 1]
    alpha  : weight of semantic network in final combination (0-1)
    k      : number of nearest neighbours in KNN kernel
    n_iter : diffusion iterations (10-20 is usually enough)
    sparse_fallback_frac : if W_cit has fewer than this fraction of non-zeros
                           compared to W_sem, fall back to simple weighting.

    Returns
    -------
    W_fused : (n, n) symmetric fused similarity matrix, values in [0, 1]
    """
    n = W_sem.shape[0]

    # Fallback: citation data is too sparse for meaningful diffusion
    nonzero_sem = np.count_nonzero(W_sem)
    nonzero_cit = np.count_nonzero(W_cit)
    if nonzero_sem == 0:
        return _min_max_scale(W_cit) if nonzero_cit else np.eye(n) * 0.5
    if nonzero_cit == 0 or nonzero_cit < nonzero_sem * sparse_fallback_frac:
        return _min_max_scale(W_sem)

    # Scale both to [0, 1]
    W1 = _min_max_scale(W_sem.copy())
    W2 = _min_max_scale(W_cit.copy())

    # Build KNN kernels (captures local neighbourhood structure)
    K1 = _knn_kernel(W1, k)
    K2 = _knn_kernel(W2, k)

    # Initialise diffusion matrices as row-normalised versions
    P1 = _row_normalise(W1)
    P2 = _row_normalise(W2)

    # Iterative cross-network diffusion
    for _ in range(n_iter):
        P1_new = K1 @ P2 @ K1.T
        P2_new = K2 @ P1 @ K2.T
        P1 = _row_normalise(P1_new)
        P2 = _row_normalise(P2_new)

    # Weighted combination
    W_fused = alpha * P1 + (1.0 - alpha) * P2
    # Symmetrise and scale
    W_fused = (W_fused + W_fused.T) / 2.0
    return _min_max_scale(W_fused)


def fuse_scores(
    semantic_score: float,
    citation_score: float,
    alpha: float = 0.6,
) -> float:
    """
    Lightweight per-paper score fusion (no matrix needed).
    Used at query time when we don't have a full fused matrix for a new candidate.
    """
    return alpha * semantic_score + (1.0 - alpha) * citation_score
