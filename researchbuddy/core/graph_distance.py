"""
Graph-theoretic distance & reliability measures.

These quantify *how similar* two research graphs are and *how reliable* a
merge between them is — using standard, citable measures so the answer is
defensible, not hand-wavy:

  - spectral_distance     : L2 between Laplacian eigenvalue spectra
                            (size-tolerant; no node correspondence needed)
  - deltacon_similarity   : DeltaCon0 affinity similarity in (0, 1]
                            (Koutra et al. 2013; needs a shared node ordering)
  - degree_ks             : Kolmogorov–Smirnov stat between degree distributions
  - jaccard               : overlap of two node-id sets
  - modularity            : Newman modularity of a graph's community structure
  - gromov_wasserstein    : optimal-transport distortion between two graphs,
                            using intra-graph distances ONLY → no shared labels,
                            so it aligns two researchers' landscapes privately.
                            Requires the optional POT library (`pip install pot`).

All functions take plain numpy adjacency / cost matrices or networkx graphs,
so they work on any backend and are trivial to unit-test on toy graphs.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ── Adjacency helpers ─────────────────────────────────────────────────────────


def adjacency_over_ids(G, ids: Sequence[str]) -> np.ndarray:
    """
    Dense, symmetric, weighted adjacency for `G` restricted to `ids`, in the
    given order. Missing nodes contribute zero rows/cols. Works on any
    networkx graph (directed or not); weights come from the 'weight' attr.
    """
    n = len(ids)
    idx = {pid: i for i, pid in enumerate(ids)}
    A = np.zeros((n, n), dtype=float)
    if G is None:
        return A
    for u, v, data in G.edges(data=True):
        if u in idx and v in idx:
            w = float(data.get("weight", 1.0) or 1.0)
            i, j = idx[u], idx[v]
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)   # symmetrise (undirected similarity)
    return A


def degree_sequence(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float).sum(axis=1)


# ── Spectral distance ─────────────────────────────────────────────────────────


def laplacian_spectrum(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    ev = np.linalg.eigvalsh(L)          # symmetric ⇒ real eigenvalues
    return np.sort(ev)[::-1]            # descending


def spectral_distance(A: np.ndarray, B: np.ndarray, k: Optional[int] = None) -> float:
    """
    L2 distance between the top-`k` Laplacian eigenvalues of A and B. Spectra
    of different sizes are zero-padded to align, so graphs need not match in
    node count and no node correspondence is required.
    """
    sa, sb = laplacian_spectrum(A), laplacian_spectrum(B)
    m = k or max(len(sa), len(sb))
    pa = np.zeros(m); pa[:min(m, len(sa))] = sa[:m]
    pb = np.zeros(m); pb[:min(m, len(sb))] = sb[:m]
    return float(np.linalg.norm(pa - pb))


# ── DeltaCon ──────────────────────────────────────────────────────────────────


def _fabp_affinity(A: np.ndarray) -> np.ndarray:
    """Fast Belief Propagation affinity S = (I + eps^2 D - eps A)^-1."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    deg = A.sum(axis=1)
    eps = 1.0 / (1.0 + float(deg.max())) if deg.size and deg.max() > 0 else 0.1
    D = np.diag(deg)
    M = np.eye(n) + (eps ** 2) * D - eps * A
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def deltacon_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    DeltaCon0 similarity in (0, 1]; 1.0 == identical. A and B must share a
    node ordering (same shape) — use adjacency_over_ids() with one shared id
    list to build them (e.g. over the DOI-intersection of two graphs).
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError("DeltaCon needs a shared node ordering (equal shapes).")
    if A.size == 0:
        return 1.0
    S1, S2 = _fabp_affinity(A), _fabp_affinity(B)
    # Matusita (root-Euclidean) distance between affinity matrices.
    root_ed = float(np.sqrt(np.sum((np.sqrt(np.abs(S1)) -
                                    np.sqrt(np.abs(S2))) ** 2)))
    return 1.0 / (1.0 + root_ed)


# ── Degree-distribution KS ────────────────────────────────────────────────────


def degree_ks(A: np.ndarray, B: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic between the two degree distributions."""
    da, db = degree_sequence(A), degree_sequence(B)
    if da.size == 0 or db.size == 0:
        return 1.0
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(da, db).statistic)
    except Exception:
        # Manual ECDF KS as a dependency-free fallback.
        grid = np.unique(np.concatenate([da, db]))
        fa = np.searchsorted(np.sort(da), grid, side="right") / da.size
        fb = np.searchsorted(np.sort(db), grid, side="right") / db.size
        return float(np.max(np.abs(fa - fb)))


# ── Set overlap ───────────────────────────────────────────────────────────────


def jaccard(a: set, b: set) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Modularity ────────────────────────────────────────────────────────────────


def modularity(G) -> Optional[float]:
    """Greedy-community Newman modularity of an undirected view of G."""
    try:
        import networkx as nx
        H = nx.Graph(G)
        if H.number_of_edges() == 0:
            return 0.0
        comms = nx.community.greedy_modularity_communities(H, weight="weight")
        return float(nx.community.modularity(H, comms, weight="weight"))
    except Exception as e:
        logger.debug("modularity failed: %s", e)
        return None


# ── Gromov–Wasserstein (optional, label-free alignment) ───────────────────────


def cost_from_embeddings(embs: np.ndarray) -> np.ndarray:
    """Intra-graph cost matrix = pairwise (1 - cosine) over unit-normed rows."""
    X = np.asarray(embs, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0))
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = X / norm
    return 1.0 - (U @ U.T)


def gw_available() -> bool:
    try:
        import ot  # noqa: F401
        return True
    except Exception:
        return False


def gromov_wasserstein(C1: np.ndarray, C2: np.ndarray,
                       p: Optional[np.ndarray] = None,
                       q: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    Gromov–Wasserstein discrepancy between two graphs described only by their
    intra-graph cost matrices C1 (n×n) and C2 (m×m). Returns
    {distortion, coupling} or None when POT is not installed.

    Because it uses *only* within-graph distances, it needs no shared node
    labels — the foundation for privately aligning two research landscapes.
    """
    C1 = np.asarray(C1, dtype=float); C2 = np.asarray(C2, dtype=float)
    n, m = C1.shape[0], C2.shape[0]
    if n == 0 or m == 0:
        return None
    try:
        import ot
    except Exception:
        logger.info("Gromov–Wasserstein needs the POT library "
                    "(pip install 'researchbuddy[social]').")
        return None
    pv = np.full(n, 1.0 / n) if p is None else np.asarray(p, float)
    qv = np.full(m, 1.0 / m) if q is None else np.asarray(q, float)
    T = ot.gromov.gromov_wasserstein(C1, C2, pv, qv, "square_loss")
    dist = ot.gromov.gromov_wasserstein2(C1, C2, pv, qv, "square_loss")
    return {"distortion": float(dist), "coupling": np.asarray(T)}
