"""
hierarchy.py
Adaptive Hierarchical Small World Network (HSWN).

The algorithm automatically determines how many levels of hierarchy are
appropriate for the current corpus — no user-specified depth needed.

Algorithm
---------
1. Compute pairwise cosine distances between paper embeddings.
2. Build a Ward-linkage dendrogram (scipy).
3. Detect "phase transitions" in the merge-distance sequence using
   acceleration peaks (second-derivative) + relative-jump analysis.
   Each peak = a meaningful level boundary.
4. Cut the dendrogram at every detected boundary → one cluster level each.
5. Assign papers to clusters at each level; compute centroids.
6. Build a multi-level graph:
     Level 0   : individual papers (leaves)
     Level 1   : research niches   (~sqrt(n) clusters)
     Level 2   : research areas    (clusters of niches)
     Level 3+  : broader domains   (if data supports it)
   Dense intra-niche edges + sparse cross-niche shortcut edges (small-world).

References
----------
- Ward (1963) — hierarchical clustering criterion
- Watts & Strogatz (1998) — small-world networks
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class ClusterNode:
    node_id: str
    level: int                       # 1 = niche, 2 = area, 3 = domain, …
    centroid: np.ndarray             # mean embedding of all papers (unit norm)
    paper_ids: list[str]             # all papers in this subtree
    child_ids: list[str]             = field(default_factory=list)
    n_levels_detected: int           = 0  # stored at root level for info


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _pca_reduce(emb: np.ndarray, target: int = 64) -> np.ndarray:
    n, d = emb.shape
    dim  = min(target, d, n - 1)
    return PCA(n_components=dim, random_state=42).fit_transform(emb) if dim < d else emb


# ── Adaptive level detection ───────────────────────────────────────────────────

def _find_adaptive_cuts(
    Z: np.ndarray,
    n_papers: int,
    min_cluster_size: int,
    max_levels: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Detect significant hierarchy levels from a Ward dendrogram.

    Strategy: identify "phase transitions" — places where the merge distance
    jumps sharply, meaning qualitatively distinct groups are being merged.
    Two complementary signals are combined:
      1. Acceleration peaks (second derivative of merge distances)
      2. Relative jump ratio (merge_dist[i+1] / merge_dist[i])

    Returns list of (n_clusters, label_array) tuples, finest → coarsest.
    """
    merge_dists = Z[:, 2]
    n           = len(merge_dists)
    if n < 2:
        return []

    # ── Signal 1: acceleration (second derivative) ─────────────────────────
    accel           = np.diff(np.diff(merge_dists))          # length n-2
    accel_mean      = np.mean(accel)
    accel_std       = np.std(accel)
    accel_threshold = accel_mean + 0.8 * accel_std
    accel_peaks     = set((np.where(accel > accel_threshold)[0] + 1).tolist())

    # ── Signal 2: relative jump ────────────────────────────────────────────
    safe_dists  = merge_dists[:-1] + 1e-10
    rel_jumps   = np.diff(merge_dists) / safe_dists
    rel_thr     = np.percentile(rel_jumps, 75)
    rel_peaks   = set(np.where(rel_jumps > rel_thr)[0].tolist())

    combined_peaks = sorted(accel_peaks | rel_peaks)

    # ── Convert peaks → (k, labels) ───────────────────────────────────────
    cuts: list[tuple[int, np.ndarray, int]] = []
    for idx in combined_peaks:
        if idx >= n:
            continue
        cut_d  = merge_dists[idx]
        labels = fcluster(Z, cut_d, criterion="distance")
        k      = int(len(set(labels)))
        if min_cluster_size <= k <= (n_papers - min_cluster_size):
            cuts.append((k, labels.copy(), idx))

    # ── Fallback: always include at least k = sqrt(n) ────────────────────
    k_sqrt = max(2, int(np.sqrt(n_papers)))
    if not any(abs(c[0] - k_sqrt) < 2 for c in cuts) and n_papers >= 2 * min_cluster_size:
        lbl = fcluster(Z, k_sqrt, criterion="maxclust")
        k_a = int(len(set(lbl)))
        if k_a >= 2:
            cuts.append((k_a, lbl.copy(), -1))

    # ── Deduplicate and sort finest → coarsest ────────────────────────────
    cuts.sort(key=lambda x: -x[0])
    seen_k : set[int] = set()
    result : list[tuple[int, np.ndarray]] = []
    for k, labels, _ in cuts:
        if k not in seen_k:
            seen_k.add(k)
            result.append((k, labels))
            if len(result) >= max_levels:
                break

    return result


# ── Graph construction ─────────────────────────────────────────────────────────

def build_adaptive_hswn(
    paper_ids: list[str],
    embeddings: list[np.ndarray],
    min_cluster_size: int = 3,
    max_levels: int = 8,
    intra_threshold: float = 0.45,
    shortcut_threshold: float = 0.60,
) -> tuple[nx.DiGraph, dict[str, ClusterNode]]:
    """
    Build a Hierarchical Small World Network with adaptively chosen levels.

    Returns
    -------
    G        : nx.DiGraph  (paper nodes + cluster nodes + all edge types)
    clusters : dict  node_id → ClusterNode
    """
    n = len(paper_ids)
    G : nx.DiGraph              = nx.DiGraph()
    clusters: dict[str, ClusterNode] = {}

    # Add leaf paper nodes
    for pid in paper_ids:
        G.add_node(pid, level=0, node_type="paper")

    if n < 3:
        return G, clusters

    emb_matrix = np.stack(embeddings)                  # (n, dim) unit-normed
    emb_red    = _pca_reduce(emb_matrix, 64)           # for stable clustering

    # ── Paper–paper semantic edges ─────────────────────────────────────────
    sim_mat = emb_matrix @ emb_matrix.T                # cosine sim, already normed
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_mat[i, j])
            if s >= intra_threshold:
                G.add_edge(paper_ids[i], paper_ids[j], weight=s, etype="semantic")
                G.add_edge(paper_ids[j], paper_ids[i], weight=s, etype="semantic")

    # ── Build dendrogram ───────────────────────────────────────────────────
    dists = pdist(emb_red, metric="euclidean")          # Ward requires euclidean
    Z     = linkage(dists, method="ward")

    level_cuts = _find_adaptive_cuts(Z, n, min_cluster_size, max_levels)
    if not level_cuts:
        return G, clusters

    n_detected = len(level_cuts)

    # ── Create cluster nodes per level ────────────────────────────────────
    prev_level_map: dict[str, str] = {}   # paper_id → parent cluster_id at prev level

    for level_idx, (k, raw_labels) in enumerate(level_cuts):
        level_num = level_idx + 1         # 1-indexed

        # Map paper → cluster index at this level
        paper_to_cluster: dict[str, int] = {
            paper_ids[i]: int(raw_labels[i]) - 1   # fcluster uses 1-based
            for i in range(n)
        }

        # Group papers by cluster
        from collections import defaultdict
        groups: dict[int, list[str]] = defaultdict(list)
        for pid, cidx in paper_to_cluster.items():
            groups[cidx].append(pid)

        # Create ClusterNode for each group
        level_clusters: list[ClusterNode] = []
        idx_to_node: dict[int, ClusterNode] = {}
        for cidx, pids in groups.items():
            node_id  = f"L{level_num}C{cidx}"
            indices  = [paper_ids.index(p) for p in pids if p in paper_ids]
            centroid = _norm(emb_matrix[indices].mean(axis=0)) if indices else np.zeros(emb_matrix.shape[1])
            node     = ClusterNode(
                node_id  = node_id,
                level    = level_num,
                centroid = centroid,
                paper_ids= pids,
            )
            if level_idx == 0:
                node.n_levels_detected = n_detected
            clusters[node_id]   = node
            idx_to_node[cidx]   = node
            G.add_node(node_id, level=level_num, node_type="cluster")

            level_clusters.append(node)

        # ── Membership edges (child → parent) ─────────────────────────────
        for cidx, node in idx_to_node.items():
            for pid in node.paper_ids:
                if level_idx == 0:
                    # Paper → niche
                    G.add_edge(pid, node.node_id, weight=1.0, etype="member")
                else:
                    # Previous-level cluster → current cluster (if child)
                    parent_id = prev_level_map.get(pid)
                    if parent_id and parent_id not in node.child_ids:
                        node.child_ids.append(parent_id)
                        G.add_edge(parent_id, node.node_id, weight=1.0, etype="member")

        # ── Cluster–cluster similarity edges ──────────────────────────────
        centroids = np.stack([nd.centroid for nd in level_clusters])
        csim = centroids @ centroids.T
        for i, nd_i in enumerate(level_clusters):
            for j, nd_j in enumerate(level_clusters):
                if j <= i:
                    continue
                s = float(csim[i, j])
                if s > 0.25:
                    G.add_edge(nd_i.node_id, nd_j.node_id, weight=s, etype="cluster_sim")
                    G.add_edge(nd_j.node_id, nd_i.node_id, weight=s, etype="cluster_sim")

        # ── Small-world shortcut edges between niches (level 1 only) ─────
        if level_num == 1:
            pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
            for a_idx, nd_a in enumerate(level_clusters):
                for b_idx, nd_b in enumerate(level_clusters):
                    if b_idx <= a_idx:
                        continue
                    pids_a = [p for p in nd_a.paper_ids if p in pid_to_idx]
                    pids_b = [p for p in nd_b.paper_ids if p in pid_to_idx]
                    if not pids_a or not pids_b:
                        continue
                    ia  = [pid_to_idx[p] for p in pids_a]
                    ib  = [pid_to_idx[p] for p in pids_b]
                    cross = emb_matrix[ia] @ emb_matrix[ib].T
                    ri, ci = np.unravel_index(np.argmax(cross), cross.shape)
                    best_sim = float(cross[ri, ci])
                    if best_sim >= shortcut_threshold:
                        pa, pb = pids_a[ri], pids_b[ci]
                        G.add_edge(pa, pb, weight=best_sim * 0.8, etype="shortcut")
                        G.add_edge(pb, pa, weight=best_sim * 0.8, etype="shortcut")

        # Update prev_level_map for next iteration
        prev_level_map = {pid: idx_to_node[paper_to_cluster[pid]].node_id
                          for pid in paper_ids if pid in paper_to_cluster}

    return G, clusters


# ── Utilities ──────────────────────────────────────────────────────────────────

def compute_paper_level_map(
    clusters: dict[str, ClusterNode],
    paper_ids: list[str],
) -> dict[str, list[str]]:
    """Return {paper_id: [niche_id, area_id, …]} — ancestry path for each paper."""
    child_to_parent: dict[str, str] = {}
    for nid, node in clusters.items():
        for child in node.child_ids:
            child_to_parent[child] = nid

    result = {}
    for pid in paper_ids:
        path, cur = [], pid
        while cur in child_to_parent:
            cur = child_to_parent[cur]
            path.append(cur)
        result[pid] = path
    return result


def n_levels_detected(clusters: dict[str, ClusterNode]) -> int:
    """Return the maximum hierarchy depth in the current cluster set."""
    if not clusters:
        return 0
    return max(nd.level for nd in clusters.values())
