"""
hierarchy.py
Build a Hierarchical Small World Network (HSWN) from paper embeddings.

Levels
------
0  : Individual paper nodes  (leaves)
1  : Research niches          ~sqrt(n_papers) clusters
2  : Research areas           ~sqrt(n_niches) clusters
3+ : Broader domains          (optional, grows with corpus)

Small-world property
--------------------
- DENSE edges within a niche  (cosine sim >= threshold)
- SPARSE shortcut edges between niches (top-k inter-niche pairs above shortcut_threshold)
- Cluster nodes sit above paper nodes; edges are directed upward (child → parent)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class ClusterNode:
    node_id: str
    level: int                     # 1 = niche, 2 = area, …
    centroid: np.ndarray           # mean of member embeddings (unit norm)
    paper_ids: list[str]           # all papers in this subtree
    child_ids: list[str]           = field(default_factory=list)  # direct children


# ── Internal helpers ───────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cluster(emb: np.ndarray, ids: list[str], k: int, prefix: str, level: int
             ) -> tuple[list[ClusterNode], np.ndarray]:
    """Agglomerative (Ward) clustering → ClusterNode list + label array."""
    k = min(k, len(ids) - 1)
    k = max(k, 1)

    if k == 1 or len(ids) <= 2:
        centroid = _norm(emb.mean(axis=0))
        node = ClusterNode(f"{prefix}_0", level, centroid, list(ids))
        return [node], np.zeros(len(ids), dtype=int)

    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(emb)
    nodes = []
    for i in range(k):
        mask = labels == i
        if not mask.any():
            continue
        members = [ids[j] for j in range(len(ids)) if mask[j]]
        centroid = _norm(emb[mask].mean(axis=0))
        nodes.append(ClusterNode(f"{prefix}_{i}", level, centroid, members))
    return nodes, labels


def _pca_reduce(emb: np.ndarray, target_dim: int = 64) -> np.ndarray:
    n, d = emb.shape
    dim = min(target_dim, d, n - 1)
    if dim >= d:
        return emb
    return PCA(n_components=dim, random_state=42).fit_transform(emb)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_hswn(
    paper_ids: list[str],
    embeddings: list[np.ndarray],
    n_levels: int = 2,
    intra_threshold: float = 0.45,   # min sim for within-niche edges
    shortcut_threshold: float = 0.60, # min sim for cross-niche shortcut edges
    max_shortcuts_per_pair: int = 1,  # shortcuts added per niche-pair
) -> tuple[nx.DiGraph, dict[str, ClusterNode]]:
    """
    Build a Hierarchical Small World Network.

    Returns
    -------
    G        : nx.DiGraph with paper nodes + cluster nodes
    clusters : dict node_id → ClusterNode
    """
    n = len(paper_ids)
    G: nx.DiGraph = nx.DiGraph()
    clusters: dict[str, ClusterNode] = {}

    # Add leaf (paper) nodes
    for pid in paper_ids:
        G.add_node(pid, level=0, node_type="paper")

    if n < 3:
        return G, clusters

    emb_matrix = np.stack(embeddings)          # (n, dim)
    emb_reduced = _pca_reduce(emb_matrix, 64)  # for clustering stability

    # --- Paper→Paper intra edges (cosine sim between raw embeddings) ----------
    sim_matrix = emb_matrix @ emb_matrix.T     # (n, n) — already unit-normed
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= intra_threshold:
                G.add_edge(paper_ids[i], paper_ids[j], weight=s, etype="semantic")
                G.add_edge(paper_ids[j], paper_ids[i], weight=s, etype="semantic")

    # --- Build levels ---------------------------------------------------------
    cur_ids   = list(paper_ids)
    cur_emb   = emb_reduced
    cur_full  = emb_matrix   # full-dim centroids for similarity

    for level in range(1, n_levels + 1):
        k = max(2, int(np.sqrt(len(cur_ids))))
        if len(cur_ids) <= 2:
            break

        nodes, labels = _cluster(cur_emb, cur_ids, k, f"L{level}", level)
        if not nodes:
            break

        # Build id→node lookup for this level
        id_to_node: dict[str, ClusterNode] = {}
        for ni, node in enumerate(nodes):
            G.add_node(node.node_id, level=level, node_type="cluster")
            clusters[node.node_id] = node
            id_to_node[node.node_id] = node

        # Child → parent membership edges
        for child_id, label_idx in zip(cur_ids, labels):
            if label_idx < len(nodes):
                parent = nodes[label_idx]
                parent.child_ids.append(child_id)
                G.add_edge(child_id, parent.node_id, weight=1.0, etype="member")

        # Cluster↔Cluster similarity edges
        centroids = np.stack([nd.centroid for nd in nodes])
        csim = centroids @ centroids.T
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                s = float(csim[i, j])
                if s > 0.30:
                    G.add_edge(nodes[i].node_id, nodes[j].node_id, weight=s, etype="cluster_sim")
                    G.add_edge(nodes[j].node_id, nodes[i].node_id, weight=s, etype="cluster_sim")

        # Small-world shortcut edges between niches (cross-cluster high-sim pairs)
        if level == 1:
            _add_shortcuts(G, nodes, paper_ids, emb_matrix, shortcut_threshold, max_shortcuts_per_pair)

        # Prepare next level from cluster centroids
        cur_ids  = [nd.node_id for nd in nodes]
        cur_emb  = _pca_reduce(centroids, 32)
        cur_full = centroids

    return G, clusters


def _add_shortcuts(
    G: nx.DiGraph,
    niches: list[ClusterNode],
    paper_ids: list[str],
    emb_matrix: np.ndarray,
    threshold: float,
    max_per_pair: int,
):
    """Add small-world shortcut edges between the most similar cross-niche paper pairs."""
    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    for a in range(len(niches)):
        for b in range(a + 1, len(niches)):
            # Intersect paper_ids with emb_matrix index
            pids_a = [p for p in niches[a].paper_ids if p in pid_to_idx]
            pids_b = [p for p in niches[b].paper_ids if p in pid_to_idx]
            if not pids_a or not pids_b:
                continue

            idxs_a = [pid_to_idx[p] for p in pids_a]
            idxs_b = [pid_to_idx[p] for p in pids_b]
            cross = emb_matrix[idxs_a] @ emb_matrix[idxs_b].T  # (|a|, |b|)

            added = 0
            # Find best cross-niche pairs
            flat = [(float(cross[i, j]), pids_a[i], pids_b[j])
                    for i in range(len(pids_a)) for j in range(len(pids_b))]
            for sim, pa, pb in sorted(flat, reverse=True):
                if sim < threshold or added >= max_per_pair:
                    break
                G.add_edge(pa, pb, weight=sim * 0.8, etype="shortcut")
                G.add_edge(pb, pa, weight=sim * 0.8, etype="shortcut")
                added += 1


def compute_paper_level_map(
    clusters: dict[str, ClusterNode], paper_ids: list[str]
) -> dict[str, list[str]]:
    """
    Returns {paper_id: [niche_id, area_id, ...]} — the cluster path for each paper.
    """
    # Build child→parent map
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
