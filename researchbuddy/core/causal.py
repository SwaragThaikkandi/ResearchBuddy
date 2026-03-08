"""
causal.py — Causal DAG construction for ResearchBuddy.

Converts the combined (cyclic) graph into an acyclic directed graph where
edges represent intellectual influence flow: older/influencing → newer/influenced.

Algorithm:
  1. Resolve the best-available publication year for every paper.
  2. Orient each edge using three signals (strongest first):
       a. Direct citation direction (cited paper → citing paper = influence)
       b. Publication year gap  (older → newer)
       c. Canonical fallback    (min(id) → max(id), low confidence)
  3. Break any remaining cycles by iteratively reversing the lowest-
     confidence edge in each detected cycle.
  4. Assert the result is a proper DAG.

The resulting G_causal is used by:
  - Reasoner: directed influence-chain tracing  (_trace_lineages)
  - Arguer : topological ordering of papers for evolution arguments
"""

from __future__ import annotations

import networkx as nx
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import PaperMeta


# ── Year resolution ──────────────────────────────────────────────────────────

def resolve_year(paper: "PaperMeta") -> Optional[int]:
    """
    Best-effort year for a paper.

    Priority: paper.year → extract from abstract metadata → None.
    """
    if paper.year and 1900 <= paper.year <= 2030:
        return paper.year

    # Reuse the abstract-year extractor from arguer
    from researchbuddy.core.arguer import _try_extract_year
    y = _try_extract_year(paper.abstract or "")
    if y:
        return y
    return None


# ── Edge orientation ─────────────────────────────────────────────────────────

def orient_edge(
    u: str,
    v: str,
    year_u: Optional[int],
    year_v: Optional[int],
    edge_data: dict,
    G_citation: nx.DiGraph,
) -> tuple[str, str, float]:
    """
    Determine causal direction and confidence for a single edge.

    Returns (source, target, causal_confidence) where source → target
    means "source influenced target".

    Signals used (strongest first):
      1. Direct citation: if A cites B, then B influenced A  → B→A
      2. Year gap: older paper influenced newer  → older→newer
      3. Fallback: canonical ID order, low confidence
    """
    # ── Signal 1: direct citation (strongest) ─────────────────────────
    # Check if one paper directly cites the other
    if G_citation.has_edge(u, v):
        etype = G_citation[u][v].get("etype", "")
        if etype == "citation":
            # u cites v → v influenced u → edge: v → u
            return v, u, 0.90
    if G_citation.has_edge(v, u):
        etype = G_citation[v][u].get("etype", "")
        if etype == "citation":
            # v cites u → u influenced v → edge: u → v
            return u, v, 0.90

    # ── Signal 2: publication year (strong) ───────────────────────────
    if year_u is not None and year_v is not None:
        if year_u < year_v:
            gap = year_v - year_u
            conf = min(0.95, 0.50 + gap * 0.05)
            return u, v, conf
        elif year_v < year_u:
            gap = year_u - year_v
            conf = min(0.95, 0.50 + gap * 0.05)
            return v, u, conf
        else:
            # Same year — use canonical order, moderate confidence
            src, tgt = (min(u, v), max(u, v))
            return src, tgt, 0.35

    # One year known, one missing
    if year_u is not None and year_v is None:
        return u, v, 0.40          # known-year paper is the influencer
    if year_v is not None and year_u is None:
        return v, u, 0.40

    # ── Signal 3: fallback — both years unknown ──────────────────────
    src, tgt = (min(u, v), max(u, v))
    return src, tgt, 0.25


# ── Cycle breaking ───────────────────────────────────────────────────────────

def break_cycles(G: nx.DiGraph) -> int:
    """
    Mutate *G* in-place until it is acyclic.

    Strategy: iteratively find a cycle, reverse the edge with the lowest
    ``causal_confidence``.  Reversing (rather than removing) preserves
    connectivity while fixing direction.

    Returns the number of edges reversed.
    """
    n_reversed = 0
    max_iter = G.number_of_edges() + 1      # safety bound

    for _ in range(max_iter):
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break

        # Find the weakest edge in this cycle
        worst_conf = float("inf")
        worst_edge: Optional[tuple[str, str]] = None
        for u, v, _dir in cycle:
            conf = G[u][v].get("causal_confidence", 0.5)
            if conf < worst_conf:
                worst_conf = conf
                worst_edge = (u, v)

        if worst_edge is None:
            break                            # shouldn't happen

        u, v = worst_edge
        data = dict(G[u][v])
        G.remove_edge(u, v)
        # Reverse with halved confidence and a flag
        data["causal_confidence"] = max(0.10, data.get("causal_confidence", 0.5) * 0.5)
        data["was_reversed"] = True
        G.add_edge(v, u, **data)
        n_reversed += 1

    return n_reversed


# ── Main DAG builder ─────────────────────────────────────────────────────────

def build_causal_dag(
    G_combined: nx.DiGraph,
    G_citation: nx.DiGraph,
    papers: dict[str, "PaperMeta"],
    min_confidence: float = 0.20,
) -> nx.DiGraph:
    """
    Build a causal DAG from the combined graph.

    Parameters
    ----------
    G_combined : nx.DiGraph
        The merged semantic + citation graph (may contain cycles).
    G_citation : nx.DiGraph
        The citation-only graph (used for direct-citation signal).
    papers : dict[str, PaperMeta]
        Paper metadata (needed for year resolution).
    min_confidence : float
        Edges with causal_confidence below this threshold are dropped.

    Returns
    -------
    nx.DiGraph
        An acyclic directed graph with ``causal_confidence`` on every edge.
    """
    paper_ids = set(papers.keys())

    # ── 1. Resolve years ──────────────────────────────────────────────
    year_cache: dict[str, Optional[int]] = {}
    for pid, meta in papers.items():
        year_cache[pid] = resolve_year(meta)

    # ── 2. Build the oriented graph ──────────────────────────────────
    dag = nx.DiGraph()

    # Add paper-only nodes (skip cluster / hierarchy nodes)
    for pid in paper_ids:
        if G_combined.has_node(pid):
            dag.add_node(pid, **G_combined.nodes[pid])

    # Orient each edge that connects two papers
    for u, v, data in G_combined.edges(data=True):
        if u not in paper_ids or v not in paper_ids:
            continue                         # skip cluster edges

        src, tgt, conf = orient_edge(
            u, v,
            year_cache.get(u), year_cache.get(v),
            data,
            G_citation,
        )

        if conf < min_confidence:
            continue                         # too uncertain

        # Keep the higher-confidence edge if one already exists
        if dag.has_edge(src, tgt):
            existing = dag[src][tgt].get("causal_confidence", 0.0)
            if conf <= existing:
                continue

        edge_attrs = dict(data)
        edge_attrs["causal_confidence"] = conf
        edge_attrs.pop("was_reversed", None)
        dag.add_edge(src, tgt, **edge_attrs)

    # ── 3. Break remaining cycles ────────────────────────────────────
    n_reversed = break_cycles(dag)
    if n_reversed:
        print(f"[causal] Reversed {n_reversed} edge(s) to eliminate cycles")

    # ── 4. Validate ──────────────────────────────────────────────────
    assert nx.is_directed_acyclic_graph(dag), (
        "build_causal_dag failed: result still has cycles"
    )

    return dag
