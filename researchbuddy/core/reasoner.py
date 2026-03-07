"""
reasoner.py  —  "Prefrontal Cortex" reasoning engine.

Reasons over the graph STRUCTURE — not just embeddings — to provide
network-aware insights.  Uses centrality, citation paths, cluster
topology, and temporal analysis to help users understand their research
landscape.

Key difference from *search*:
  Search  → fetches NEW papers from external APIs.
  Reasoner → analyses the EXISTING graph structure.

Feedback (1-10 ratings) actively reshapes the graph by strengthening /
creating edges between papers the user links through their queries.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from researchbuddy.core.embedder import embed, cosine_similarity

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
    from researchbuddy.core.hierarchy  import ClusterNode


# ── Stored interaction (persisted inside the graph pickle) ────────────────────

@dataclass
class QueryInteraction:
    """One user query + feedback event."""
    query_embedding: np.ndarray
    paper_ids: list[str]          # papers shown in the response
    rating: float                 # 1-10 (0 = unrated)
    timestamp: float = 0.0


# ── Auxiliary dataclasses ─────────────────────────────────────────────────────

@dataclass
class ClusterProfile:
    """Rich analysis of a single niche cluster."""
    cluster: "ClusterNode"
    similarity: float            # cosine sim of cluster centroid to query
    n_papers: int
    density: float               # fraction of possible internal edges present
    avg_year: Optional[float]
    central_paper: Optional["PaperMeta"]   # most-connected paper in cluster
    maturity: str                # "emerging" / "growing" / "established"


@dataclass
class ResearchLineage:
    """A directed path through the citation / semantic graph."""
    path: list[str]              # paper_ids in order
    path_type: str               # "citation_chain" or "semantic_path"


# ── Query result ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    query_embedding: np.ndarray

    # Top papers — ranked by blended similarity + centrality
    relevant_papers: list       # [(PaperMeta, score, info_dict)]
    #   info_dict keys: centrality, role, degree

    # Cluster profiles (density, maturity, central paper)
    cluster_profiles: list      # [ClusterProfile]

    # Graph-edge connections among the relevant papers
    connections: list           # [(paper_id_a, paper_id_b, description)]

    # Citation / semantic paths between relevant papers
    lineages: list              # [ResearchLineage]

    # Papers bridging multiple relevant niches
    bridge_papers: list         # [PaperMeta]

    # Frontier: similar to query but few graph connections (underexplored)
    frontier_papers: list       # [(PaperMeta, similarity)]

    # Temporal evolution narrative
    temporal_narrative: str

    # Coverage gap note
    gap_note: str


# ── Reasoner ──────────────────────────────────────────────────────────────────

class Reasoner:
    """
    Graph-structure-aware reasoning engine.

    Unlike the simple similarity search, the Reasoner:
      * Boosts papers by PageRank centrality (not just embedding distance)
      * Profiles clusters (density, maturity, central paper)
      * Traces citation lineages between relevant papers
      * Detects frontier papers (similar but underconnected)
      * Builds temporal narratives
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    # ── Main entry point ──────────────────────────────────────────────────

    def reason(self, query: str, graph: "HierarchicalResearchGraph") -> QueryResult:
        query_emb = embed(query)

        papers   = self._find_relevant_papers(query_emb, graph)
        clusters = self._profile_clusters(query_emb, papers, graph)
        conns    = self._find_connections(papers, graph)
        lineages = self._trace_lineages(papers, graph)
        bridges  = self._find_bridges(papers, clusters, graph)
        frontier = self._find_frontier(query_emb, papers, graph)
        timeline = self._temporal_narrative(papers, graph)
        gap      = self._assess_coverage(query_emb, papers)

        return QueryResult(
            query=query,
            query_embedding=query_emb,
            relevant_papers=papers,
            cluster_profiles=clusters,
            connections=conns,
            lineages=lineages,
            bridge_papers=bridges,
            frontier_papers=frontier,
            temporal_narrative=timeline,
            gap_note=gap,
        )

    # ── 1. Paper retrieval (similarity + centrality) ─────────────────────

    def _find_relevant_papers(
        self,
        query_emb: np.ndarray,
        graph: "HierarchicalResearchGraph",
    ) -> list[tuple["PaperMeta", float, dict]]:
        # PageRank on paper-only subgraph of the combined graph
        paper_ids_set = {m.paper_id for m in graph.all_papers()}
        sub = graph.G.subgraph(
            [n for n in graph.G.nodes if n in paper_ids_set]
        )
        try:
            pr = nx.pagerank(sub, alpha=0.85, max_iter=100)
        except Exception:
            pr = {}

        scored: list[tuple["PaperMeta", float, dict]] = []
        for meta in graph.all_papers():
            if meta.embedding is None:
                continue
            sim = float(cosine_similarity(query_emb, meta.embedding))
            centrality = pr.get(meta.paper_id, 0.0)
            degree = sub.degree(meta.paper_id) if sub.has_node(meta.paper_id) else 0

            # Paper role in the network
            if degree >= 5:
                role = "hub"
            elif degree == 0:
                role = "isolated"
            else:
                role = "connected"

            scored.append((meta, sim, {
                "centrality": centrality,
                "role": role,
                "degree": degree,
            }))

        if not scored:
            return []

        # Normalise centrality to [0,1] and blend with similarity
        max_c = max((info["centrality"] for _, _, info in scored), default=1.0) or 1e-9
        for i, (meta, sim, info) in enumerate(scored):
            norm_c = info["centrality"] / max_c
            blended = 0.80 * sim + 0.20 * norm_c
            scored[i] = (meta, blended, info)

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]

    # ── 2. Cluster profiling ─────────────────────────────────────────────

    def _profile_clusters(
        self,
        query_emb: np.ndarray,
        papers: list[tuple["PaperMeta", float, dict]],
        graph: "HierarchicalResearchGraph",
    ) -> list[ClusterProfile]:
        profiles: list[ClusterProfile] = []

        for cluster in graph._clusters.values():
            if cluster.level != 1:          # focus on niches
                continue
            sim = float(cosine_similarity(query_emb, cluster.centroid))
            if sim < 0.30:
                continue

            n_papers = len(cluster.paper_ids)

            # Edge density inside this cluster
            density = 0.0
            if n_papers >= 2:
                actual = sum(
                    1 for a in cluster.paper_ids for b in cluster.paper_ids
                    if a != b and graph.G.has_edge(a, b)
                )
                possible = n_papers * (n_papers - 1)
                density = actual / possible if possible else 0.0

            # Average publication year
            years = [
                graph.get_paper(pid).year
                for pid in cluster.paper_ids
                if graph.get_paper(pid) and graph.get_paper(pid).year
            ]
            avg_year = float(np.mean(years)) if years else None

            # Most-connected paper inside the cluster
            central_paper = None
            best_deg = -1
            for pid in cluster.paper_ids:
                deg = graph.G.degree(pid) if graph.G.has_node(pid) else 0
                if deg > best_deg:
                    best_deg = deg
                    central_paper = graph.get_paper(pid)

            # Maturity bucket
            if avg_year is None:
                maturity = "unknown"
            elif avg_year >= 2020:
                maturity = "emerging"
            elif avg_year >= 2010:
                maturity = "growing"
            else:
                maturity = "established"

            profiles.append(ClusterProfile(
                cluster=cluster,
                similarity=sim,
                n_papers=n_papers,
                density=density,
                avg_year=avg_year,
                central_paper=central_paper,
                maturity=maturity,
            ))

        profiles.sort(key=lambda p: p.similarity, reverse=True)
        return profiles[:4]

    # ── 3. Connections via actual graph edges ─────────────────────────────

    def _find_connections(
        self,
        papers: list[tuple["PaperMeta", float, dict]],
        graph: "HierarchicalResearchGraph",
    ) -> list[tuple[str, str, str]]:
        p2n = graph.paper_to_niche()
        seen: set[tuple[str, str, str]] = set()
        connections: list[tuple[str, str, str]] = []

        paper_list = [m for m, _, _ in papers]
        for i, meta_a in enumerate(paper_list):
            pid_a = meta_a.paper_id
            refs_a = graph._refs.get(pid_a, set())
            niche_a = p2n.get(pid_a)

            for meta_b in paper_list[i + 1:]:
                pid_b = meta_b.paper_id
                key_base = (min(pid_a, pid_b), max(pid_a, pid_b))

                # Direct graph edge (check which networks contain it)
                has_cit = (graph.G_citation.has_edge(pid_a, pid_b)
                           or graph.G_citation.has_edge(pid_b, pid_a))
                has_sem = (graph.G_semantic.has_edge(pid_a, pid_b)
                           or graph.G_semantic.has_edge(pid_b, pid_a))
                if has_cit and has_sem:
                    desc = "linked by citation + semantic similarity"
                elif has_cit:
                    desc = "citation link"
                elif has_sem:
                    desc = "semantic link"
                else:
                    desc = None
                if desc:
                    key = (*key_base, "edge")
                    if key not in seen:
                        seen.add(key)
                        connections.append((pid_a, pid_b, desc))

                # Bibliographic coupling
                refs_b = graph._refs.get(pid_b, set())
                if refs_a and refs_b:
                    shared = refs_a & refs_b
                    if shared:
                        key = (*key_base, "bibcoupling")
                        if key not in seen:
                            seen.add(key)
                            connections.append((
                                pid_a, pid_b,
                                f"share {len(shared)} reference(s)",
                            ))

                # Same niche
                niche_b = p2n.get(pid_b)
                if niche_a and niche_b and niche_a == niche_b:
                    key = (*key_base, "niche")
                    if key not in seen:
                        seen.add(key)
                        connections.append((pid_a, pid_b, f"same niche ({niche_a})"))

        return connections

    # ── 4. Research lineages (citation / semantic paths) ──────────────────

    def _trace_lineages(
        self,
        papers: list[tuple["PaperMeta", float, dict]],
        graph: "HierarchicalResearchGraph",
    ) -> list[ResearchLineage]:
        lineages: list[ResearchLineage] = []
        top_ids = [m.paper_id for m, _, _ in papers[:6]]

        # Try citation graph first (directed intellectual lineage)
        G_cit = graph.G_citation
        for i, pid_a in enumerate(top_ids):
            for pid_b in top_ids[i + 1:]:
                if not G_cit.has_node(pid_a) or not G_cit.has_node(pid_b):
                    continue
                for src, tgt in [(pid_a, pid_b), (pid_b, pid_a)]:
                    try:
                        path = nx.shortest_path(G_cit, src, tgt)
                        if 2 <= len(path) <= 5:
                            lineages.append(ResearchLineage(
                                path=path, path_type="citation_chain",
                            ))
                    except nx.NetworkXNoPath:
                        pass

        # Fall back to combined graph for semantic paths
        if not lineages:
            G_all = graph.G
            for i, pid_a in enumerate(top_ids[:4]):
                for pid_b in top_ids[i + 1: 4]:
                    if not G_all.has_node(pid_a) or not G_all.has_node(pid_b):
                        continue
                    try:
                        path = nx.shortest_path(G_all, pid_a, pid_b)
                        # Keep only paper nodes (skip cluster nodes)
                        paper_path = [p for p in path if graph.get_paper(p)]
                        if len(paper_path) >= 2:
                            lineages.append(ResearchLineage(
                                path=paper_path, path_type="semantic_path",
                            ))
                    except nx.NetworkXNoPath:
                        pass

        return lineages[:3]

    # ── 5. Bridge papers ──────────────────────────────────────────────────

    def _find_bridges(
        self,
        papers: list[tuple["PaperMeta", float, dict]],
        clusters: list[ClusterProfile],
        graph: "HierarchicalResearchGraph",
    ) -> list["PaperMeta"]:
        relevant_niches = [cp.cluster for cp in clusters][:3]
        if len(relevant_niches) < 2:
            return []

        shown_ids = {m.paper_id for m, _, _ in papers}
        bridges: list[tuple["PaperMeta", float]] = []
        for meta in graph.all_papers():
            if meta.embedding is None:
                continue
            sims = [
                float(cosine_similarity(meta.embedding, c.centroid))
                for c in relevant_niches
            ]
            min_sim = min(sims)
            if min_sim > 0.40:
                bridges.append((meta, min_sim))

        bridges.sort(key=lambda x: x[1], reverse=True)
        out: list["PaperMeta"] = []
        for meta, _ in bridges:
            if meta.paper_id not in shown_ids:
                out.append(meta)
            if len(out) >= 3:
                break
        return out

    # ── 6. Frontier papers (relevant but underconnected) ──────────────────

    def _find_frontier(
        self,
        query_emb: np.ndarray,
        papers: list[tuple["PaperMeta", float, dict]],
        graph: "HierarchicalResearchGraph",
    ) -> list[tuple["PaperMeta", float]]:
        shown_ids = {m.paper_id for m, _, _ in papers}
        frontier: list[tuple["PaperMeta", float]] = []

        for meta in graph.all_papers():
            if meta.embedding is None or meta.paper_id in shown_ids:
                continue
            sim = float(cosine_similarity(query_emb, meta.embedding))
            if sim < 0.40:
                continue
            degree = graph.G.degree(meta.paper_id) if graph.G.has_node(meta.paper_id) else 0
            if degree <= 2:
                frontier.append((meta, sim))

        frontier.sort(key=lambda x: x[1], reverse=True)
        return frontier[:3]

    # ── 7. Temporal narrative ─────────────────────────────────────────────

    def _temporal_narrative(
        self,
        papers: list[tuple["PaperMeta", float, dict]],
        graph: "HierarchicalResearchGraph",
    ) -> str:
        dated = [(m, m.year) for m, _, _ in papers if m.year]
        if len(dated) < 2:
            return ""

        dated.sort(key=lambda x: x[1])
        oldest_m, oldest_y = dated[0]
        newest_m, newest_y = dated[-1]
        span = newest_y - oldest_y
        if span == 0:
            return f"All relevant papers are from {oldest_y}."

        parts = [
            f"Research spans {span} years: from "
            f"\"{oldest_m.title[:45]}\" ({oldest_y}) to "
            f"\"{newest_m.title[:45]}\" ({newest_y})."
        ]

        # Decade clustering
        decades: dict[int, int] = defaultdict(int)
        for _, y in dated:
            decades[y // 10 * 10] += 1
        peak = max(decades, key=decades.get)
        if decades[peak] > len(dated) * 0.5:
            parts.append(f"Most activity in the {peak}s.")

        return " ".join(parts)

    # ── 8. Coverage assessment ────────────────────────────────────────────

    def _assess_coverage(
        self,
        query_emb: np.ndarray,
        papers: list[tuple["PaperMeta", float, dict]],
    ) -> str:
        if not papers:
            return "Your collection has no papers yet. Add PDFs (option 3) to get started."
        top_sim = papers[0][1]
        if top_sim < 0.30:
            return (
                "This topic is barely covered in your collection. "
                "Consider searching for new papers (option 1)."
            )
        elif top_sim < 0.50:
            return (
                "You have tangentially related papers, but this angle is "
                "underexplored. A targeted search could help."
            )
        return ""
