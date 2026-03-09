"""
graph_model.py
HierarchicalResearchGraph — central persistent object.

Three networks are maintained separately:
  G_semantic  — Hierarchical Small World Network built from NLP embeddings
  G_citation  — Directed citation graph (paper A cites paper B)
  G           — Combined/fused graph (semantic + citation edges merged)

Prediction is multi-level:
  1. Paper-level cosine similarity to global context vector
  2. Similarity to each niche centroid, weighted by that niche's importance
  3. Similarity to each area (and higher) centroid
  4. Citation coupling score (bibliographic coupling + co-citation)
  5. Graph proximity: shortest-path distance from candidate to highly-rated papers
  All streams are combined using SNF-derived weights (alpha parameter).
"""

from __future__ import annotations

import hashlib
import re
import numpy as np
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from researchbuddy.config import (
    DEFAULT_SEED_WEIGHT, SIMILARITY_THRESHOLD,
    EXPLORATION_RATIO, MIN_NOVELTY_DISTANCE,
    FUSION_ALPHA, SNF_KNN, SNF_ITER,
    MIN_CLUSTER_SIZE, MAX_HIERARCHY_LEVELS,
    QUERY_FEEDBACK_WEIGHT,
)
from researchbuddy.core.embedder import embed, cosine_similarity, mean_pool
from researchbuddy.core.hierarchy import (
    build_adaptive_hswn, ClusterNode, n_levels_detected,
)
from researchbuddy.core.citation_network import (
    citation_similarity_matrix, build_citation_graph,
    fetch_all_refs, fetch_all_references, extract_doi_from_text,
    _looks_like_journal_header,
)
from researchbuddy.core.pdf_processor import reextract_title_doi
from researchbuddy.core.fusion import snf, fuse_scores
from researchbuddy.core.reasoner import QueryInteraction
from researchbuddy.core.arguer import ArgumentInteraction, StyleProfile


# ── Paper data model ───────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]              = field(default_factory=list)
    year: Optional[int]             = None
    url: str                        = ""
    doi: str                        = ""
    source: str                     = "discovered"
    filepath: str                   = ""
    s2_id: str                      = ""
    arxiv_id: str                   = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    venue: str                      = ""
    is_peer_reviewed: Optional[bool]= None   # None=unknown, True=journal/conf, False=preprint
    user_rating: Optional[float]    = None
    times_shown: int                = 0
    last_shown: float               = 0.0

    @property
    def effective_weight(self) -> float:
        if self.user_rating is not None:
            return self.user_rating
        return DEFAULT_SEED_WEIGHT if self.source == "seed" else 0.0


# ── Hierarchical research graph ────────────────────────────────────────────────

class HierarchicalResearchGraph:
    """
    The central object saved between sessions.
    Three graphs are maintained and exported as separate PDFs.
    """

    def __init__(self, alpha: float = FUSION_ALPHA):
        # ── Four networks ────────────────────────────────────────────────
        self.G_semantic: nx.DiGraph  = nx.DiGraph()   # NLP / HSWN
        self.G_citation: nx.DiGraph  = nx.DiGraph()   # citation relationships
        self.G: nx.DiGraph           = nx.DiGraph()   # fused (combined)
        self.G_causal: nx.DiGraph    = nx.DiGraph()   # causal DAG (acyclic influence)

        # ── Paper registry ────────────────────────────────────────────────
        self._papers: dict[str, PaperMeta]       = {}
        self._seen_titles: set[str]               = set()

        # ── Hierarchical clusters (all levels ≥ 1) ───────────────────────
        self._clusters: dict[str, ClusterNode]   = {}

        # ── Citation data ─────────────────────────────────────────────────
        self._refs: dict[str, set[str]]           = {}   # s2_id → set of cited s2_ids
        self._ref_sources: dict[str, list]        = {}   # paper_id → list[RefResult]
        self._edge_anomalies: list[tuple]         = []   # (src, tgt, reason, penalty)

        # ── Fused adjacency cache ─────────────────────────────────────────
        self._fused_W: Optional[np.ndarray]       = None
        self._fused_ids: list[str]                = []

        # ── Context vector cache ──────────────────────────────────────────
        self._context_vec: Optional[np.ndarray]   = None
        self._context_dirty: bool                 = True

        # ── Query interactions (reasoner feedback) ────────────────────────
        self._query_interactions: list[QueryInteraction] = []

        # ── Argument interactions (creative mode feedback) ─────────────────
        self._argument_interactions: list[ArgumentInteraction] = []
        self._style_profile: Optional[StyleProfile]            = None

        # ── Hyper-parameters ──────────────────────────────────────────────
        self.alpha = alpha

    # ── Add paper ─────────────────────────────────────────────────────────────

    def add_paper(self, meta: PaperMeta, embedding: Optional[np.ndarray] = None) -> bool:
        norm = meta.title.lower().strip()
        if meta.paper_id in self._papers or norm in self._seen_titles:
            return False
        if embedding is not None:
            meta.embedding = embedding
        self._infer_peer_review_status(meta)
        self._papers[meta.paper_id] = meta
        self._seen_titles.add(norm)
        for G in (self.G_semantic, self.G_citation, self.G, self.G_causal):
            G.add_node(meta.paper_id, level=0, node_type="paper",
                       weight=meta.effective_weight)
        self._invalidate()
        return True

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def embed_paper(self, meta: PaperMeta, text_chunks: list[str]):
        if not text_chunks:
            return
        vecs = embed(text_chunks)
        meta.embedding = vecs.mean(axis=0)
        n = np.linalg.norm(meta.embedding)
        if n > 0:
            meta.embedding /= n

    def embed_abstract(self, meta: PaperMeta):
        text = meta.abstract or meta.title
        if text:
            meta.embedding = embed(text)

    # ── Hierarchy rebuild ──────────────────────────────────────────────────────

    def rebuild_hierarchy(self):
        """
        Rebuild all three networks from scratch using current papers.
        The number of hierarchy levels is determined automatically by the data.
        Auto-fetches missing citation data before rebuilding.
        """
        # Auto-fetch citations for papers that don't have them yet
        n_missing = sum(1 for m in self._papers.values()
                        if m.paper_id not in self._refs)
        if n_missing > 0:
            print(f"[graph] Auto-fetching citations for {n_missing} papers ...")
            self.fetch_citations(verbose=True)

        papers_with_emb = [m for m in self._papers.values() if m.embedding is not None]
        if len(papers_with_emb) < 3:
            return

        ids  = [m.paper_id for m in papers_with_emb]
        embs = [m.embedding for m in papers_with_emb]

        # ── 1. Semantic HSWN ──────────────────────────────────────────────
        G_sem, clusters = build_adaptive_hswn(
            paper_ids         = ids,
            embeddings        = embs,
            min_cluster_size  = MIN_CLUSTER_SIZE,
            max_levels        = MAX_HIERARCHY_LEVELS,
            intra_threshold   = SIMILARITY_THRESHOLD,
            shortcut_threshold= SIMILARITY_THRESHOLD + 0.15,
        )
        self.G_semantic = G_sem
        self._clusters  = clusters

        # Restore node weights
        for pid, meta in self._papers.items():
            if pid in self.G_semantic:
                self.G_semantic.nodes[pid]["weight"] = meta.effective_weight

        n_found = n_levels_detected(clusters)
        if n_found:
            print(f"[graph] Hierarchy: {n_found} level(s) detected automatically "
                  f"({sum(1 for c in clusters.values() if c.level==1)} niches, "
                  f"{sum(1 for c in clusters.values() if c.level==2)} areas"
                  + (f", {sum(1 for c in clusters.values() if c.level>=3)} higher)" if n_found>=3 else ")"))

        # ── 2. Citation graph ─────────────────────────────────────────────
        s2_ids = [self._papers[pid].s2_id for pid in ids]
        self.G_citation = build_citation_graph(
            ids, s2_ids, self._refs, ref_sources=self._ref_sources
        )
        # Ensure all paper nodes exist
        for pid in ids:
            if not self.G_citation.has_node(pid):
                self.G_citation.add_node(pid, level=0, node_type="paper",
                                         weight=self._papers[pid].effective_weight)

        # ── 3. Fused similarity matrix ────────────────────────────────────
        self._build_fused_matrix(ids, embs, s2_ids)

        # ── 4. Combined graph = semantic + citation edges ─────────────────
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.G_semantic.nodes(data=True))
        self.G.add_edges_from(self.G_semantic.edges(data=True))
        for u, v, d in self.G_citation.edges(data=True):
            if self.G.has_node(u) and self.G.has_node(v):
                self.G.add_edge(u, v, **d)

        # ── 5. Annotate citation edges with relationship types ─────────────
        try:
            from researchbuddy.core.citation_classifier import annotate_citation_types
            annotate_citation_types(self.G_citation, self._papers)
        except Exception as e:
            print(f"[graph] Citation type annotation skipped: {e}")

        # ── 6. Causal DAG (acyclic influence flow) ────────────────────────
        try:
            from researchbuddy.core.causal import build_causal_dag
            from researchbuddy.config import CAUSAL_CONFIDENCE_THRESHOLD
            self.G_causal, self._edge_anomalies = build_causal_dag(
                self.G, self.G_citation, self._papers,
                min_confidence=CAUSAL_CONFIDENCE_THRESHOLD,
            )
            print(f"[graph] Causal DAG: {self.G_causal.number_of_edges()} edges, "
                  f"acyclic={nx.is_directed_acyclic_graph(self.G_causal)}")
        except Exception as e:
            print(f"[graph] Causal DAG construction skipped: {e}")
            self.G_causal = nx.DiGraph()
            self._edge_anomalies = []

        self._context_dirty = True

    def _build_fused_matrix(self, ids: list[str], embs: list[np.ndarray], s2_ids: list[str]):
        n = len(ids)
        if n < 2:
            return
        E      = np.stack(embs)
        W_sem  = E @ E.T
        np.fill_diagonal(W_sem, 0.0)
        # _refs is now keyed by paper_id; s2_ids passed for legacy edge building
        W_cit  = citation_similarity_matrix(ids, s2_ids, self._refs)
        np.fill_diagonal(W_cit, 0.0)
        self._fused_W   = snf(W_sem, W_cit, alpha=self.alpha, k=SNF_KNN, n_iter=SNF_ITER)
        self._fused_ids = ids

    # ── Context vector ─────────────────────────────────────────────────────────

    def context_vector(self) -> Optional[np.ndarray]:
        """
        Hierarchical context vector: weighted mean of paper embeddings AND
        niche/area centroids (to capture higher-level research direction).
        """
        if not self._context_dirty and self._context_vec is not None:
            return self._context_vec

        vecs, weights = [], []

        # Paper-level contributions
        for m in self._papers.values():
            if m.embedding is not None and m.effective_weight > 0:
                vecs.append(m.embedding)
                weights.append(m.effective_weight)

        # Niche-level contributions (weighted by sum of paper weights in niche)
        for node in self._clusters.values():
            if node.level == 1:
                nw = sum(self._papers[p].effective_weight
                         for p in node.paper_ids if p in self._papers)
                if nw > 0:
                    vecs.append(node.centroid)
                    weights.append(nw * 0.5)

        # Area-level contributions (lighter weight)
        for node in self._clusters.values():
            if node.level == 2:
                aw = sum(self._papers[p].effective_weight
                         for p in node.paper_ids if p in self._papers)
                if aw > 0:
                    vecs.append(node.centroid)
                    weights.append(aw * 0.25)

        # Query-level contributions (reasoner feedback)
        for qi in self._query_interactions:
            if qi.rating >= 6:
                vecs.append(qi.query_embedding)
                weights.append((qi.rating - 5) * QUERY_FEEDBACK_WEIGHT)

        if not vecs:
            return None

        self._context_vec   = mean_pool(vecs, weights)
        self._context_dirty = False
        return self._context_vec

    # ── User rating ────────────────────────────────────────────────────────────

    def apply_query_feedback(
        self,
        query_embedding: np.ndarray,
        paper_ids: list[str],
        rating: float,
    ):
        """
        Store a rated query interaction AND mutate the network.

        High ratings (>= 7):
          * Strengthen / create edges between shown papers in G_semantic & G
          * Boost node weights for the top shown papers
        Low ratings (<= 3):
          * Slightly dampen node weights of shown papers

        The context vector is also shifted via _query_interactions (see
        context_vector()).
        """
        import time
        qi = QueryInteraction(
            query_embedding=query_embedding,
            paper_ids=paper_ids,
            rating=rating,
            timestamp=time.time(),
        )
        self._query_interactions.append(qi)

        if rating >= 7:
            boost = (rating - 6) * 0.5     # 7→0.5, 8→1.0, 9→1.5, 10→2.0

            # ── Strengthen / create edges between shown papers ────────────
            for i, pid_a in enumerate(paper_ids):
                for pid_b in paper_ids[i + 1:]:
                    for G in (self.G_semantic, self.G):
                        if G.has_edge(pid_a, pid_b):
                            w = G[pid_a][pid_b].get("weight", 0.5)
                            G[pid_a][pid_b]["weight"] = min(1.0, w + boost * 0.1)
                        elif G.has_node(pid_a) and G.has_node(pid_b):
                            # Create an interest edge if papers are at least
                            # somewhat related
                            ma = self._papers.get(pid_a)
                            mb = self._papers.get(pid_b)
                            if (ma and mb
                                    and ma.embedding is not None
                                    and mb.embedding is not None):
                                sim = float(cosine_similarity(
                                    ma.embedding, mb.embedding))
                                if sim > 0.30:
                                    G.add_edge(pid_a, pid_b,
                                               weight=sim,
                                               edge_type="interest")

            # ── Boost node weights for top shown papers ───────────────────
            for pid in paper_ids[:5]:
                meta = self._papers.get(pid)
                if meta and meta.user_rating is None:
                    for G in (self.G_semantic, self.G_citation, self.G, self.G_causal):
                        if G.has_node(pid):
                            old_w = G.nodes[pid].get("weight",
                                                     DEFAULT_SEED_WEIGHT)
                            G.nodes[pid]["weight"] = min(10.0,
                                                         old_w + boost * 0.2)

        elif rating <= 3:
            dampen = (4 - rating) * 0.2    # 1→0.6, 2→0.4, 3→0.2
            for pid in paper_ids[:5]:
                if pid in self._papers:
                    for G in (self.G_semantic, self.G_citation, self.G, self.G_causal):
                        if G.has_node(pid):
                            old_w = G.nodes[pid].get("weight",
                                                     DEFAULT_SEED_WEIGHT)
                            G.nodes[pid]["weight"] = max(1.0, old_w - dampen)

        self._invalidate()

    # ── Creative mode feedback ─────────────────────────────────────────────────

    def apply_argument_feedback(self, interaction: ArgumentInteraction):
        """
        Record an argument interaction and update the StyleProfile.

        The StyleProfile biases future argument generation toward types the
        user rates highly (correctness + usefulness).
        """
        from researchbuddy.config import ARGUER_STYLE_LR
        self._argument_interactions.append(interaction)
        if self._style_profile is None:
            self._style_profile = StyleProfile()
        self._style_profile.update(
            interaction.argument_type,
            interaction.correctness,
            interaction.usefulness,
            lr=ARGUER_STYLE_LR,
        )

    def get_style_profile(self) -> StyleProfile:
        """Return the persistent StyleProfile, creating one if absent."""
        if self._style_profile is None:
            self._style_profile = StyleProfile()
        return self._style_profile

    def rate_paper(self, paper_id: str, rating: float):
        if paper_id not in self._papers:
            raise KeyError(paper_id)
        self._papers[paper_id].user_rating = float(rating)
        for G in (self.G_semantic, self.G_citation, self.G, self.G_causal):
            if G.has_node(paper_id):
                G.nodes[paper_id]["weight"] = float(rating)
        self._invalidate()

    # ── Fetch citations ────────────────────────────────────────────────────────

    def fetch_citations(self, verbose: bool = True):
        """
        Fetch references for every paper that does not yet have citation data.
        Strategy (per paper):
          0. Fix garbage titles by re-reading PDF metadata / font analysis
          1. Extract DOI from title/abstract text if not already set
          2. CrossRef by DOI  (returns cited DOIs — most reliable)
          3. CrossRef bibliographic query  (fuzzy, uses abstract text)
          4. OpenAlex by DOI / title
          5. Semantic Scholar fallback
        """
        # ── Step 0: Re-extract titles/DOIs for papers still missing refs ──
        # Re-read the original PDF (metadata + font-size analysis) for every
        # paper that failed citation lookup previously. This fixes two-column
        # journal headers, ligature damage, download-notice titles, etc.
        missing_ids = {pid for pid in self._papers
                       if pid not in self._refs}
        n_title_fixed = 0
        n_doi_fixed = 0
        for meta in self._papers.values():
            if meta.paper_id not in missing_ids:
                continue
            fp = getattr(meta, 'filepath', '')
            if not fp:
                continue
            new_title, new_doi = reextract_title_doi(fp)
            if new_title and new_title != meta.title:
                if verbose:
                    print(f"  [title fix] {(meta.title or '')[:35]!r}"
                          f" → {new_title[:50]!r}")
                meta.title = new_title
                n_title_fixed += 1
            if new_doi and not getattr(meta, 'doi', ''):
                meta.doi = new_doi
                n_doi_fixed += 1
        if verbose and (n_title_fixed or n_doi_fixed):
            print(f"[graph] Re-extracted from PDFs:"
                  f" {n_title_fixed} titles, {n_doi_fixed} DOIs fixed.")

        # ── Step 1: Scan text for DOIs not extracted during import ────────
        for meta in self._papers.values():
            if not getattr(meta, "doi", ""):
                doi = extract_doi_from_text(
                    (meta.title or "") + " " + (meta.abstract or "")
                )
                if doi:
                    meta.doi = doi

        need = [m for m in self._papers.values()
                if m.paper_id not in self._refs]
        if not need:
            if verbose:
                print("[graph] Citation data already up to date.")
            return
        if verbose:
            print(f"[graph] Fetching citations for {len(need)} papers ...")
        self._refs = fetch_all_refs(
            need, existing=self._refs, verbose=verbose,
            ref_sources_out=self._ref_sources,
        )
        self._invalidate()

    # ── Publication quality ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_peer_review_status(meta: PaperMeta) -> None:
        """
        Best-effort inference of peer-review status from available metadata.
        Only sets is_peer_reviewed if it can be reasonably inferred.
        """
        if meta.is_peer_reviewed is not None:
            return   # already set

        venue = (getattr(meta, "venue", "") or "").lower()

        # ArXiv-only → preprint
        if getattr(meta, "arxiv_id", "") and not getattr(meta, "doi", ""):
            meta.is_peer_reviewed = False
            if not meta.venue:
                meta.venue = "arXiv"
            return

        # Venue contains "arxiv" or "preprint"
        if "arxiv" in venue or "preprint" in venue:
            meta.is_peer_reviewed = False
            return

        # Has DOI and not ArXiv → likely peer-reviewed
        if getattr(meta, "doi", "") and not getattr(meta, "arxiv_id", ""):
            meta.is_peer_reviewed = True
            return

        # Otherwise → unknown
        # meta.is_peer_reviewed remains None

    def _pub_quality_score(self, meta: PaperMeta) -> float:
        """
        Publication quality bonus — nudge, not dominate.

        Peer-reviewed: 1.0
        Unknown:       0.5  (neutral)
        Preprint:      0.3  (mild penalty — preprints can still be excellent)
        """
        pr = getattr(meta, "is_peer_reviewed", None)
        if pr is True:
            return 1.0
        elif pr is False:
            return 0.3
        return 0.5

    # ── Comprehensive multi-level scoring ──────────────────────────────────────

    def score_candidate(self, meta: PaperMeta) -> float:
        """
        Comprehensive relevance score combining five signals:
          1. Cosine similarity to global context vector (paper level)
          2. Similarity to each niche centroid, weighted by niche importance
          3. Similarity to each area centroid (and higher levels)
          4. Citation coupling (bibliographic coupling + co-citation)
          5. SNF-fused adjacency score (if matrix available)
        """
        if meta.embedding is None:
            return 0.0

        score_parts: list[float] = []
        weight_parts: list[float] = []

        # ── 1. Global context similarity ──────────────────────────────────
        ctx = self.context_vector()
        if ctx is not None:
            score_parts.append(float(cosine_similarity(ctx, meta.embedding)))
            weight_parts.append(3.0)

        # ── 2. Niche-level similarities ───────────────────────────────────
        for node in self._clusters.values():
            if node.level == 1:
                niche_w = sum(self._papers[p].effective_weight
                              for p in node.paper_ids if p in self._papers)
                if niche_w > 0:
                    sim = float(cosine_similarity(node.centroid, meta.embedding))
                    score_parts.append(sim)
                    weight_parts.append(niche_w / 10.0 * 2.0)

        # ── 3. Area/domain-level similarities ─────────────────────────────
        for node in self._clusters.values():
            if node.level >= 2:
                area_w = sum(self._papers[p].effective_weight
                             for p in node.paper_ids if p in self._papers)
                if area_w > 0:
                    sim = float(cosine_similarity(node.centroid, meta.embedding))
                    # Discount deeper levels slightly (coarser = less precise)
                    discount = 0.8 ** (node.level - 1)
                    score_parts.append(sim * discount)
                    weight_parts.append(area_w / 20.0)

        # ── 4. Citation coupling ───────────────────────────────────────────
        cit = self._citation_score(meta)
        score_parts.append(cit)
        weight_parts.append(2.0 * (1.0 - self.alpha))

        # ── 5. SNF fused adjacency (if available) ─────────────────────────
        snf_score = self._snf_score(meta)
        if snf_score > 0:
            score_parts.append(snf_score)
            weight_parts.append(1.5)

        # ── 6. Publication quality (peer-review bonus) ────────────────────
        pub_q = self._pub_quality_score(meta)
        score_parts.append(pub_q)
        weight_parts.append(0.5)   # light weight — nudge, not dominate

        if not score_parts:
            return 0.0

        total_w = sum(weight_parts) or 1.0
        fused   = sum(s * w for s, w in zip(score_parts, weight_parts)) / total_w
        return float(np.clip(fused, 0.0, 1.0))

    def _citation_score(self, meta: PaperMeta) -> float:
        """
        Bibliographic coupling between a candidate and all corpus papers.
        Uses paper_id-keyed refs (OpenAlex IDs or S2 IDs).
        """
        refs_c = self._refs.get(meta.paper_id, set())
        if not refs_c:
            return 0.0
        scores = []
        for m in self._papers.values():
            if m.effective_weight == 0:
                continue
            refs_m = self._refs.get(m.paper_id, set())
            if not refs_m:
                continue
            overlap = len(refs_c & refs_m)
            if overlap:
                coupling = overlap / np.sqrt(len(refs_c) * len(refs_m))
                scores.append(coupling * m.effective_weight / 10.0)
        return float(np.mean(scores)) if scores else 0.0

    def _snf_score(self, meta: PaperMeta) -> float:
        """Average fused similarity to top-rated papers (from precomputed matrix)."""
        if self._fused_W is None or meta.embedding is None:
            return 0.0
        # Find top-weighted papers in fused matrix
        top_ids = [pid for pid in self._fused_ids
                   if self._papers.get(pid) and self._papers[pid].effective_weight >= 6][:10]
        if not top_ids:
            return 0.0
        # Approximate: use cosine similarity to those papers' embeddings,
        # then scale by their row in the fused matrix (captures indirect connections)
        ctx_vecs = [self._papers[pid].embedding for pid in top_ids
                    if self._papers[pid].embedding is not None]
        if not ctx_vecs:
            return 0.0
        sims = [float(cosine_similarity(v, meta.embedding)) for v in ctx_vecs]
        return float(np.mean(sims))

    # ── Novelty score ──────────────────────────────────────────────────────────

    def novelty_score(self, meta: PaperMeta) -> float:
        """
        How far is this paper from ALL existing papers AND cluster centroids?
        High novelty = useful for exploration.
        """
        if meta.embedding is None:
            return 0.0
        embs = [m.embedding for m in self._papers.values() if m.embedding is not None]
        # Include cluster centroids for richer comparison
        embs += [nd.centroid for nd in self._clusters.values()]
        if not embs:
            return 1.0
        max_sim = max(float(cosine_similarity(meta.embedding, e)) for e in embs)
        return float(1.0 - max_sim)

    # ── Ranking ────────────────────────────────────────────────────────────────

    def rank_candidates(
        self,
        candidates: list[PaperMeta],
        n: int = 10,
        exploration_ratio: float = EXPLORATION_RATIO,
        hyde_embedding: Optional[np.ndarray] = None,
    ) -> list[tuple[PaperMeta, float, str]]:
        """
        Rank candidates by fused (semantic + citation) relevance, with
        optional HyDE embedding blending.

        When ``hyde_embedding`` is provided (from the LLM-generated hypothetical
        abstract), each candidate's score is blended:
            final = graph_score * 0.6 + hyde_sim * 0.4
        This lets the LLM's understanding of the research intent supplement
        the graph-based scoring, while keeping the graph as the primary signal.
        """
        new_cands = [
            c for c in candidates
            if c.paper_id not in self._papers
            and c.title.lower().strip() not in self._seen_titles
        ]
        if not new_cands:
            return []

        scored = []
        for c in new_cands:
            if c.embedding is None:
                self.embed_abstract(c)
            rel   = self.score_candidate(c)

            # Blend HyDE similarity when available
            if hyde_embedding is not None and c.embedding is not None:
                hyde_sim = float(cosine_similarity(hyde_embedding, c.embedding))
                hyde_sim = max(0.0, hyde_sim)  # clamp negatives
                rel = rel * 0.6 + hyde_sim * 0.4

            novel = self.novelty_score(c)
            scored.append((c, rel, novel))

        n_explore  = max(1, int(n * exploration_ratio))
        n_relevant = n - n_explore

        by_rel   = sorted(scored, key=lambda x: x[1], reverse=True)
        by_novel = sorted(
            [s for s in scored if s[1] > 0.12 and s[2] >= MIN_NOVELTY_DISTANCE],
            key=lambda x: x[2], reverse=True,
        )

        results: list[tuple[PaperMeta, float, str]] = []
        seen: set[str] = set()

        for meta, rel, _ in by_rel[:n_relevant]:
            if meta.paper_id not in seen:
                results.append((meta, rel, "relevant"))
                seen.add(meta.paper_id)

        for meta, _, nov in by_novel[:n_explore]:
            if meta.paper_id not in seen and len(results) < n:
                results.append((meta, nov, "explore"))
                seen.add(meta.paper_id)

        return results

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_paper(self, pid: str) -> Optional[PaperMeta]:
        return self._papers.get(pid)

    def all_papers(self) -> list[PaperMeta]:
        return list(self._papers.values())

    def seed_papers(self) -> list[PaperMeta]:
        return [m for m in self._papers.values() if m.source == "seed"]

    def rated_papers(self) -> list[PaperMeta]:
        return [m for m in self._papers.values() if m.user_rating is not None]

    def paper_to_niche(self) -> dict[str, str]:
        return {pid: nid
                for nid, nd in self._clusters.items()
                if nd.level == 1
                for pid in nd.paper_ids}

    def top_seed_keywords(self, n: int = 8) -> list[str]:
        seed = self.seed_papers()
        if not seed:
            return []
        combined = " ".join(m.title + " " + m.abstract for m in seed[:10])
        try:
            from keybert import KeyBERT
            kws = KeyBERT().extract_keywords(
                combined, keyphrase_ngram_range=(1, 3),
                stop_words="english", top_n=n,
            )
            return [kw for kw, _ in kws]
        except Exception:
            words = set()
            for m in seed[:5]:
                words.update(m.title.split())
            stop = {"a","an","the","of","in","and","or","for","to","with","on"}
            return list(words - stop)[:n]

    def stats(self) -> dict:
        nd = n_levels_detected(self._clusters)
        return {
            "total_papers"          : len(self._papers),
            "seed_papers"           : len(self.seed_papers()),
            "rated_papers"          : len(self.rated_papers()),
            "hierarchy_levels"      : nd,
            "niche_clusters"        : sum(1 for c in self._clusters.values() if c.level == 1),
            "area_clusters"         : sum(1 for c in self._clusters.values() if c.level == 2),
            "higher_clusters"       : sum(1 for c in self._clusters.values() if c.level >= 3),
            "semantic_edges"        : self.G_semantic.number_of_edges(),
            "citation_edges"        : self.G_citation.number_of_edges(),
            "combined_edges"        : self.G.number_of_edges(),
            "causal_edges"          : self.G_causal.number_of_edges(),
            "causal_is_dag"         : nx.is_directed_acyclic_graph(self.G_causal)
                                      if self.G_causal.number_of_edges() > 0 else True,
            "citations_loaded"      : len(self._refs),
            "multi_source_refs"     : sum(1 for v in self._ref_sources.values()
                                         if len(v) >= 2),
            "peer_reviewed"         : sum(1 for m in self._papers.values()
                                         if getattr(m, "is_peer_reviewed", None) is True),
            "preprints"             : sum(1 for m in self._papers.values()
                                         if getattr(m, "is_peer_reviewed", None) is False),
            "edge_anomalies"        : len(getattr(self, "_edge_anomalies", [])),
            "context_ready"         : self.context_vector() is not None,
        }

    def _invalidate(self):
        self._context_dirty = True
        self._fused_W       = None

    # ── Pickle backward compatibility ──────────────────────────────────────────

    def __setstate__(self, state: dict):
        """
        Called by pickle.load() when deserialising an older saved graph.
        Initialises any attributes that did not exist in earlier versions.
        """
        self.__dict__.update(state)
        # v0.3.0 additions — absent from v0.2.0 pickles
        if not hasattr(self, "_clusters"):
            self._clusters: dict = {}
        if not hasattr(self, "G_semantic"):
            self.G_semantic = nx.DiGraph()
        if not hasattr(self, "G_citation"):
            self.G_citation = nx.DiGraph()
        if not hasattr(self, "_refs"):
            self._refs: dict = {}
        else:
            # v0.3.0 refs were keyed by s2_id; v0.3.1+ uses paper_id.
            # Detect by checking if any key looks like a paper_id (short hex/prefix).
            # If refs are s2-keyed, reset — they will be re-fetched via OpenAlex.
            paper_id_set = set(getattr(self, "_papers", {}).keys())
            if self._refs and not any(k in paper_id_set for k in self._refs):
                print("[graph] Migrating citation refs to new format (will re-fetch) ...")
                self._refs = {}
        if not hasattr(self, "_fused_W"):
            self._fused_W = None
        if not hasattr(self, "_fused_ids"):
            self._fused_ids: list = []
        if not hasattr(self, "_context_vec"):
            self._context_vec = None
        if not hasattr(self, "_context_dirty"):
            self._context_dirty = True
        # v0.4.0 — query interactions (reasoner feedback)
        if not hasattr(self, "_query_interactions"):
            self._query_interactions: list = []
        # v0.5.0 — argument interactions + style profile (creative mode)
        if not hasattr(self, "_argument_interactions"):
            self._argument_interactions: list = []
        if not hasattr(self, "_style_profile"):
            self._style_profile = None
        # v0.6.0 — causal DAG (acyclic influence flow)
        if not hasattr(self, "G_causal"):
            self.G_causal = nx.DiGraph()
        # v0.8.0 — citation cross-validation + publication quality
        if not hasattr(self, "_ref_sources"):
            self._ref_sources: dict = {}
        if not hasattr(self, "_edge_anomalies"):
            self._edge_anomalies: list = []
        for meta in self._papers.values():
            if not hasattr(meta, "venue"):
                meta.venue = ""
            if not hasattr(meta, "is_peer_reviewed"):
                meta.is_peer_reviewed = None
        # v0.2.0 used n_levels; v0.3.0 uses alpha only
        if not hasattr(self, "alpha"):
            self.alpha = FUSION_ALPHA
        # Ensure combined graph exists
        if not hasattr(self, "G") or self.G is None:
            self.G = nx.DiGraph()
        # Rebuild three networks from scratch if we only have the old single G
        if (self.G_semantic.number_of_nodes() == 0
                and len(self._papers) >= 3):
            print("[graph] Migrating v0.2.0 graph to v0.3.0 three-network format ...")
            # Seed all paper nodes into the three graphs
            for meta in self._papers.values():
                for graph in (self.G_semantic, self.G_citation, self.G):
                    if not graph.has_node(meta.paper_id):
                        graph.add_node(meta.paper_id, level=0,
                                       node_type="paper",
                                       weight=meta.effective_weight)

    @staticmethod
    def make_id(title: str, doi: str = "", s2_id: str = "", arxiv_id: str = "") -> str:
        if s2_id:
            return f"s2_{s2_id}"
        if doi:
            return f"doi_{hashlib.sha1(doi.encode()).hexdigest()[:10]}"
        if arxiv_id:
            return f"arxiv_{arxiv_id.replace('/', '_')}"
        return f"title_{hashlib.sha1(title.lower().encode()).hexdigest()[:10]}"

    @classmethod
    def from_legacy(cls, old) -> "HierarchicalResearchGraph":
        new = cls()
        for meta in getattr(old, "_papers", {}).values():
            new._papers[meta.paper_id] = meta
            new._seen_titles.add(meta.title.lower().strip())
            for G in (new.G_semantic, new.G_citation, new.G, new.G_causal):
                G.add_node(meta.paper_id, level=0, node_type="paper",
                           weight=meta.effective_weight)
        print(f"[graph] Migrated {len(new._papers)} papers from legacy format.")
        return new


# Alias for backward compatibility with old pickles
ResearchGraph = HierarchicalResearchGraph
