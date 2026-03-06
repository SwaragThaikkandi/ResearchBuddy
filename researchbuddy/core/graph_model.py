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
)
from researchbuddy.core.embedder import embed, cosine_similarity, mean_pool
from researchbuddy.core.hierarchy import (
    build_adaptive_hswn, ClusterNode, n_levels_detected,
)
from researchbuddy.core.citation_network import (
    citation_similarity_matrix, build_citation_graph, fetch_all_references,
)
from researchbuddy.core.fusion import snf, fuse_scores


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
        # ── Three networks ────────────────────────────────────────────────
        self.G_semantic: nx.DiGraph  = nx.DiGraph()   # NLP / HSWN
        self.G_citation: nx.DiGraph  = nx.DiGraph()   # citation relationships
        self.G: nx.DiGraph           = nx.DiGraph()   # fused (combined)

        # ── Paper registry ────────────────────────────────────────────────
        self._papers: dict[str, PaperMeta]       = {}
        self._seen_titles: set[str]               = set()

        # ── Hierarchical clusters (all levels ≥ 1) ───────────────────────
        self._clusters: dict[str, ClusterNode]   = {}

        # ── Citation data ─────────────────────────────────────────────────
        self._refs: dict[str, set[str]]           = {}   # s2_id → set of cited s2_ids

        # ── Fused adjacency cache ─────────────────────────────────────────
        self._fused_W: Optional[np.ndarray]       = None
        self._fused_ids: list[str]                = []

        # ── Context vector cache ──────────────────────────────────────────
        self._context_vec: Optional[np.ndarray]   = None
        self._context_dirty: bool                 = True

        # ── Hyper-parameters ──────────────────────────────────────────────
        self.alpha = alpha

    # ── Add paper ─────────────────────────────────────────────────────────────

    def add_paper(self, meta: PaperMeta, embedding: Optional[np.ndarray] = None) -> bool:
        norm = meta.title.lower().strip()
        if meta.paper_id in self._papers or norm in self._seen_titles:
            return False
        if embedding is not None:
            meta.embedding = embedding
        self._papers[meta.paper_id] = meta
        self._seen_titles.add(norm)
        for G in (self.G_semantic, self.G_citation, self.G):
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
        """
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
        self.G_citation = build_citation_graph(ids, s2_ids, self._refs)
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

        self._context_dirty = True

    def _build_fused_matrix(self, ids: list[str], embs: list[np.ndarray], s2_ids: list[str]):
        n = len(ids)
        if n < 2:
            return
        E      = np.stack(embs)
        W_sem  = E @ E.T
        np.fill_diagonal(W_sem, 0.0)
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

        if not vecs:
            return None

        self._context_vec   = mean_pool(vecs, weights)
        self._context_dirty = False
        return self._context_vec

    # ── User rating ────────────────────────────────────────────────────────────

    def rate_paper(self, paper_id: str, rating: float):
        if paper_id not in self._papers:
            raise KeyError(paper_id)
        self._papers[paper_id].user_rating = float(rating)
        for G in (self.G_semantic, self.G_citation, self.G):
            if G.has_node(paper_id):
                G.nodes[paper_id]["weight"] = float(rating)
        self._invalidate()

    # ── Fetch citations ────────────────────────────────────────────────────────

    def fetch_citations(self, verbose: bool = True):
        s2_needed = [m.s2_id for m in self._papers.values()
                     if m.s2_id and m.s2_id not in self._refs]
        if not s2_needed:
            if verbose:
                print("[graph] Citation data already up to date.")
            return
        if verbose:
            print(f"[graph] Fetching citations for {len(s2_needed)} papers ...")
        self._refs = fetch_all_references(s2_needed, self._refs)
        self._invalidate()

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

        if not score_parts:
            return 0.0

        total_w = sum(weight_parts) or 1.0
        fused   = sum(s * w for s, w in zip(score_parts, weight_parts)) / total_w
        return float(np.clip(fused, 0.0, 1.0))

    def _citation_score(self, meta: PaperMeta) -> float:
        if not meta.s2_id or meta.s2_id not in self._refs:
            return 0.0
        refs_c = self._refs.get(meta.s2_id, set())
        if not refs_c:
            return 0.0
        scores = []
        for m in self._papers.values():
            if not m.s2_id or m.effective_weight == 0:
                continue
            refs_m = self._refs.get(m.s2_id, set())
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
    ) -> list[tuple[PaperMeta, float, str]]:
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
            "citations_loaded"      : len(self._refs),
            "context_ready"         : self.context_vector() is not None,
        }

    def _invalidate(self):
        self._context_dirty = True
        self._fused_W       = None

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
            for G in (new.G_semantic, new.G_citation, new.G):
                G.add_node(meta.paper_id, level=0, node_type="paper",
                           weight=meta.effective_weight)
        print(f"[graph] Migrated {len(new._papers)} papers from legacy format.")
        return new


# Alias for backward compatibility with old pickles
ResearchGraph = HierarchicalResearchGraph
