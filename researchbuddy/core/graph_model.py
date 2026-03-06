"""
graph_model.py
HierarchicalResearchGraph — the central persistent object.

Architecture
------------
Semantic stream
    Paper embeddings (384-dim, sentence-transformers)
    → AgglomerativeClustering → niche / area cluster nodes
    → Intra-niche edges (cosine sim) + inter-niche shortcuts (small-world)

Citation stream
    S2 reference lists → bibliographic-coupling + co-citation matrices

Fusion
    Similarity Network Fusion (SNF) merges both streams
    → A single fused similarity matrix is used for context + ranking

Context vector
    Weighted mean of paper embeddings where weight = user rating (or 5 for seeds)
    For ranking new candidates, cosine similarity to this vector is combined
    with the fused citation score.
"""

from __future__ import annotations

import hashlib
import time
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

from researchbuddy.config import (
    DEFAULT_SEED_WEIGHT, LEARNING_RATE, SIMILARITY_THRESHOLD,
    EXPLORATION_RATIO, MIN_NOVELTY_DISTANCE, N_HIERARCHY_LEVELS,
    FUSION_ALPHA, SNF_KNN, SNF_ITER,
)
from researchbuddy.core.embedder    import embed, cosine_similarity, mean_pool
from researchbuddy.core.hierarchy   import build_hswn, ClusterNode, compute_paper_level_map
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
    Persistent object saved via state_manager.pkl between sessions.
    """

    def __init__(self, n_levels: int = N_HIERARCHY_LEVELS, alpha: float = FUSION_ALPHA):
        # Graph (papers + cluster nodes, all edges)
        self.G: nx.DiGraph = nx.DiGraph()

        # Paper registry
        self._papers: dict[str, PaperMeta]  = {}
        self._seen_titles: set[str]          = set()

        # Hierarchical cluster nodes (all levels ≥ 1)
        self._clusters: dict[str, ClusterNode] = {}

        # Citation data: s2_id → set of cited s2_ids
        self._refs: dict[str, set[str]]      = {}

        # Fused similarity matrix cache
        # Rows/cols aligned to self._ordered_ids()
        self._fused_W: Optional[np.ndarray]  = None
        self._fused_ids: list[str]           = []   # paper_ids when W was last built

        # Context vector cache
        self._context_vec: Optional[np.ndarray] = None
        self._context_dirty: bool               = True

        # Hyper-parameters (can be overridden via CLI)
        self.n_levels  = n_levels
        self.alpha     = alpha

    # ── Ordered paper list (stable order for matrix alignment) ─────────────────

    def _ordered_ids(self) -> list[str]:
        return list(self._papers.keys())

    # ── Add paper ──────────────────────────────────────────────────────────────

    def add_paper(self, meta: PaperMeta, embedding: Optional[np.ndarray] = None) -> bool:
        norm = meta.title.lower().strip()
        if meta.paper_id in self._papers or norm in self._seen_titles:
            return False
        if embedding is not None:
            meta.embedding = embedding

        self._papers[meta.paper_id] = meta
        self._seen_titles.add(norm)
        self.G.add_node(meta.paper_id, level=0, node_type="paper",
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
        (Re)build the HSWN from current paper embeddings.
        Also rebuilds the fused similarity matrix.
        Called after adding papers or updating ratings.
        """
        papers_with_emb = [m for m in self._papers.values() if m.embedding is not None]
        if len(papers_with_emb) < 3:
            return  # not enough papers for a meaningful hierarchy

        ids  = [m.paper_id for m in papers_with_emb]
        embs = [m.embedding for m in papers_with_emb]

        # --- Semantic HSWN ---------------------------------------------------
        G_sem, clusters = build_hswn(
            paper_ids=ids,
            embeddings=embs,
            n_levels=self.n_levels,
            intra_threshold=SIMILARITY_THRESHOLD,
            shortcut_threshold=SIMILARITY_THRESHOLD + 0.15,
        )

        # Merge new hierarchical graph into self.G
        self.G = G_sem   # replace (keeps all paper nodes + new cluster nodes)
        self._clusters = clusters

        # Restore node weights from paper registry
        for pid, meta in self._papers.items():
            if pid in self.G:
                self.G.nodes[pid]["weight"] = meta.effective_weight

        # --- Citation graph --------------------------------------------------
        s2_ids = [self._papers[pid].s2_id for pid in ids]
        if any(s2_ids):
            G_cit = build_citation_graph(ids, s2_ids, self._refs)
            # Merge citation edges into self.G
            for u, v, d in G_cit.edges(data=True):
                if self.G.has_node(u) and self.G.has_node(v):
                    self.G.add_edge(u, v, etype="citation", weight=d.get("weight", 1.0))

        # --- Fused similarity matrix -----------------------------------------
        self._build_fused_matrix(ids, embs, s2_ids)

        self._context_dirty = True

    def _build_fused_matrix(self, ids: list[str], embs: list[np.ndarray], s2_ids: list[str]):
        n = len(ids)
        if n < 2:
            return

        # Semantic similarity
        E       = np.stack(embs)      # (n, dim)
        W_sem   = E @ E.T             # cosine sim (unit-normed)
        np.fill_diagonal(W_sem, 0.0)

        # Citation similarity
        W_cit = citation_similarity_matrix(ids, s2_ids, self._refs)
        np.fill_diagonal(W_cit, 0.0)

        # SNF fusion
        self._fused_W   = snf(W_sem, W_cit, alpha=self.alpha, k=SNF_KNN, n_iter=SNF_ITER)
        self._fused_ids = ids

    # ── Context vector ─────────────────────────────────────────────────────────

    def context_vector(self) -> Optional[np.ndarray]:
        """
        Weighted mean of paper embeddings (weight = effective_weight).
        Level-aware: niche centroids from HSWN carry boosted weight.
        """
        if not self._context_dirty and self._context_vec is not None:
            return self._context_vec

        active = [
            (m.embedding, m.effective_weight)
            for m in self._papers.values()
            if m.embedding is not None and m.effective_weight > 0
        ]
        if not active:
            return None

        vecs, weights = zip(*active)

        # Also blend in niche centroids weighted by sum of paper weights per niche
        extra_vecs, extra_weights = [], []
        for node in self._clusters.values():
            if node.level != 1:
                continue
            niche_w = sum(
                self._papers[pid].effective_weight
                for pid in node.paper_ids
                if pid in self._papers
            )
            if niche_w > 0:
                extra_vecs.append(node.centroid)
                extra_weights.append(niche_w * 0.5)   # half-weight for cluster centroids

        all_vecs    = list(vecs) + extra_vecs
        all_weights = list(weights) + extra_weights

        self._context_vec   = mean_pool(all_vecs, all_weights)
        self._context_dirty = False
        return self._context_vec

    # ── User rating ────────────────────────────────────────────────────────────

    def rate_paper(self, paper_id: str, rating: float):
        if paper_id not in self._papers:
            raise KeyError(paper_id)
        meta = self._papers[paper_id]
        meta.user_rating = float(rating)
        if paper_id in self.G:
            self.G.nodes[paper_id]["weight"] = float(rating)
        self._invalidate()

    # ── Fetch citations for existing papers ────────────────────────────────────

    def fetch_citations(self, verbose: bool = True):
        """Fetch S2 references for all papers that have an S2 ID but no refs yet."""
        s2_ids_needed = [
            m.s2_id for m in self._papers.values()
            if m.s2_id and m.s2_id not in self._refs
        ]
        if not s2_ids_needed:
            return
        if verbose:
            print(f"[graph] Fetching citations for {len(s2_ids_needed)} papers ...")
        self._refs = fetch_all_references(s2_ids_needed, self._refs)
        self._invalidate()

    # ── Scoring candidates ─────────────────────────────────────────────────────

    def score_candidate(self, meta: PaperMeta) -> float:
        """Fused relevance score for a candidate paper."""
        # Semantic score (cosine to context)
        ctx = self.context_vector()
        if ctx is None or meta.embedding is None:
            return 0.0
        sem_score = float(cosine_similarity(ctx, meta.embedding))

        # Citation score: bibliographic coupling to existing highly-rated papers
        cit_score = self._citation_score(meta)

        return fuse_scores(sem_score, cit_score, alpha=self.alpha)

    def _citation_score(self, meta: PaperMeta) -> float:
        """Bibliographic coupling between candidate and our existing papers."""
        if not meta.s2_id or meta.s2_id not in self._refs:
            return 0.0
        refs_cand = self._refs.get(meta.s2_id, set())
        if not refs_cand:
            return 0.0

        scores = []
        for m in self._papers.values():
            if not m.s2_id or m.effective_weight == 0:
                continue
            refs_m = self._refs.get(m.s2_id, set())
            if not refs_m:
                continue
            overlap = len(refs_cand & refs_m)
            if overlap:
                coupling = overlap / np.sqrt(len(refs_cand) * len(refs_m))
                scores.append(coupling * m.effective_weight / 10.0)

        return float(np.mean(scores)) if scores else 0.0

    def novelty_score(self, meta: PaperMeta) -> float:
        if meta.embedding is None:
            return 0.0
        sims = [
            cosine_similarity(meta.embedding, m.embedding)
            for m in self._papers.values()
            if m.embedding is not None
        ]
        return float(1.0 - max(sims)) if sims else 1.0

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
            [s for s in scored if s[1] > 0.15 and s[2] >= MIN_NOVELTY_DISTANCE],
            key=lambda x: x[2], reverse=True
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
        """Return {paper_id: niche_node_id} for all papers that have been clustered."""
        mapping: dict[str, str] = {}
        for nid, nd in self._clusters.items():
            if nd.level == 1:
                for pid in nd.paper_ids:
                    mapping[pid] = nid
        return mapping

    def top_seed_keywords(self, n: int = 8) -> list[str]:
        seed = self.seed_papers()
        if not seed:
            return []
        combined = " ".join(m.title + " " + m.abstract for m in seed[:10])
        try:
            from keybert import KeyBERT
            kws = KeyBERT().extract_keywords(
                combined, keyphrase_ngram_range=(1, 3),
                stop_words="english", top_n=n
            )
            return [kw for kw, _ in kws]
        except Exception:
            words = set()
            for m in seed[:5]:
                words.update(m.title.split())
            stop = {"a","an","the","of","in","and","or","for","to","with","on"}
            return list(words - stop)[:n]

    def stats(self) -> dict:
        return {
            "total_papers"   : len(self._papers),
            "seed_papers"    : len(self.seed_papers()),
            "rated_papers"   : len(self.rated_papers()),
            "niche_clusters" : sum(1 for nd in self._clusters.values() if nd.level == 1),
            "area_clusters"  : sum(1 for nd in self._clusters.values() if nd.level == 2),
            "graph_edges"    : self.G.number_of_edges(),
            "citations_loaded": len(self._refs),
            "context_ready"  : self.context_vector() is not None,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

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

    # ── Backward compatibility shim ────────────────────────────────────────────

    @classmethod
    def from_legacy(cls, old) -> "HierarchicalResearchGraph":
        """Migrate an old flat ResearchGraph pickle to HierarchicalResearchGraph."""
        new = cls()
        for meta in getattr(old, "_papers", {}).values():
            new._papers[meta.paper_id] = meta
            new._seen_titles.add(meta.title.lower().strip())
            new.G.add_node(meta.paper_id, level=0, node_type="paper",
                           weight=meta.effective_weight)
        print(f"[graph] Migrated {len(new._papers)} papers from legacy format.")
        return new


# Keep old name as alias so old pickles that reference ResearchGraph still load
ResearchGraph = HierarchicalResearchGraph
