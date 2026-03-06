"""
graph_model.py
Core graph-based model.

Nodes  = papers (seed PDFs and papers discovered during search sessions).
         Every node stores an embedding vector (its "context vector").
Edges  = weighted semantic relationships between papers.
         Weights updated by user ratings via rate_paper().

Context vector (global) = weighted mean of all node embeddings, where each
paper's weight comes from its user rating (or DEFAULT_SEED_WEIGHT for seeds).
"""

from __future__ import annotations

import time
import hashlib
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

from researchbuddy.config import (
    DEFAULT_SEED_WEIGHT, LEARNING_RATE, SIMILARITY_THRESHOLD,
    MAX_EDGE_WEIGHT, EXPLORATION_RATIO, MIN_NOVELTY_DISTANCE,
)
from researchbuddy.core.embedder import embed, cosine_similarity, mean_pool


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    """Everything we know about a paper (seed or discovered)."""
    paper_id: str
    title: str
    abstract: str
    authors: list[str]               = field(default_factory=list)
    year: Optional[int]              = None
    url: str                         = ""
    doi: str                         = ""
    source: str                      = "discovered"   # 'seed' | 'discovered'
    filepath: str                    = ""
    s2_id: str                       = ""
    arxiv_id: str                    = ""
    embedding: Optional[np.ndarray]  = field(default=None, repr=False)

    user_rating: Optional[float]     = None
    times_shown: int                 = 0
    last_shown: float                = 0.0

    @property
    def effective_weight(self) -> float:
        if self.user_rating is not None:
            return self.user_rating
        if self.source == "seed":
            return DEFAULT_SEED_WEIGHT
        return 0.0


# ── Graph model ────────────────────────────────────────────────────────────────

class ResearchGraph:
    """Central object persisted across sessions via state_manager.pkl."""

    def __init__(self):
        self.G: nx.DiGraph           = nx.DiGraph()
        self._papers: dict[str, PaperMeta] = {}
        self._context_vec: Optional[np.ndarray] = None
        self._context_dirty: bool    = True
        self._seen_titles: set[str]  = set()

    # ── Node management ────────────────────────────────────────────────────────

    def add_paper(self, meta: PaperMeta, embedding: Optional[np.ndarray] = None) -> bool:
        """Add a paper. Returns True if added, False if already present."""
        norm_title = meta.title.lower().strip()
        if meta.paper_id in self._papers or norm_title in self._seen_titles:
            return False

        if embedding is not None:
            meta.embedding = embedding

        self._papers[meta.paper_id] = meta
        self._seen_titles.add(norm_title)
        self.G.add_node(meta.paper_id, weight=meta.effective_weight)
        self._context_dirty = True

        if meta.embedding is not None:
            self._connect_to_graph(meta)
        return True

    def _connect_to_graph(self, new_meta: PaperMeta):
        """Draw weighted edges between new_meta and similar existing papers."""
        for pid, existing in self._papers.items():
            if pid == new_meta.paper_id or existing.embedding is None:
                continue
            sim = cosine_similarity(new_meta.embedding, existing.embedding)
            if sim >= SIMILARITY_THRESHOLD:
                w = sim * (
                    (new_meta.effective_weight + existing.effective_weight) / 2
                ) / 10.0
                w = min(w, 1.0)
                self.G.add_edge(new_meta.paper_id, pid, weight=w)
                self.G.add_edge(pid, new_meta.paper_id, weight=w)

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def embed_paper(self, meta: PaperMeta, text_chunks: list[str]):
        """Embed from chunks (mean-pool), normalise, store in meta."""
        if not text_chunks:
            return
        chunk_vecs   = embed(text_chunks)
        meta.embedding = chunk_vecs.mean(axis=0)
        norm = np.linalg.norm(meta.embedding)
        if norm > 0:
            meta.embedding /= norm

    def embed_abstract(self, meta: PaperMeta):
        """Fast embedding from abstract only (for discovered papers)."""
        text = meta.abstract if meta.abstract else meta.title
        if text:
            meta.embedding = embed(text)

    # ── Context vector ─────────────────────────────────────────────────────────

    def context_vector(self) -> Optional[np.ndarray]:
        """Global context = weighted mean of seed + rated paper embeddings."""
        if not self._context_dirty and self._context_vec is not None:
            return self._context_vec

        active = [
            (m.embedding, m.effective_weight)
            for m in self._papers.values()
            if m.embedding is not None and m.effective_weight > 0
        ]
        if not active:
            return None

        vecs, weights       = zip(*active)
        self._context_vec   = mean_pool(list(vecs), list(weights))
        self._context_dirty = False
        return self._context_vec

    # ── User rating ────────────────────────────────────────────────────────────

    def rate_paper(self, paper_id: str, rating: float):
        """
        Apply user rating (1-10). Updates node weight and adjacent edge weights.
        """
        if paper_id not in self._papers:
            raise KeyError(f"Unknown paper_id: {paper_id}")

        meta       = self._papers[paper_id]
        old_weight = meta.effective_weight
        meta.user_rating = float(rating)
        new_weight = float(rating)

        self.G.nodes[paper_id]['weight'] = new_weight

        delta = (new_weight - old_weight) / 10.0 * LEARNING_RATE
        for nbr in list(self.G.successors(paper_id)):
            if self.G.has_edge(paper_id, nbr):
                old_ew = self.G[paper_id][nbr]['weight']
                new_ew = min(max(old_ew + delta, 0.0), 1.0)
                self.G[paper_id][nbr]['weight'] = new_ew
                self.G[nbr][paper_id]['weight'] = new_ew

        self._connect_to_graph(meta)
        self._context_dirty = True

    # ── Scoring candidates ─────────────────────────────────────────────────────

    def score_candidate(self, meta: PaperMeta) -> float:
        """Cosine similarity of candidate embedding to current context vector."""
        ctx = self.context_vector()
        if ctx is None or meta.embedding is None:
            return 0.0
        return float(cosine_similarity(ctx, meta.embedding))

    def novelty_score(self, meta: PaperMeta) -> float:
        """1 - max_similarity_to_any_existing_node (high = more exploratory)."""
        if meta.embedding is None:
            return 0.0
        sims = [
            cosine_similarity(meta.embedding, m.embedding)
            for m in self._papers.values()
            if m.embedding is not None
        ]
        if not sims:
            return 1.0
        return float(1.0 - max(sims))

    def rank_candidates(
        self,
        candidates: list[PaperMeta],
        n: int = 10,
        exploration_ratio: float = EXPLORATION_RATIO,
    ) -> list[tuple[PaperMeta, float, str]]:
        """
        Return list of (meta, score, label) mixing relevance and exploration.
        label is 'relevant' or 'explore'.
        """
        new_candidates = [
            c for c in candidates
            if c.paper_id not in self._papers
            and c.title.lower().strip() not in self._seen_titles
        ]
        if not new_candidates:
            return []

        scored = []
        for c in new_candidates:
            if c.embedding is None:
                self.embed_abstract(c)
            rel   = self.score_candidate(c)
            novel = self.novelty_score(c)
            scored.append((c, rel, novel))

        n_explore  = max(1, int(n * exploration_ratio))
        n_relevant = n - n_explore

        by_relevance = sorted(scored, key=lambda x: x[1], reverse=True)
        by_novelty   = sorted(
            [s for s in scored if s[1] > 0.15 and s[2] >= MIN_NOVELTY_DISTANCE],
            key=lambda x: x[2], reverse=True
        )

        results: list[tuple[PaperMeta, float, str]] = []
        seen_ids: set[str] = set()

        for meta, rel, _ in by_relevance[:n_relevant]:
            if meta.paper_id not in seen_ids:
                results.append((meta, rel, "relevant"))
                seen_ids.add(meta.paper_id)

        for meta, _, nov in by_novelty[:n_explore]:
            if meta.paper_id not in seen_ids and len(results) < n:
                results.append((meta, nov, "explore"))
                seen_ids.add(meta.paper_id)

        return results

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_paper(self, paper_id: str) -> Optional[PaperMeta]:
        return self._papers.get(paper_id)

    def all_papers(self) -> list[PaperMeta]:
        return list(self._papers.values())

    def seed_papers(self) -> list[PaperMeta]:
        return [m for m in self._papers.values() if m.source == "seed"]

    def rated_papers(self) -> list[PaperMeta]:
        return [m for m in self._papers.values() if m.user_rating is not None]

    def top_seed_keywords(self, n: int = 8) -> list[str]:
        """Top keyword phrases from seed papers via KeyBERT (falls back to title words)."""
        seed = self.seed_papers()
        if not seed:
            return []
        combined = " ".join(m.title + " " + m.abstract for m in seed[:10])
        try:
            from keybert import KeyBERT
            kb = KeyBERT()
            keywords = kb.extract_keywords(
                combined, keyphrase_ngram_range=(1, 3),
                stop_words='english', top_n=n
            )
            return [kw for kw, _ in keywords]
        except Exception:
            words = set()
            for m in seed[:5]:
                words.update(m.title.split())
            stop = {'a','an','the','of','in','and','or','for','to','with','on'}
            return list(words - stop)[:n]

    def stats(self) -> dict:
        return {
            "total_papers" : len(self._papers),
            "seed_papers"  : len(self.seed_papers()),
            "rated_papers" : len(self.rated_papers()),
            "graph_edges"  : self.G.number_of_edges(),
            "context_ready": self.context_vector() is not None,
        }

    @staticmethod
    def make_id(title: str, doi: str = "", s2_id: str = "", arxiv_id: str = "") -> str:
        if s2_id:
            return f"s2_{s2_id}"
        if doi:
            return f"doi_{hashlib.sha1(doi.encode()).hexdigest()[:10]}"
        if arxiv_id:
            return f"arxiv_{arxiv_id.replace('/', '_')}"
        return f"title_{hashlib.sha1(title.lower().encode()).hexdigest()[:10]}"
