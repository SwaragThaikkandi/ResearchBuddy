"""
Regression: abstract-only search candidates must not be floored to ~0 just
because the user's learned weights favour per-section similarity signals that
only full-text (GROBID-ingested) papers can supply.

Repro of the field bug: 142 candidates fetched, 0 shown, "No new candidates
found" — because every fresh candidate (abstract only, no section embeddings)
scored below REL_FLOOR=0.30 once section weights dominated.
"""

from __future__ import annotations

import numpy as np

from researchbuddy.config import SCORED_SECTION_TYPES
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


def _clustered_graph(n=20, dim=384, seed=0):
    """A tight, single-topic graph whose papers all carry section embeddings."""
    rng = np.random.RandomState(seed)
    base = rng.randn(dim); base /= np.linalg.norm(base)
    g = HierarchicalResearchGraph(alpha=0.6)
    for i in range(n):
        v = base + 0.05 * rng.randn(dim)
        v /= np.linalg.norm(v)
        m = PaperMeta(paper_id=f"p{i}", title=f"Paper {i}", abstract="a",
                      source="seed", embedding=v, year=2022)
        m.section_embeddings = {"methods": v, "results": v}
        if i < 8:
            m.user_rating = 8.0
        g.add_paper(m, v)
    g.rebuild_hierarchy()
    return g, base


def _section_heavy_weights():
    w = np.full(7 + len(SCORED_SECTION_TYPES), 0.3)
    w[7] = 5.0          # methods section dominates (mirrors the real graph)
    if 7 + 2 < len(w):
        w[7 + 2] = 4.0  # discussion
    return w


def test_abstract_only_candidate_not_floored():
    g, base = _clustered_graph()
    g._learned_signal_weights = _section_heavy_weights()
    g._section_context_dirty = True

    # Relevant, abstract-only candidate (no section embeddings) — exactly what
    # every OpenAlex / CrossRef / S2 search result looks like.
    cand = PaperMeta(paper_id="cand", title="A new relevant paper",
                     abstract="x", embedding=base, year=2023)
    score = g.score_candidate(cand)
    assert score >= 0.30, f"abstract-only candidate floored at {score:.3f}"


def test_masking_preserves_relevance_ordering():
    g, base = _clustered_graph()
    g._learned_signal_weights = _section_heavy_weights()
    g._section_context_dirty = True

    relevant = PaperMeta(paper_id="rel", title="relevant", abstract="x",
                         embedding=base, year=2023)
    rng = np.random.RandomState(99)
    off = rng.randn(len(base)); off /= np.linalg.norm(off)   # orthogonal-ish
    irrelevant = PaperMeta(paper_id="irr", title="off-topic", abstract="x",
                           embedding=off, year=2023)

    s_rel = g.score_candidate(relevant)
    s_irr = g.score_candidate(irrelevant)
    assert s_rel > s_irr      # masking must not flatten the ranking signal


def test_candidate_with_sections_scores_at_least_as_high():
    g, base = _clustered_graph()
    g._learned_signal_weights = _section_heavy_weights()
    g._section_context_dirty = True

    abstract_only = PaperMeta(paper_id="a", title="t", abstract="x",
                              embedding=base, year=2023)
    with_sections = PaperMeta(paper_id="b", title="t2", abstract="x",
                              embedding=base, year=2023)
    with_sections.section_embeddings = {"methods": base, "results": base}

    # Having strong matching sections should not score lower than not having
    # them (the masking only removes a penalty, it doesn't invert the signal).
    assert g.score_candidate(with_sections) >= g.score_candidate(abstract_only) - 1e-9


def test_full_ranking_surfaces_abstract_only_candidates():
    """End-to-end: rank_candidates must return abstract-only candidates."""
    g, base = _clustered_graph()
    g._learned_signal_weights = _section_heavy_weights()
    g._section_context_dirty = True
    rng = np.random.RandomState(7)

    candidates = []
    for i in range(10):
        v = base + 0.04 * rng.randn(len(base))
        v /= np.linalg.norm(v)
        candidates.append(PaperMeta(paper_id=f"c{i}", title=f"Cand {i}",
                                    abstract="x", embedding=v, year=2023))

    results = g.rank_candidates(candidates, n=5, exploration_ratio=0.0)
    assert len(results) >= 1, "abstract-only candidates were all floored out"


def test_snowball_and_watcher_candidates_not_floored():
    """
    Snowballing (menu 16), living-review watches (menu 18), and search (menu 1)
    all funnel through graph.rank_candidates → score_candidate. Their candidates
    are abstract-only metadata (OpenAlex/CrossRef/S2), so the same section-weight
    deflation would have silently zeroed them. Guard all three at once.
    """
    g, base = _clustered_graph()
    g._learned_signal_weights = _section_heavy_weights()
    g._section_context_dirty = True
    rng = np.random.RandomState(11)

    for source in ("snowball", "watch", "discovered"):
        cands = []
        for i in range(8):
            v = base + 0.05 * rng.randn(len(base)); v /= np.linalg.norm(v)
            m = PaperMeta(paper_id=f"{source}_{i}", title=f"{source} cand {i}",
                          abstract="", doi=f"10.x/{source}{i}",
                          source=source, embedding=v)
            assert not m.section_embeddings          # the failing shape
            cands.append(m)
        results = g.rank_candidates(cands, n=5, exploration_ratio=0.0)
        assert len(results) >= 1, f"{source} candidates all floored out"
