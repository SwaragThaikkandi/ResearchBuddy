"""
Tests for the discovery-engine upgrades: Personalized PageRank signal, MMR
slate diversification, Reciprocal Rank Fusion, focus mode, and the
age-normalised impact prior — plus weight-schema migration.
"""

from __future__ import annotations

import numpy as np
import pytest

import researchbuddy.core.graph_model as gm
from researchbuddy.core.graph_model import (
    HierarchicalResearchGraph, PaperMeta, EXTRA_SIGNAL_TYPES,
)
from researchbuddy.config import SCORED_SECTION_TYPES


def _vec(seed, dim=384, base=None, jitter=0.05):
    rng = np.random.RandomState(seed)
    v = (base + jitter * rng.randn(dim)) if base is not None else rng.randn(dim)
    return v / np.linalg.norm(v)


def _paper(pid, v, **kw):
    return PaperMeta(paper_id=pid, title=f"Paper {pid}", abstract="x",
                     embedding=v, **kw)


def _chain_graph(n=6, rate_first=2):
    """Path graph p0—p1—...—p(n-1) in the semantic layer; p0..rated."""
    base = _vec(0)
    g = HierarchicalResearchGraph(alpha=0.6)
    papers = []
    for i in range(n):
        m = _paper(f"p{i}", _vec(i + 1, base=base), source="seed", year=2020)
        if i < rate_first:
            m.user_rating = 9.0
        g.add_paper(m, m.embedding)
        papers.append(m)
    import networkx as nx
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(f"p{i}", f"p{i+1}", weight=1.0)
        G.add_edge(f"p{i+1}", f"p{i}", weight=1.0)
    g.G_semantic = G
    return g, papers


# ── Personalized PageRank ──────────────────────────────────────────────────────

def test_ppr_mass_decays_with_distance_from_rated():
    g, _ = _chain_graph()
    mass = g._ppr_mass()
    assert mass is not None
    # p1 is adjacent to the rated restart nodes; p5 is four hops away.
    assert mass["p1"] > mass["p4"] > 0
    assert mass["p1"] > mass["p5"]


def test_ppr_none_without_edges():
    g = HierarchicalResearchGraph(alpha=0.6)
    m = _paper("solo", _vec(1))
    m.user_rating = 8.0
    g.add_paper(m, m.embedding)
    assert g._ppr_mass() is None          # no edges → masked, never fabricated


def test_ppr_signal_for_off_graph_candidate():
    g, papers = _chain_graph()
    near = _paper("cand_near", papers[1].embedding)   # clone of central node
    far = _paper("cand_far", _vec(999))                # unrelated direction
    s_near = g._ppr_signal(near)
    s_far = g._ppr_signal(far)
    assert s_near is not None and s_far is not None
    assert s_near > s_far


def test_ppr_leakage_removed_for_restart_nodes():
    """Restart-set members must be scored by NETWORK flow, not by their own
    restart injection — otherwise weight learning inflates the ppr signal."""
    g, _ = _chain_graph()
    mass = g._ppr_mass()
    # p2 (unrated, two rated neighbours upstream) receives flow comparable to
    # the rated nodes' own network mass — injection was subtracted.
    assert mass["p2"] > 0.1


# ── Impact prior ───────────────────────────────────────────────────────────────

def test_impact_signal_monotonic_and_bounded():
    old_low = _paper("a", _vec(1), year=2000, cited_by_count=5)
    new_hot = _paper("b", _vec(2), year=2024, cited_by_count=200)
    unknown = _paper("c", _vec(3), year=2024)
    s_low = HierarchicalResearchGraph._impact_signal(old_low)
    s_hot = HierarchicalResearchGraph._impact_signal(new_hot)
    assert HierarchicalResearchGraph._impact_signal(unknown) is None
    assert 0.0 <= s_low < s_hot <= 1.0


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def test_rrf_rewards_multi_source_consensus():
    consensus = _paper("x", _vec(1))
    consensus.source_ranks = {"openalex": 0, "crossref": 1, "s2_search": 2}
    single = _paper("y", _vec(2))
    single.source_ranks = {"openalex": 0}
    none = _paper("z", _vec(3))
    s_c = HierarchicalResearchGraph._rrf_score(consensus)
    s_s = HierarchicalResearchGraph._rrf_score(single)
    assert HierarchicalResearchGraph._rrf_score(none) == 0.0
    assert 0.0 < s_s < s_c <= 1.0


def test_rrf_rank_order_matters():
    top = _paper("t", _vec(1)); top.source_ranks = {"openalex": 0}
    deep = _paper("d", _vec(2)); deep.source_ranks = {"openalex": 40}
    assert (HierarchicalResearchGraph._rrf_score(top)
            > HierarchicalResearchGraph._rrf_score(deep))


# ── Signal vector layout + migration ──────────────────────────────────────────

def test_extract_signals_length_includes_extras():
    g, papers = _chain_graph()
    sig = g._extract_signals(papers[0])
    assert len(sig) == 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)
    assert len(g._default_signal_weights) == len(sig)


def test_old_learned_weights_are_padded_not_crashed():
    """A graph pickled with the 11-dim schema must keep scoring safely."""
    import pickle
    g, papers = _chain_graph()
    g._learned_signal_weights = np.full(7 + len(SCORED_SECTION_TYPES), 1.0)
    g2 = pickle.loads(pickle.dumps(g))
    expected = 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)
    assert len(g2._learned_signal_weights) == expected
    # scoring an abstract-only candidate does not crash and stays bounded
    cand = _paper("cand", papers[0].embedding, year=2023)
    s = g2.score_candidate(cand)
    assert 0.0 <= s <= 1.0


# ── MMR diversification ────────────────────────────────────────────────────────

def test_mmr_slate_prefers_coverage_over_clones():
    """Given one good hit plus its exact clone plus an equally-relevant but
    DIFFERENT paper, MMR must pick the different one over the clone for
    slot 2 (identical relevance, redundancy 1.0 vs ~0.5)."""
    g, papers = _chain_graph(n=6, rate_first=3)
    base = papers[0].embedding

    best = _paper("best", _vec(50, base=base, jitter=0.05), year=2023)
    clone = _paper("clone", best.embedding.copy(), year=2023)
    # Same jitter magnitude, different direction: relevance ~= best's, but
    # cosine to best is far below 1.0.
    distinct = _paper("distinct", _vec(60, base=base, jitter=0.05), year=2023)

    results = g.rank_candidates([best, clone, distinct], n=2,
                                exploration_ratio=0.0)
    ids = [m.paper_id for m, _, _ in results]
    assert len(ids) == 2
    if "best" in ids and "clone" in ids:
        pytest.fail("MMR picked the exact duplicate over the distinct paper")


# ── Focus mode ─────────────────────────────────────────────────────────────────

def test_focus_mode_recenters_ranking():
    """Two topic clusters in one graph; focusing on cluster B must rank a
    B-flavoured candidate above an A-flavoured one, and vice versa."""
    baseA, baseB = _vec(1000), _vec(2000)
    g = HierarchicalResearchGraph(alpha=0.6)
    a_ids, b_ids = [], []
    for i in range(6):
        ma = _paper(f"A{i}", _vec(i, base=baseA), source="seed", year=2020)
        mb = _paper(f"B{i}", _vec(100 + i, base=baseB), source="seed", year=2020)
        ma.user_rating = 8.0
        g.add_paper(ma, ma.embedding); a_ids.append(ma.paper_id)
        g.add_paper(mb, mb.embedding); b_ids.append(mb.paper_id)

    cand_a = _paper("cand_a", _vec(500, base=baseA), year=2023)
    cand_b = _paper("cand_b", _vec(600, base=baseB), year=2023)

    # Baseline (no focus): the RATED A-cluster dominates → cand_a wins.
    res_plain = g.rank_candidates([cand_a, cand_b], n=2, exploration_ratio=0.0)
    plain = {m.paper_id: s for m, s, _ in res_plain}
    assert plain.get("cand_a", 0) > plain.get("cand_b", 0)

    # Focused on the (unrated) B cluster: ranking must flip.
    res_b = g.rank_candidates([cand_a, cand_b], n=2, exploration_ratio=0.0,
                              focus_ids=b_ids)
    scores_b = {m.paper_id: s for m, s, _ in res_b}
    assert scores_b.get("cand_b", 0) > scores_b.get("cand_a", 0)

    # focus state fully restored afterwards
    assert g._focus_ids is None and g._context_override is None


def test_focused_context_uses_only_chosen_papers():
    g, papers = _chain_graph()
    ctx = g.focused_context([papers[3].paper_id])
    assert ctx is not None
    assert float(np.dot(ctx / np.linalg.norm(ctx),
                        papers[3].embedding)) > 0.99
