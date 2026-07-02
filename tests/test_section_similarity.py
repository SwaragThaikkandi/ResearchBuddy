"""
Tests for the per-section similarity graph layer.

The section_embeddings + section-similarity layer adds new dimensions to
the recommender (papers' methods <-> other papers' methods, etc.). These
tests use synthetic embeddings so they don't depend on the real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.graph_backend import LAYER_SECTION_SIM


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def _make_paper(g: HierarchicalResearchGraph, pid: str, title: str,
                section_vecs: dict[str, np.ndarray]) -> PaperMeta:
    """Add a paper with hand-rolled per-section embeddings."""
    meta = PaperMeta(paper_id=pid, title=title, abstract="")
    meta.section_embeddings = {k: _unit(v) for k, v in section_vecs.items()}
    # The paper itself needs *some* top-level embedding for add_paper checks
    meta.embedding = _unit(np.mean(list(section_vecs.values()), axis=0))
    g.add_paper(meta)
    return meta


def test_section_layer_built_for_papers_sharing_section():
    """Two papers with similar methods sections should get a methods edge."""
    g = HierarchicalResearchGraph()
    # methods vectors are nearly identical (~1.0 cosine)
    methods_a = np.array([1.0, 0.0, 0.0, 0.0])
    methods_b = np.array([0.99, 0.01, 0.0, 0.01])
    # results vectors are orthogonal (~0 cosine)
    results_a = np.array([0.0, 1.0, 0.0, 0.0])
    results_b = np.array([0.0, 0.0, 1.0, 0.0])

    _make_paper(g, "a", "A", {"methods": methods_a, "results": results_a})
    _make_paper(g, "b", "B", {"methods": methods_b, "results": results_b})

    # Use a low threshold so methods qualify and results doesn't
    g._build_section_similarity_layer(threshold=0.5)

    edges = list(g._backend.edges_data(LAYER_SECTION_SIM))
    sections_present = {e[2].get("section") for e in edges}
    assert "methods" in sections_present
    assert "results" not in sections_present


def test_section_edges_are_symmetric():
    """Both directions are written so undirected queries find them."""
    g = HierarchicalResearchGraph()
    v = np.array([1.0, 0.0, 0.0, 0.0])
    _make_paper(g, "a", "A", {"methods": v})
    _make_paper(g, "b", "B", {"methods": v + np.array([0.01, 0.01, 0.0, 0.0])})

    g._build_section_similarity_layer(threshold=0.5)
    pairs = {(u, v_) for u, v_, _ in g._backend.edges_data(LAYER_SECTION_SIM)}
    assert ("a", "b") in pairs and ("b", "a") in pairs


def test_section_edges_carry_weight_and_section_props():
    g = HierarchicalResearchGraph()
    v = np.array([1.0, 0.0, 0.0, 0.0])
    _make_paper(g, "a", "A", {"results": v})
    _make_paper(g, "b", "B", {"results": v})

    g._build_section_similarity_layer(threshold=0.5)
    edges = list(g._backend.edges_data(LAYER_SECTION_SIM))
    assert edges, "expected at least one edge"
    e = edges[0][2]
    assert e["section"] == "results"
    assert "weight" in e
    assert 0.99 <= e["weight"] <= 1.0


def test_section_layer_handles_papers_without_section_embeddings():
    """Papers with no section_embeddings simply don't participate."""
    g = HierarchicalResearchGraph()
    v = np.array([1.0, 0.0, 0.0, 0.0])
    _make_paper(g, "a", "A", {"methods": v})
    _make_paper(g, "b", "B", {"methods": v})

    # A third paper without section embeddings — must not crash builder
    bare = PaperMeta(paper_id="c", title="C", abstract="")
    bare.embedding = _unit(np.array([0.5, 0.5, 0.0, 0.0]))
    g.add_paper(bare)

    g._build_section_similarity_layer(threshold=0.5)
    nodes_in_layer = {u for u, _, _ in g._backend.edges_data(LAYER_SECTION_SIM)}
    assert "c" not in nodes_in_layer  # never linked
    assert {"a", "b"}.issubset(nodes_in_layer | {"a", "b"})


def test_below_threshold_pairs_get_no_edge():
    g = HierarchicalResearchGraph()
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])   # orthogonal -> sim = 0
    _make_paper(g, "a", "A", {"methods": a})
    _make_paper(g, "b", "B", {"methods": b})

    g._build_section_similarity_layer(threshold=0.5)
    edges = list(g._backend.edges_data(LAYER_SECTION_SIM))
    assert edges == []


# ── Phase 2: section signals as scoring dimensions ──────────────────────────

class TestSectionScoringSignals:
    """
    The recommender's _extract_signals now returns 7 + N section signals.
    These tests pin down (a) the dimensionality, (b) that a candidate sharing
    section content with rated positives gets a non-zero section signal, and
    (c) that weight learning still works on the extended feature vector.
    """

    def _seed_graph(self, n_papers: int = 10) -> HierarchicalResearchGraph:
        """Tiny fixture: papers with hand-rolled embeddings + section vectors."""
        from researchbuddy.config import COLD_START_THRESHOLD
        # We need at least COLD_START_THRESHOLD papers for the full scoring
        # path to fire (otherwise we get the cold-start fast path).
        n = max(n_papers, COLD_START_THRESHOLD + 1)
        g = HierarchicalResearchGraph()
        rng = np.random.default_rng(0)
        for i in range(n):
            paper_emb = _unit(rng.standard_normal(8))
            sections = {
                "methods":     _unit(rng.standard_normal(8)),
                "results":     _unit(rng.standard_normal(8)),
                "discussion":  _unit(rng.standard_normal(8)),
                "introduction":_unit(rng.standard_normal(8)),
            }
            meta = PaperMeta(
                paper_id=f"p{i}", title=f"Paper {i}", abstract="",
                year=2020 + (i % 5),
            )
            meta.embedding = paper_emb
            meta.section_embeddings = sections
            g.add_paper(meta)
        return g

    def test_extract_signals_returns_extended_vector(self):
        from researchbuddy.config import SCORED_SECTION_TYPES
        g = self._seed_graph()
        any_meta = next(iter(g._papers.values()))
        sig = g._extract_signals(any_meta)
        assert sig is not None
        from researchbuddy.core.graph_model import EXTRA_SIGNAL_TYPES
        assert len(sig) == 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)

    def test_section_signal_zero_without_user_ratings(self):
        """No rated papers => no user-section context => section signals == 0."""
        g = self._seed_graph()
        meta = next(iter(g._papers.values()))
        sig = g._extract_signals(meta)
        assert np.all(sig[7:] == 0.0)

    def test_section_signal_lights_up_after_positive_rating(self):
        """After rating a paper highly, candidates similar to its methods
        section should get a non-zero methods signal."""
        from researchbuddy.config import SCORED_SECTION_TYPES
        g = self._seed_graph()

        rated = list(g._papers.values())[0]
        cand  = list(g._papers.values())[1]

        # Make the candidate's methods near-identical to the rated paper's
        cand.section_embeddings["methods"] = rated.section_embeddings["methods"].copy()

        g.rate_paper(rated.paper_id, 9.0)

        sig = g._extract_signals(cand)
        methods_idx = 7 + SCORED_SECTION_TYPES.index("methods")
        assert sig[methods_idx] > 0.95, \
            "candidate sharing methods with positive should signal strongly"

    def test_negative_ratings_do_not_pull_user_context(self):
        """Negative ratings (<5) must not contribute to the section context."""
        g = self._seed_graph()
        only_neg = list(g._papers.values())[0]
        g.rate_paper(only_neg.paper_id, 2.0)

        cand = list(g._papers.values())[1]
        cand.section_embeddings["methods"] = only_neg.section_embeddings["methods"].copy()

        sig = g._extract_signals(cand)
        # No positives rated -> methods context is None -> signal stays 0
        from researchbuddy.config import SCORED_SECTION_TYPES
        methods_idx = 7 + SCORED_SECTION_TYPES.index("methods")
        assert sig[methods_idx] == 0.0

    def test_default_weights_include_section_dims(self):
        from researchbuddy.config import SCORED_SECTION_TYPES
        from researchbuddy.core.graph_model import EXTRA_SIGNAL_TYPES
        g = HierarchicalResearchGraph()
        assert len(g._default_signal_weights) == (
            7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES))

    def test_weight_learning_works_with_extended_features(self):
        """The logistic regression must accept the wider feature matrix
        and produce a same-length learned weight vector."""
        from researchbuddy.config import (
            SCORED_SECTION_TYPES, WEIGHT_LEARNING_MIN_RATINGS,
        )
        g = self._seed_graph(n_papers=max(20, WEIGHT_LEARNING_MIN_RATINGS * 2))
        # Mix positives (>=7) and negatives (<=4) so learning is well-posed
        papers = list(g._papers.values())
        for i, m in enumerate(papers):
            g.rate_paper(m.paper_id, 9.0 if i % 2 == 0 else 2.0)

        ok = g.learn_signal_weights()
        assert ok, "weight learning should succeed with enough positives/negatives"
        assert g._learned_signal_weights is not None
        from researchbuddy.core.graph_model import EXTRA_SIGNAL_TYPES
        assert len(g._learned_signal_weights) == (
            7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES))
        # All learned weights should be positive (clamped to >= 0.05)
        assert (g._learned_signal_weights >= 0.05 - 1e-9).all()

    def test_setstate_migrates_old_7dim_default_weights(self):
        """Pickles from v1.0–v2.2 had 7-dim defaults; setstate must extend."""
        from researchbuddy.config import SCORED_SECTION_TYPES
        from researchbuddy.core.graph_model import EXTRA_SIGNAL_TYPES
        g = HierarchicalResearchGraph()
        # Simulate an old pickled state
        g._default_signal_weights = np.array([3.0, 2.0, 1.0, 2.0, 1.5, 0.5, 0.3])
        g._learned_signal_weights = np.ones(7) * 1.5
        # Re-run __setstate__ migration to upgrade
        state = g.__dict__.copy()
        new_g = HierarchicalResearchGraph.__new__(HierarchicalResearchGraph)
        new_g.__setstate__(state)
        n_total = 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)
        assert len(new_g._default_signal_weights) == n_total
        # Old learned weights are PRESERVED for the trained positions and
        # padded with defaults for the new tail (v0.8.1 behaviour — no more
        # throwing away the user's training on every schema bump).
        assert new_g._learned_signal_weights is not None
        assert len(new_g._learned_signal_weights) == n_total
        assert np.allclose(new_g._learned_signal_weights[:7], 1.5)


def test_section_layer_clears_stale_state_on_rebuild():
    """A second build must not leave stale edges from the first."""
    g = HierarchicalResearchGraph()
    v = np.array([1.0, 0.0, 0.0, 0.0])
    _make_paper(g, "a", "A", {"methods": v})
    _make_paper(g, "b", "B", {"methods": v})

    g._build_section_similarity_layer(threshold=0.5)
    assert g._backend.edge_count(LAYER_SECTION_SIM) > 0

    # Wipe the section_embeddings — next build should produce zero edges
    for m in g._papers.values():
        m.section_embeddings = {}
    g._build_section_similarity_layer(threshold=0.5)
    assert g._backend.edge_count(LAYER_SECTION_SIM) == 0
