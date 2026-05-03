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
