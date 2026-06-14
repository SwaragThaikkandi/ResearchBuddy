"""Tests for graph-theoretic distance / reliability measures."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from researchbuddy.core import graph_distance as gd


def _path_graph_adj(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


# ── adjacency_over_ids ─────────────────────────────────────────────────────────

def test_adjacency_over_ids_symmetric_and_ordered():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=0.7)
    G.add_edge("b", "c", weight=0.3)
    A = gd.adjacency_over_ids(G, ["a", "b", "c"])
    assert A.shape == (3, 3)
    assert A[0, 1] == A[1, 0] == 0.7
    assert A[1, 2] == A[2, 1] == 0.3
    assert A[0, 2] == 0.0


def test_adjacency_ignores_unknown_nodes():
    G = nx.Graph()
    G.add_edge("a", "z", weight=1.0)      # z not in id list
    A = gd.adjacency_over_ids(G, ["a", "b"])
    assert A.sum() == 0.0


# ── spectral distance ──────────────────────────────────────────────────────────

def test_spectral_distance_zero_for_identical():
    A = _path_graph_adj(5)
    assert gd.spectral_distance(A, A) == pytest.approx(0.0, abs=1e-9)


def test_spectral_distance_positive_for_different():
    A = _path_graph_adj(5)
    B = _path_graph_adj(5).copy()
    B[0, 4] = B[4, 0] = 1.0               # add an edge → different spectrum
    assert gd.spectral_distance(A, B) > 0


def test_spectral_distance_handles_size_mismatch():
    A = _path_graph_adj(4)
    B = _path_graph_adj(7)
    d = gd.spectral_distance(A, B)        # zero-padded, no exception
    assert d >= 0


# ── DeltaCon ───────────────────────────────────────────────────────────────────

def test_deltacon_identical_is_one():
    A = _path_graph_adj(6)
    assert gd.deltacon_similarity(A, A) == pytest.approx(1.0, abs=1e-9)


def test_deltacon_different_below_one():
    A = _path_graph_adj(6)
    B = _path_graph_adj(6).copy()
    B[0, 5] = B[5, 0] = 1.0
    s = gd.deltacon_similarity(A, B)
    assert 0.0 < s < 1.0


def test_deltacon_requires_equal_shape():
    with pytest.raises(ValueError):
        gd.deltacon_similarity(_path_graph_adj(4), _path_graph_adj(5))


# ── degree KS ──────────────────────────────────────────────────────────────────

def test_degree_ks_zero_for_identical():
    A = _path_graph_adj(8)
    assert gd.degree_ks(A, A) == pytest.approx(0.0, abs=1e-9)


def test_degree_ks_detects_difference():
    A = _path_graph_adj(8)               # mostly degree-2
    star = np.zeros((8, 8))
    star[0, 1:] = star[1:, 0] = 1.0      # one hub, rest degree-1
    assert gd.degree_ks(A, star) > 0


# ── jaccard ────────────────────────────────────────────────────────────────────

def test_jaccard():
    assert gd.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert gd.jaccard(set(), set()) == 1.0
    assert gd.jaccard({"a"}, {"a"}) == 1.0


# ── modularity ─────────────────────────────────────────────────────────────────

def test_modularity_two_clusters():
    G = nx.Graph()
    # two triangles joined by a single bridge → clear community structure
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    m = gd.modularity(G)
    assert m is not None and m > 0.2


# ── cost matrices + Gromov–Wasserstein ─────────────────────────────────────────

def test_cost_from_embeddings_diagonal_zero():
    X = np.random.RandomState(0).randn(5, 8)
    C = gd.cost_from_embeddings(X)
    assert C.shape == (5, 5)
    assert np.allclose(np.diag(C), 0.0, atol=1e-6)


def test_gromov_wasserstein_optional():
    X1 = np.random.RandomState(1).randn(6, 8)
    X2 = np.random.RandomState(2).randn(5, 8)
    C1, C2 = gd.cost_from_embeddings(X1), gd.cost_from_embeddings(X2)
    res = gd.gromov_wasserstein(C1, C2)
    if gd.gw_available():
        assert res is not None
        assert res["distortion"] >= 0
        assert res["coupling"].shape == (6, 5)
    else:
        assert res is None
