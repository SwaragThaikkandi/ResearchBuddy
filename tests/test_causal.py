"""Tests for causal.py — edge orientation, cycle breaking, anomaly detection."""

from __future__ import annotations

import networkx as nx
import pytest

from researchbuddy.core.causal import break_cycles, orient_edge


class TestOrientEdge:
    def test_direct_citation(self):
        G_cit = nx.DiGraph()
        G_cit.add_edge("A", "B", etype="citation")
        # A cites B → B influenced A → edge: B → A
        src, tgt, conf = orient_edge("A", "B", 2020, 2015, {}, G_cit)
        assert src == "B"
        assert tgt == "A"
        assert conf >= 0.85

    def test_year_gap(self):
        G_cit = nx.DiGraph()
        src, tgt, conf = orient_edge("A", "B", 2010, 2020, {}, G_cit)
        assert src == "A"  # older → newer
        assert tgt == "B"
        assert conf >= 0.50

    def test_same_year(self):
        G_cit = nx.DiGraph()
        src, tgt, conf = orient_edge("A", "B", 2020, 2020, {}, G_cit)
        assert conf < 0.5  # low confidence

    def test_no_year(self):
        G_cit = nx.DiGraph()
        src, tgt, conf = orient_edge("A", "B", None, None, {}, G_cit)
        assert conf <= 0.25


class TestBreakCycles:
    def test_acyclic_unchanged(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", causal_confidence=0.9)
        G.add_edge("B", "C", causal_confidence=0.8)
        n_reversed = break_cycles(G)
        assert n_reversed == 0
        assert nx.is_directed_acyclic_graph(G)

    def test_cycle_broken(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", causal_confidence=0.9)
        G.add_edge("B", "C", causal_confidence=0.8)
        G.add_edge("C", "A", causal_confidence=0.3)  # weakest
        n_reversed = break_cycles(G)
        assert n_reversed >= 1
        assert nx.is_directed_acyclic_graph(G)

    def test_empty_graph(self):
        G = nx.DiGraph()
        assert break_cycles(G) == 0
