"""
Tests for the graph backend abstraction layer.

Covers NetworkXBackend; Neo4jBackend tests are skipped if Neo4j is not available.
"""

import pytest
import networkx as nx

from researchbuddy.core.graph_backend import (
    NetworkXBackend,
    LAYER_SEMANTIC,
    LAYER_CITATION,
    LAYER_COMBINED,
    LAYER_CAUSAL,
)


@pytest.fixture
def backend():
    return NetworkXBackend()


class TestNetworkXBackendNodes:
    def test_add_and_has_node(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC, level=0, node_type="paper", weight=5.0)
        assert backend.has_node("p1", LAYER_SEMANTIC)
        assert not backend.has_node("p1", LAYER_CITATION)

    def test_node_attrs(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC, level=0, weight=5.0)
        assert backend.get_node_attr("p1", "weight", LAYER_SEMANTIC) == 5.0
        backend.set_node_attr("p1", "weight", 8.0, LAYER_SEMANTIC)
        assert backend.get_node_attr("p1", "weight", LAYER_SEMANTIC) == 8.0

    def test_node_count(self, backend):
        assert backend.node_count(LAYER_SEMANTIC) == 0
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        assert backend.node_count(LAYER_SEMANTIC) == 2

    def test_add_nodes_batch(self, backend):
        nodes = [
            ("p1", {"level": 0, "node_type": "paper", "weight": 5.0}),
            ("p2", {"level": 0, "node_type": "paper", "weight": 3.0}),
        ]
        backend.add_nodes_batch(nodes, LAYER_SEMANTIC)
        assert backend.has_node("p1", LAYER_SEMANTIC)
        assert backend.has_node("p2", LAYER_SEMANTIC)
        assert backend.node_count(LAYER_SEMANTIC) == 2


class TestNetworkXBackendEdges:
    def test_add_and_has_edge(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC, weight=0.8, etype="semantic")
        assert backend.has_edge("p1", "p2", LAYER_SEMANTIC)
        assert not backend.has_edge("p2", "p1", LAYER_SEMANTIC)
        assert not backend.has_edge("p1", "p2", LAYER_CITATION)

    def test_edge_attrs(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC, weight=0.8)
        assert backend.get_edge_attr("p1", "p2", "weight", LAYER_SEMANTIC) == 0.8
        backend.set_edge_attr("p1", "p2", "weight", 0.9, LAYER_SEMANTIC)
        assert backend.get_edge_attr("p1", "p2", "weight", LAYER_SEMANTIC) == 0.9

    def test_edge_count(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        assert backend.edge_count(LAYER_SEMANTIC) == 0
        backend.add_edge("p1", "p2", LAYER_SEMANTIC)
        assert backend.edge_count(LAYER_SEMANTIC) == 1

    def test_remove_edge(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC)
        backend.remove_edge("p1", "p2", LAYER_SEMANTIC)
        assert not backend.has_edge("p1", "p2", LAYER_SEMANTIC)

    def test_edges_data(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC, weight=0.5)
        edges = list(backend.edges_data(LAYER_SEMANTIC))
        assert len(edges) == 1
        u, v, data = edges[0]
        assert u == "p1"
        assert v == "p2"
        assert data["weight"] == 0.5

    def test_degree(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_node("p3", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC)
        backend.add_edge("p3", "p1", LAYER_SEMANTIC)
        assert backend.degree("p1", LAYER_SEMANTIC) == 2

    def test_add_edges_batch(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_node("p3", LAYER_SEMANTIC)
        edges = [
            ("p1", "p2", {"weight": 0.5}),
            ("p2", "p3", {"weight": 0.7}),
        ]
        backend.add_edges_batch(edges, LAYER_SEMANTIC)
        assert backend.edge_count(LAYER_SEMANTIC) == 2


class TestNetworkXBackendAlgorithms:
    def test_shortest_path(self, backend):
        for pid in ("p1", "p2", "p3"):
            backend.add_node(pid, LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC)
        backend.add_edge("p2", "p3", LAYER_SEMANTIC)
        path = backend.shortest_path("p1", "p3", LAYER_SEMANTIC)
        assert path == ["p1", "p2", "p3"]

    def test_shortest_path_no_path(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        assert backend.shortest_path("p1", "p2", LAYER_SEMANTIC) is None

    def test_pagerank(self, backend):
        for pid in ("p1", "p2", "p3"):
            backend.add_node(pid, LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC)
        backend.add_edge("p2", "p3", LAYER_SEMANTIC)
        pr = backend.pagerank(LAYER_SEMANTIC)
        assert len(pr) == 3
        assert all(v > 0 for v in pr.values())

    def test_is_dag(self, backend):
        for pid in ("p1", "p2", "p3"):
            backend.add_node(pid, LAYER_CAUSAL)
        backend.add_edge("p1", "p2", LAYER_CAUSAL)
        backend.add_edge("p2", "p3", LAYER_CAUSAL)
        assert backend.is_dag(LAYER_CAUSAL) is True

    def test_is_not_dag(self, backend):
        for pid in ("p1", "p2"):
            backend.add_node(pid, LAYER_CAUSAL)
        backend.add_edge("p1", "p2", LAYER_CAUSAL)
        backend.add_edge("p2", "p1", LAYER_CAUSAL)
        assert backend.is_dag(LAYER_CAUSAL) is False

    def test_find_cycle(self, backend):
        for pid in ("p1", "p2"):
            backend.add_node(pid, LAYER_CAUSAL)
        backend.add_edge("p1", "p2", LAYER_CAUSAL)
        backend.add_edge("p2", "p1", LAYER_CAUSAL)
        cycle = backend.find_cycle(LAYER_CAUSAL)
        assert cycle is not None

    def test_find_no_cycle(self, backend):
        backend.add_node("p1", LAYER_CAUSAL)
        backend.add_node("p2", LAYER_CAUSAL)
        backend.add_edge("p1", "p2", LAYER_CAUSAL)
        assert backend.find_cycle(LAYER_CAUSAL) is None


class TestNetworkXBackendLayers:
    def test_clear_layer(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_edge("p1", "p1", LAYER_SEMANTIC)
        backend.clear_layer(LAYER_SEMANTIC)
        assert backend.node_count(LAYER_SEMANTIC) == 0
        assert backend.edge_count(LAYER_SEMANTIC) == 0

    def test_layers_are_independent(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p1", LAYER_CITATION)
        assert backend.has_node("p1", LAYER_SEMANTIC)
        assert backend.has_node("p1", LAYER_CITATION)
        assert not backend.has_node("p1", LAYER_CAUSAL)

    def test_to_networkx(self, backend):
        backend.add_node("p1", LAYER_SEMANTIC)
        backend.add_node("p2", LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC, weight=0.5)
        G = backend.to_networkx(LAYER_SEMANTIC)
        assert isinstance(G, nx.DiGraph)
        assert G.has_node("p1")
        assert G.has_edge("p1", "p2")

    def test_set_from_networkx(self, backend):
        G = nx.DiGraph()
        G.add_node("p1", level=0)
        G.add_node("p2", level=0)
        G.add_edge("p1", "p2", weight=0.7)
        backend.set_from_networkx(LAYER_SEMANTIC, G)
        assert backend.has_node("p1", LAYER_SEMANTIC)
        assert backend.has_edge("p1", "p2", LAYER_SEMANTIC)

    def test_subgraph_edges(self, backend):
        for pid in ("p1", "p2", "p3"):
            backend.add_node(pid, LAYER_SEMANTIC)
        backend.add_edge("p1", "p2", LAYER_SEMANTIC, weight=0.5)
        backend.add_edge("p2", "p3", LAYER_SEMANTIC, weight=0.6)
        backend.add_edge("p1", "p3", LAYER_SEMANTIC, weight=0.7)
        # Subgraph of just p1, p2 should only have p1->p2
        edges = list(backend.subgraph_edges({"p1", "p2"}, LAYER_SEMANTIC))
        assert len(edges) == 1
        assert edges[0][0] == "p1" and edges[0][1] == "p2"


class TestNetworkXBackendProperties:
    def test_backend_name(self, backend):
        assert backend.backend_name == "NetworkX"

    def test_sync_is_noop(self, backend):
        backend.sync()  # should not raise

    def test_close_is_noop(self, backend):
        backend.close()  # should not raise
