"""
Tests for state_manager — focused on the backend-migration logic that
caused real user-visible breakage (Neo4j had stale nodes, no edges).
"""

from __future__ import annotations

from researchbuddy.core.state_manager import _migrate_to_backend
from researchbuddy.core.graph_model import HierarchicalResearchGraph
from researchbuddy.core.graph_backend import (
    NetworkXBackend,
    LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _populate_graph_with_edges(graph: HierarchicalResearchGraph) -> None:
    """Add a few nodes + edges to every relevant layer."""
    b = graph._backend
    for nid, title in [("a", "A"), ("b", "B"), ("c", "C")]:
        b.add_node(nid, LAYER_SEMANTIC, node_type="paper", paper_id=nid, title=title)
        b.add_node(nid, LAYER_CITATION, node_type="paper", paper_id=nid, title=title)
    b.add_edge("a", "b", LAYER_SEMANTIC, etype="semantic", weight=0.8)
    b.add_edge("b", "c", LAYER_SEMANTIC, etype="semantic", weight=0.6)
    b.add_edge("a", "c", LAYER_CITATION, etype="citation", weight=1.0)


# ── Behaviour ────────────────────────────────────────────────────────────────

def test_migration_to_empty_backend_transfers_nodes_and_edges():
    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)
    src_sem_edges = g._backend.edge_count(LAYER_SEMANTIC)
    src_cit_edges = g._backend.edge_count(LAYER_CITATION)

    target = NetworkXBackend()
    _migrate_to_backend(g, target)

    assert target.edge_count(LAYER_SEMANTIC) == src_sem_edges
    assert target.edge_count(LAYER_CITATION) == src_cit_edges
    assert g._backend is target


def test_migration_skipped_when_target_already_has_more_edges():
    """
    If the backend already has equal-or-more edges than the local pickle, the
    backend wins — it's been written to externally and the pickle is stale.
    """
    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)

    # Target with more edges than source
    target = NetworkXBackend()
    for nid in ("a", "b", "c", "d"):
        target.add_node(nid, LAYER_SEMANTIC, node_type="paper",
                        paper_id=nid, title=nid)
    target.add_edge("a", "b", LAYER_SEMANTIC, etype="semantic", weight=0.9)
    target.add_edge("b", "c", LAYER_SEMANTIC, etype="semantic", weight=0.5)
    target.add_edge("c", "d", LAYER_SEMANTIC, etype="semantic", weight=0.4)
    target.add_edge("a", "c", LAYER_CITATION, etype="citation", weight=1.0)
    target.add_node("a", LAYER_CITATION, node_type="paper", paper_id="a", title="A")
    target.add_node("c", LAYER_CITATION, node_type="paper", paper_id="c", title="C")

    target_sem_before = target.edge_count(LAYER_SEMANTIC)
    _migrate_to_backend(g, target)

    # Migration was skipped — target untouched
    assert target.edge_count(LAYER_SEMANTIC) == target_sem_before
    assert g._backend is target


def test_migration_runs_when_target_has_nodes_but_no_edges():
    """
    The actual user-reported bug: Neo4j had nodes but no edges (left over
    from an APOC-bug import). Old code skipped migration because it only
    checked node count > 0. New code re-migrates because edge_count == 0
    while pickle has edges.
    """
    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)
    src_sem_edges = g._backend.edge_count(LAYER_SEMANTIC)
    src_cit_edges = g._backend.edge_count(LAYER_CITATION)
    assert src_sem_edges > 0 and src_cit_edges > 0

    target = NetworkXBackend()
    # Pre-populate target with nodes only — simulates the broken-Neo4j state
    for nid in ("a", "b", "c"):
        target.add_node(nid, LAYER_SEMANTIC, node_type="paper",
                        paper_id=nid, title=nid)
    assert target.edge_count(LAYER_SEMANTIC) == 0

    _migrate_to_backend(g, target)

    assert target.edge_count(LAYER_SEMANTIC) == src_sem_edges
    assert target.edge_count(LAYER_CITATION) == src_cit_edges
    assert g._backend is target


def test_migration_no_op_when_both_empty():
    g = HierarchicalResearchGraph()  # empty
    target = NetworkXBackend()       # empty
    _migrate_to_backend(g, target)
    assert target.edge_count(LAYER_SEMANTIC) == 0
    assert g._backend is target


def test_migration_replaces_stale_layer_completely():
    """
    set_from_networkx clears the layer first, so old data doesn't leak
    through. Verify by giving the target a different edge structure.
    """
    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)

    target = NetworkXBackend()
    # Stale single edge that should be wiped
    for nid in ("a", "b"):
        target.add_node(nid, LAYER_SEMANTIC, node_type="paper",
                        paper_id=nid, title=nid)
    target.add_edge("a", "b", LAYER_SEMANTIC, etype="semantic", weight=0.1)
    assert target.edge_count(LAYER_SEMANTIC) == 1   # < src

    _migrate_to_backend(g, target)
    # All source edges should be present
    assert target.edge_count(LAYER_SEMANTIC) == g._backend.edge_count(LAYER_SEMANTIC)
