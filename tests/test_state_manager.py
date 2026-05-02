"""
Tests for state_manager — focused on the backend-migration logic that
caused real user-visible breakage (Neo4j had stale nodes, no edges).
"""

from __future__ import annotations

import time

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


# ── save() edge-regression guard ─────────────────────────────────────────────

def test_save_proceeds_on_first_save(tmp_path):
    """When the canonical pickle doesn't exist yet, save unconditionally."""
    from researchbuddy.core.state_manager import save

    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)
    pickle_path = tmp_path / "graph.pkl"
    history_dir = tmp_path / "history"

    import researchbuddy.core.state_manager as sm
    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = history_dir
    try:
        save(g, path=pickle_path)
        assert pickle_path.exists()
    finally:
        sm.HISTORY_DIR = orig_history


def test_save_proceeds_when_new_state_has_more_edges(tmp_path):
    """A normal incremental save (more edges than before) just works."""
    from researchbuddy.core.state_manager import save, _total_edges

    g_small = HierarchicalResearchGraph()
    g_small._backend.add_node("a", LAYER_SEMANTIC, node_type="paper",
                              paper_id="a", title="A")
    g_small._backend.add_node("b", LAYER_SEMANTIC, node_type="paper",
                              paper_id="b", title="B")
    g_small._backend.add_edge("a", "b", LAYER_SEMANTIC, etype="semantic", weight=0.5)

    pickle_path = tmp_path / "graph.pkl"

    import researchbuddy.core.state_manager as sm
    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        save(g_small, path=pickle_path)
        small_edges = _total_edges(g_small)

        g_big = HierarchicalResearchGraph()
        _populate_graph_with_edges(g_big)
        save(g_big, path=pickle_path)

        # Should have overwritten with the bigger graph
        import pickle as _pickle
        with open(pickle_path, "rb") as f:
            loaded = _pickle.load(f)
        assert _total_edges(loaded) > small_edges
    finally:
        sm.HISTORY_DIR = orig_history


def test_save_refuses_to_overwrite_with_regressed_state(tmp_path):
    """
    The exact regression we hit in production: a healthy pickle exists,
    then save() is called with a graph that has fewer edges. Refuse to
    overwrite — preserve the user's data.
    """
    from researchbuddy.core.state_manager import save, _total_edges

    g_healthy = HierarchicalResearchGraph()
    _populate_graph_with_edges(g_healthy)

    pickle_path = tmp_path / "graph.pkl"

    import researchbuddy.core.state_manager as sm
    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        save(g_healthy, path=pickle_path)
        healthy_edges = _total_edges(g_healthy)

        # Now an "empty" graph attempts to save
        g_empty = HierarchicalResearchGraph()
        save(g_empty, path=pickle_path)

        # Pickle on disk must STILL be the healthy one
        import pickle as _pickle
        with open(pickle_path, "rb") as f:
            loaded = _pickle.load(f)
        assert _total_edges(loaded) == healthy_edges, \
            "save() should refuse to overwrite with a regressed graph"

        # And the empty state should still appear in history (so we don't
        # lose any forensic evidence of what was attempted)
        snaps = list((tmp_path / "history").glob("graph_*.pkl"))
        assert len(snaps) >= 2  # one for the healthy save, one for the rejected
    finally:
        sm.HISTORY_DIR = orig_history


def test_save_allows_legitimate_partial_state(tmp_path):
    """
    Edge case: when there's nothing on disk yet, even an empty graph is
    allowed (otherwise we'd block fresh-start saves).
    """
    from researchbuddy.core.state_manager import save

    pickle_path = tmp_path / "graph.pkl"

    import researchbuddy.core.state_manager as sm
    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        g_empty = HierarchicalResearchGraph()
        save(g_empty, path=pickle_path)
        assert pickle_path.exists()
    finally:
        sm.HISTORY_DIR = orig_history


# ── Lightweight evolution log + pickle retention ─────────────────────────────

def test_save_appends_evolution_jsonl(tmp_path):
    """Every save should append one line to history/evolution.jsonl."""
    import json
    from researchbuddy.core.state_manager import save

    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)
    pickle_path = tmp_path / "graph.pkl"

    import researchbuddy.core.state_manager as sm
    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        save(g, path=pickle_path)
        save(g, path=pickle_path)   # equal-edges save still appends
        log = sm.HISTORY_DIR / "evolution.jsonl"
        assert log.exists()
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        # Each line should parse as JSON with the expected keys
        for ln in lines:
            d = json.loads(ln)
            assert "total_papers" in d and "combined_edges" in d
    finally:
        sm.HISTORY_DIR = orig_history


def test_save_caps_pickle_retention(tmp_path, monkeypatch):
    """Past STATE_HISTORY_KEEP, oldest pickles should be pruned."""
    import researchbuddy.core.state_manager as sm
    monkeypatch.setattr(sm, "STATE_HISTORY_KEEP", 2)

    g = HierarchicalResearchGraph()
    _populate_graph_with_edges(g)
    pickle_path = tmp_path / "graph.pkl"

    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        for _ in range(5):
            sm.save(g, path=pickle_path)
            time.sleep(0.01)   # so timestamps differ
        snaps = sorted((tmp_path / "history").glob("graph_*.pkl"))
        assert len(snaps) == 2, f"expected 2 pickles after retention, got {len(snaps)}"
        # The JSONL log keeps every entry though
        log = (tmp_path / "history" / "evolution.jsonl")
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 5
    finally:
        sm.HISTORY_DIR = orig_history


def test_compact_history_deletes_pickles_keeps_log(tmp_path, monkeypatch):
    """compact_history() turns N pickles into N JSONL lines and removes them."""
    import researchbuddy.core.state_manager as sm
    monkeypatch.setattr(sm, "STATE_HISTORY_KEEP", 1)

    orig_history = sm.HISTORY_DIR
    sm.HISTORY_DIR = tmp_path / "history"
    try:
        # Fabricate 3 pickle "snapshots" — full graphs, no JSONL entries yet
        sm.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            g = HierarchicalResearchGraph()
            _populate_graph_with_edges(g)
            with open(sm.HISTORY_DIR / f"graph_2026{i:02d}.pkl", "wb") as f:
                import pickle as _p
                _p.dump(g, f)

        report = sm.compact_history()
        assert report["ingested"] == 3
        # With STATE_HISTORY_KEEP=1 we keep the newest pickle and delete two
        assert report["deleted"] == 2

        # JSONL has 3 lines now
        log = sm.HISTORY_DIR / "evolution.jsonl"
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 3

        # Re-running is a no-op (already-ingested files are skipped)
        report2 = sm.compact_history()
        assert report2["ingested"] == 0
    finally:
        sm.HISTORY_DIR = orig_history
