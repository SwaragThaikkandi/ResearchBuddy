"""Tests for graph capsules: export, IO roundtrip, privacy, and merge."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core import capsule as cap
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


def _set_dois(graph):
    for i, m in enumerate(graph.all_papers()):
        m.doi = f"10.123/paper{i}"
    return graph


# ── Export / privacy ───────────────────────────────────────────────────────────

def test_export_private_hides_identifiers(graph_with_papers):
    _set_dois(graph_with_papers)
    capsule = cap.export_capsule(graph_with_papers)   # defaults: all off
    assert capsule.stats["n_papers"] == 5
    assert capsule.embeddings.shape == (5, 384)
    for node in capsule.nodes:
        assert "doi" not in node
        assert "title" not in node
        assert "rating" not in node


def test_export_with_identifiers_and_ratings(graph_with_papers):
    g = _set_dois(graph_with_papers)
    g.all_papers()[0].user_rating = 8.0
    capsule = cap.export_capsule(g, share_identifiers=True, share_ratings=True)
    assert capsule.nodes[0]["doi"] == "10.123/paper0"
    assert capsule.nodes[0]["title"]
    assert capsule.nodes[0]["rating"] == 8.0
    assert capsule.doi_set() == {f"10.123/paper{i}" for i in range(5)}


def test_export_drops_thought_nodes(graph_with_papers):
    g = graph_with_papers
    thought = PaperMeta(paper_id="t1", title="My private draft",
                        abstract="x" * 50, kind="essay",
                        embedding=np.ones(384) / np.sqrt(384))
    g._papers["t1"] = thought
    capsule = cap.export_capsule(g, share_identifiers=True)
    titles = [n.get("title") for n in capsule.nodes]
    assert "My private draft" not in titles
    assert capsule.stats["n_papers"] == 5


# ── IO roundtrip ───────────────────────────────────────────────────────────────

def test_dumps_loads_roundtrip_in_memory(graph_with_papers):
    g = _set_dois(graph_with_papers)
    capsule = cap.export_capsule(g, share_identifiers=True)
    blob = cap.dumps_capsule(capsule)
    assert isinstance(blob, bytes) and blob[:2] == b"PK"   # zip magic
    back = cap.loads_capsule(blob)
    assert back.doi_set() == capsule.doi_set()
    assert back.stats == capsule.stats
    assert np.allclose(back.embeddings, capsule.embeddings, atol=1e-5)


def test_write_load_roundtrip(graph_with_papers, tmp_path):
    g = _set_dois(graph_with_papers)
    capsule = cap.export_capsule(g, share_identifiers=True)
    path = cap.write_capsule(capsule, tmp_path / "g")
    assert path.suffix == ".rbcapsule"

    loaded = cap.load_capsule(path)
    assert loaded.version == capsule.version
    assert loaded.stats == capsule.stats
    assert loaded.doi_set() == capsule.doi_set()
    assert np.allclose(loaded.embeddings, capsule.embeddings, atol=1e-5)


# ── Merge ──────────────────────────────────────────────────────────────────────

def _graph_from(papers):
    g = HierarchicalResearchGraph(alpha=0.6)
    for m in papers:
        g.add_paper(m, m.embedding)
    return g


def test_merge_imports_new_identified_papers(sample_papers):
    # Peer A has all 5; I (B) have only the first 2.
    for i, m in enumerate(sample_papers):
        m.doi = f"10.123/paper{i}"
    gA = _graph_from(sample_papers)
    gB = _graph_from(sample_papers[:2])
    for i, m in enumerate(gB.all_papers()):
        m.doi = f"10.123/paper{i}"

    capsule = cap.export_capsule(gA, share_identifiers=True)
    before = len(gB.all_papers())
    report = cap.merge_capsule(gB, capsule)

    assert report.shared_by_doi == 2
    assert report.imported == 3
    assert len(gB.all_papers()) == before + 3
    assert report.jaccard_doi == pytest.approx(2 / 5)
    assert report.spectral_distance is not None
    assert report.modularity_self is not None


def test_merge_private_capsule_does_not_import(sample_papers):
    for i, m in enumerate(sample_papers):
        m.doi = f"10.123/paper{i}"
    gA = _graph_from(sample_papers)
    gB = _graph_from(sample_papers[:1])

    capsule = cap.export_capsule(gA)   # private: no identifiers
    before = len(gB.all_papers())
    report = cap.merge_capsule(gB, capsule)

    assert report.imported == 0
    assert len(gB.all_papers()) == before
    # paper 0 is identical embedding → matched structurally; others novel
    assert report.shared_by_embedding + report.novel_regions == 5
    assert report.shared_by_embedding >= 1
    assert any("Private capsule" in n for n in report.notes)


# ── Untrusted-capsule validation ───────────────────────────────────────────────

def test_loads_rejects_garbage_bytes():
    import pytest as _pytest
    with _pytest.raises(cap.CapsuleError):
        cap.loads_capsule(b"this is not a zip")


def test_loads_rejects_newer_version(graph_with_papers):
    capsule = cap.export_capsule(graph_with_papers)
    capsule.version = 999
    blob = cap.dumps_capsule(capsule)
    import pytest as _pytest
    with _pytest.raises(cap.CapsuleError, match="newer than supported"):
        cap.loads_capsule(blob)


def test_loads_rejects_out_of_range_idx(graph_with_papers):
    capsule = cap.export_capsule(graph_with_papers, share_identifiers=True)
    capsule.nodes[0]["idx"] = 10_000          # beyond embedding rows
    blob = cap.dumps_capsule(capsule)
    import pytest as _pytest
    with _pytest.raises(cap.CapsuleError, match="idx"):
        cap.loads_capsule(blob)


def test_loads_rejects_bad_edges(graph_with_papers):
    capsule = cap.export_capsule(graph_with_papers)
    capsule.edges_sem = [(0, 99_999, 1.0)]     # endpoint out of range
    blob = cap.dumps_capsule(capsule)
    import pytest as _pytest
    with _pytest.raises(cap.CapsuleError, match="endpoints"):
        cap.loads_capsule(blob)


def test_loads_rejects_nan_embeddings(graph_with_papers):
    capsule = cap.export_capsule(graph_with_papers)
    capsule.embeddings[0, 0] = np.nan
    blob = cap.dumps_capsule(capsule)
    import pytest as _pytest
    with _pytest.raises(cap.CapsuleError, match="NaN"):
        cap.loads_capsule(blob)


def test_merge_dim_mismatch_imports_metadata_only(sample_papers):
    """Peer on a different embedding model must not poison the local graph."""
    for i, m in enumerate(sample_papers):
        m.doi = f"10.123/paper{i}"
    gA = _graph_from(sample_papers)
    capsule = cap.export_capsule(gA, share_identifiers=True)
    # Simulate a peer with 128-dim embeddings.
    capsule.embeddings = np.random.RandomState(0)\
        .randn(capsule.embeddings.shape[0], 128).astype(np.float32)
    capsule.centroids = np.zeros((0, 128), dtype=np.float32)

    gB2 = _graph_from(sample_papers[:1])   # one local paper fixes the local dim
    for i, m in enumerate(gB2.all_papers()):
        m.doi = "10.999/other"
    report = cap.merge_capsule(gB2, capsule)

    assert any("dim mismatch" in n.lower() for n in report.notes)
    assert report.imported == 5
    # Imported papers are metadata stubs — no foreign-dim embeddings inside.
    for m in gB2.all_papers():
        if m.source == "capsule":
            assert m.embedding is None
    # NN matching was skipped, not garbage-matched.
    assert report.shared_by_embedding == 0


def test_merge_deltacon_on_shared_subgraph(sample_papers):
    for i, m in enumerate(sample_papers):
        m.doi = f"10.123/paper{i}"
    gA = _graph_from(sample_papers)
    gA.rebuild_hierarchy()                        # gives A semantic edges
    gB = _graph_from(sample_papers)
    for i, m in enumerate(gB.all_papers()):
        m.doi = f"10.123/paper{i}"
    gB.rebuild_hierarchy()

    capsule = cap.export_capsule(gA, share_identifiers=True)
    report = cap.merge_capsule(gB, capsule)
    # 5 shared DOIs ⇒ DeltaCon computed over the shared subgraph.
    assert report.deltacon_similarity is not None
    assert 0.0 < report.deltacon_similarity <= 1.0
