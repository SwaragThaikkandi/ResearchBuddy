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
