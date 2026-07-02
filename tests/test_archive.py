"""Tests for the open archival format (core/archive.py) — the no-lock-in path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from researchbuddy.core import archive as ar
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


def _rich_graph(sample_papers):
    g = HierarchicalResearchGraph(alpha=0.6)
    for i, m in enumerate(sample_papers):
        m.doi = f"10.123/p{i}"
        if i < 2:
            m.user_rating = 7.0 + i
        m.section_embeddings = {"methods": m.embedding}
        g.add_paper(m, m.embedding)
    g._learned_signal_weights = np.arange(11, dtype=float)
    g.rebuild_hierarchy()
    return g


def test_export_import_roundtrip(sample_papers, tmp_path):
    g = _rich_graph(sample_papers)
    n_sem_edges = g.G_semantic.number_of_edges()

    out = ar.export_archive(g, tmp_path / "arch")
    g2 = ar.import_archive(out, rebuild=False)

    assert len(g2.all_papers()) == len(g.all_papers())
    # metadata survives
    m0 = next(m for m in g2.all_papers() if m.doi == "10.123/p0")
    assert m0.user_rating == 7.0
    assert m0.title == sample_papers[0].title
    # embeddings survive bit-exactly (float32)
    orig = next(m for m in g.all_papers() if m.doi == "10.123/p0")
    assert np.allclose(m0.embedding, orig.embedding, atol=1e-6)
    assert "methods" in m0.section_embeddings
    # edges survive
    assert g2.G_semantic.number_of_edges() == n_sem_edges
    # scalar state survives
    assert g2.alpha == g.alpha
    assert np.allclose(g2._learned_signal_weights, g._learned_signal_weights)


def test_archive_is_open_formats(sample_papers, tmp_path):
    """Archive must be readable WITHOUT researchbuddy: plain JSON + NPZ,
    and the NPZ must be pickle-free."""
    g = _rich_graph(sample_papers)
    out = ar.export_archive(g, tmp_path / "arch")

    # JSONL parses with stdlib alone
    lines = (out / "papers.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(l) for l in lines]
    assert len(recs) == 5 and all("title" in r for r in recs)
    # NPZ loads with allow_pickle=False (the default) => no pickle inside
    arr = np.load(out / "embeddings.npz")
    assert arr["embeddings"].shape[0] == 5
    # manifest hashes present for every file
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["files"]) == {"papers.jsonl", "edges.jsonl",
                                 "embeddings.npz", "state.json"}


def test_verify_detects_tampering(sample_papers, tmp_path):
    g = _rich_graph(sample_papers)
    out = ar.export_archive(g, tmp_path / "arch")
    ar.verify_archive(out)                        # clean → no raise

    p = out / "papers.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace(
        "10.123/p0", "10.666/evil"), encoding="utf-8")
    with pytest.raises(ar.ArchiveError, match="hash mismatch"):
        ar.verify_archive(out)
    with pytest.raises(ar.ArchiveError):
        ar.import_archive(out)                    # import refuses tampered data


def test_verify_rejects_missing_manifest(tmp_path):
    with pytest.raises(ar.ArchiveError, match="manifest"):
        ar.verify_archive(tmp_path)


def test_import_ignores_unknown_future_fields(sample_papers, tmp_path):
    """Forward compatibility: a paper record with extra fields still loads."""
    g = _rich_graph(sample_papers)
    out = ar.export_archive(g, tmp_path / "arch")
    p = out / "papers.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["field_from_the_future"] = {"x": 1}
    lines[0] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # re-hash so verification passes (we're testing schema, not tampering)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["files"]["papers.jsonl"] = ar._sha256_file(p)
    (out / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    g2 = ar.import_archive(out, rebuild=False)
    assert len(g2.all_papers()) == 5
