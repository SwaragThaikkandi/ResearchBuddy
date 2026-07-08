"""Tests for the Living Graph (Bayesian scout)."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core import scout as sg
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


@pytest.fixture(autouse=True)
def _tmp_audit(monkeypatch, tmp_path):
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "prisma.jsonl")


@pytest.fixture()
def paths(tmp_path):
    return {"state_path": tmp_path / "scout.json",
            "graph_path": tmp_path / "scout_graph.pkl"}


def _prior_graph(sample_papers):
    g = HierarchicalResearchGraph(alpha=0.6)
    for i, m in enumerate(sample_papers):
        g.add_paper(m, m.embedding)
        if i < 2:
            g.rate_paper(m.paper_id, 9.0)
    return g


def _fake_openalex(base_emb):
    from tests.conftest import _fake_embed

    def _search(q, limit=25):
        out = []
        for i in range(min(limit, 6)):
            m = PaperMeta(paper_id=f"oa_{q[:6]}_{i}",
                          title=f"Found {q[:12]} {i}",
                          abstract="An abstract about the topic.",
                          doi=f"10.9/{q[:4]}{i}", year=2026,
                          cited_by_count=10 * i)
            m.embedding = _fake_embed(m.title)
            out.append(m)
        return out
    return _search


# ── Cycle mechanics ────────────────────────────────────────────────────────────

def test_cycle_acquires_prunes_slates(graph_with_papers, sample_papers,
                                      monkeypatch, paths):
    main = _prior_graph(sample_papers)
    monkeypatch.setattr("researchbuddy.core.searcher.search_openalex",
                        _fake_openalex(sample_papers[0].embedding))

    rep = sg.run_cycle(main, acquire=12, max_size=10, slate_size=5, **paths)
    assert rep["ok"]
    assert rep["acquired"] > 0
    assert rep["size"] <= 10                      # pruned to bound
    assert 0 < len(rep["slate"]) <= 5
    st = sg.load_state(paths["state_path"])
    assert st["cycles"] == 1 and st["slate"]
    # scout graph persisted
    scout = sg.load_scout_graph(paths["graph_path"])
    assert len(scout.all_papers()) == rep["size"]
    # slate never contains papers already in the main graph
    main_titles = {m.title.lower() for m in main.all_papers()}
    assert all(e["title"].lower() not in main_titles for e in rep["slate"])


def test_cycle_refuses_empty_prior(paths, monkeypatch):
    empty = HierarchicalResearchGraph()
    rep = sg.run_cycle(empty, **paths)
    assert rep["ok"] is False


def test_is_due():
    import time
    now = time.time()
    assert not sg.is_due({"enabled": False, "interval_hours": 1,
                          "last_run": 0}, now)
    assert sg.is_due({"enabled": True, "interval_hours": 1,
                      "last_run": 0}, now)
    assert not sg.is_due({"enabled": True, "interval_hours": 24,
                          "last_run": now - 3600}, now)


# ── Evidence → posterior ───────────────────────────────────────────────────────

def test_feedback_updates_posterior_with_tempered_evidence(
        sample_papers, monkeypatch, paths):
    main = _prior_graph(sample_papers)
    monkeypatch.setattr("researchbuddy.core.searcher.search_openalex",
                        _fake_openalex(sample_papers[0].embedding))
    rep = sg.run_cycle(main, acquire=12, max_size=10, slate_size=5, **paths)
    token = rep["slate"][0]["token"]

    prior_ctx = main.context_vector().copy()
    res = sg.apply_feedback(main, token, 9.0, **paths)
    assert res["ok"]
    # imported + rated in the MAIN graph
    m = main.get_paper(token)
    assert m is not None and m.user_rating == 9.0 and m.source == "scout"
    # tempered evidence: abstract-only scout paper carries beta < 1
    assert m.evidence_factor == pytest.approx(0.7)
    assert m.effective_weight == pytest.approx(9.0 * 0.7, rel=0.01)
    # posterior moved (context vector changed)
    post_ctx = main.context_vector()
    assert not np.allclose(prior_ctx, post_ctx)
    # anchor recorded for the next likelihood cycle; slate entry consumed
    st = sg.load_state(paths["state_path"])
    assert token in st["anchors"]
    assert all(e["token"] != token for e in st["slate"])


def test_feedback_low_rating_goes_to_avoid(sample_papers, monkeypatch, paths):
    main = _prior_graph(sample_papers)
    monkeypatch.setattr("researchbuddy.core.searcher.search_openalex",
                        _fake_openalex(sample_papers[0].embedding))
    rep = sg.run_cycle(main, acquire=12, max_size=10, slate_size=5, **paths)
    token = rep["slate"][0]["token"]
    sg.apply_feedback(main, token, 2.0, **paths)
    st = sg.load_state(paths["state_path"])
    assert token in st["avoid"] and token not in st["anchors"]


def test_fulltext_removes_discount(sample_papers):
    m = PaperMeta(paper_id="s1", title="T", abstract="a", source="scout")
    assert m.evidence_factor == pytest.approx(0.7)
    m.filepath = "C:/papers/t.pdf"                 # full text arrived
    assert m.evidence_factor == 1.0
    # non-scout papers were never discounted
    m2 = PaperMeta(paper_id="s2", title="T2", abstract="a", source="discovered")
    assert m2.evidence_factor == 1.0


# ── API surface ────────────────────────────────────────────────────────────────

def test_scout_endpoints(graph_with_papers, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    monkeypatch.setattr(srv, "save_graph", lambda g: None)
    monkeypatch.setattr(sg, "SCOUT_FILE", tmp_path / "scout.json")
    monkeypatch.setattr(sg, "SCOUT_GRAPH_FILE", tmp_path / "scout_graph.pkl")
    client = TestClient(srv.create_app(graph=graph_with_papers,
                                       autosave=False, scheduler=False))

    st = client.get("/api/scout").json()
    assert st["enabled"] is False and st["slate"] == []
    client.post("/api/scout", json={"enabled": True, "interval_hours": 12})
    assert client.get("/api/scout").json()["enabled"] is True
    assert client.post("/api/scout/rate",
                       json={"token": "ghost", "rating": 8}).status_code == 404
    assert client.post("/api/scout/rate",
                       json={"token": "x", "rating": 99}).status_code == 400
