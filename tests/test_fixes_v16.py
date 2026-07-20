"""
Regressions for the v0.16.0 audit fixes.

C2  title/DOI-collision rating crash (KeyError from rate_paper)
H1  surveillance recall hole (publication_date -> created_date)
H2  scout pruning the user's confirmed anchors
H3  cross-origin (CSRF) form posts into the local UI
#1  equation signal (GROBID equations were extracted but never used)
"""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.config import SCORED_SECTION_TYPES
from researchbuddy.core.graph_model import (
    HierarchicalResearchGraph, PaperMeta, EXTRA_SIGNAL_TYPES,
)


# ── C2: collision-safe add-then-rate ──────────────────────────────────────────

def test_resolve_paper_id_by_id_doi_title(graph_with_papers):
    g = graph_with_papers
    resident = g.all_papers()[0]
    resident.doi = "10.1/known"

    assert g.resolve_paper_id(resident) == resident.paper_id
    # same DOI, different id
    by_doi = PaperMeta(paper_id="other_id", title="Totally Different",
                       abstract="x", doi="10.1/KNOWN")
    assert g.resolve_paper_id(by_doi) == resident.paper_id
    # same title, different id + no DOI
    by_title = PaperMeta(paper_id="other_id2", title=resident.title.upper(),
                         abstract="x")
    assert g.resolve_paper_id(by_title) == resident.paper_id
    # genuinely new
    assert g.resolve_paper_id(
        PaperMeta(paper_id="new", title="Brand New Thing", abstract="x")) is None


def test_add_or_get_prevents_rate_keyerror(graph_with_papers):
    """The exact crash: a candidate duplicating an existing title arrives
    with a different paper_id; add_paper returns False; rating its own id
    used to raise KeyError."""
    g = graph_with_papers
    resident = g.all_papers()[0]
    dup = PaperMeta(paper_id="dup_id_999", title=resident.title,
                    abstract="x", embedding=resident.embedding)

    assert g.add_paper(dup, dup.embedding) is False        # collision
    with pytest.raises(KeyError):
        g.rate_paper(dup.paper_id, 8.0)                    # the old bug

    pid = g.add_or_get(dup, dup.embedding)                 # the fix
    assert pid == resident.paper_id
    g.rate_paper(pid, 8.0)
    assert g.get_paper(pid).user_rating == 8.0
    assert len(g.all_papers()) == 5                        # no duplicate node


def test_add_or_get_adds_new_paper(graph_with_papers):
    g = graph_with_papers
    before = len(g.all_papers())
    fresh = PaperMeta(paper_id="fresh_1", title="A Genuinely New Paper",
                      abstract="x", embedding=g.all_papers()[0].embedding)
    pid = g.add_or_get(fresh, fresh.embedding)
    assert pid == "fresh_1"
    assert len(g.all_papers()) == before + 1


def test_rate_endpoint_survives_title_collision(client_factory=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    from tests.conftest import _fake_embed

    g = HierarchicalResearchGraph(alpha=0.6)
    resident = PaperMeta(paper_id="res_1", title="Shared Title",
                         abstract="a", source="seed",
                         embedding=_fake_embed("Shared Title"))
    g.add_paper(resident, resident.embedding)
    app = srv.create_app(graph=g, autosave=False, scheduler=False)
    client = TestClient(app)
    st = app.state.rb
    # a candidate with the same title but a different id (two sources)
    dup = PaperMeta(paper_id="cand_dup", title="Shared Title", abstract="a",
                    embedding=_fake_embed("Shared Title"))
    st.remember([dup])

    r = client.post("/api/rate", json={"token": "cand_dup", "rating": 9})
    assert r.status_code == 200, r.text
    assert r.json()["paper_id"] == "res_1"
    assert g.get_paper("res_1").user_rating == 9.0


# ── H1: surveillance recall ───────────────────────────────────────────────────

def test_watcher_filters_on_created_date(monkeypatch):
    """Must filter on indexing date, not publication date, or papers
    published-before/indexed-after a check are missed forever."""
    import researchbuddy.core.watcher as wt
    seen = {}

    class _R:
        status_code = 200

        @staticmethod
        def json():
            return {"results": []}

    def _fake_get(url, params=None, headers=None, timeout=None):
        seen["filter"] = params["filter"]
        return _R()

    monkeypatch.setattr(wt.requests, "get", _fake_get)
    wt._search_since("topic", "2026-01-01")
    assert "from_created_date:2026-01-01" in seen["filter"]
    assert "from_publication_date" not in seen["filter"]


# ── H2: anchors survive pruning ───────────────────────────────────────────────

def test_prune_protects_anchors_and_rated(monkeypatch):
    from researchbuddy.core import scout as sg
    from tests.conftest import _fake_embed

    g = HierarchicalResearchGraph()
    for i in range(10):
        m = PaperMeta(paper_id=f"s{i}", title=f"Scout {i}", abstract="x",
                      source="scout", embedding=_fake_embed(f"Scout {i}"))
        g.add_paper(m, m.embedding)
    g.rate_paper("s9", 9.0)                 # user-rated inside the scout

    # worst possible scores for the ones we must keep
    keep_scores = {f"s{i}": 1.0 for i in range(10)}
    keep_scores["s0"] = 0.0                 # anchor with terrible score
    keep_scores["s9"] = 0.0                 # rated with terrible score

    pruned = sg._prune(g, keep_scores, max_size=3, protect={"s0"})
    ids = {m.paper_id for m in pruned.all_papers()}
    assert "s0" in ids, "confirmed anchor was pruned"
    assert "s9" in ids, "user-rated scout paper was pruned"


# ── H3: cross-origin guard ────────────────────────────────────────────────────

def test_csrf_blocks_foreign_origin(graph_with_papers):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    client = TestClient(srv.create_app(graph=graph_with_papers,
                                       autosave=False, scheduler=False))
    # a malicious page the user merely visited
    r = client.post("/api/rebuild", json={},
                    headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    # the real UI is allowed
    ok = client.post("/api/rebuild", json={},
                     headers={"origin": "http://127.0.0.1:8230"})
    assert ok.status_code == 200
    # no Origin at all (curl / CLI) still allowed
    assert client.post("/api/rebuild", json={}).status_code == 200


def test_csrf_allows_get_cross_origin(graph_with_papers):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    client = TestClient(srv.create_app(graph=graph_with_papers,
                                       autosave=False, scheduler=False))
    r = client.get("/api/stats", headers={"origin": "https://evil.example"})
    assert r.status_code == 200          # reads are not state-changing


# ── #1: equation signal ───────────────────────────────────────────────────────

def _eq_graph(n=20, dim=384, seed=0):
    """Full-text-like graph: every paper carries an equation embedding."""
    rng = np.random.RandomState(seed)
    base = rng.randn(dim); base /= np.linalg.norm(base)
    eq_base = rng.randn(dim); eq_base /= np.linalg.norm(eq_base)
    g = HierarchicalResearchGraph(alpha=0.6)
    for i in range(n):
        v = base + 0.05 * rng.randn(dim); v /= np.linalg.norm(v)
        m = PaperMeta(paper_id=f"p{i}", title=f"Paper {i}", abstract="x",
                      source="seed", year=2022, embedding=v)
        e = eq_base + 0.05 * rng.randn(dim); e /= np.linalg.norm(e)
        m.equations = [f"dx = v dt + s dW  # {i}"]
        m.equation_embedding = e
        g.add_paper(m, v)
        if i < 6:
            g.rate_paper(m.paper_id, 9.0)
    g.rebuild_hierarchy()
    return g, base, eq_base


def test_signal_vector_includes_equation():
    g, base, eq_base = _eq_graph()
    sig = g._extract_signals(g.all_papers()[0])
    assert len(sig) == 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)
    assert "equation" in EXTRA_SIGNAL_TYPES


def test_equation_signal_rewards_similar_math():
    g, base, eq_base = _eq_graph()
    rng = np.random.RandomState(99)
    same_math = PaperMeta(paper_id="c1", title="Similar math", abstract="x",
                          year=2026, embedding=base)
    same_math.equation_embedding = eq_base
    diff = rng.randn(len(eq_base)); diff /= np.linalg.norm(diff)
    other_math = PaperMeta(paper_id="c2", title="Other math", abstract="x",
                           year=2026, embedding=base)
    other_math.equation_embedding = diff

    s_same = g._equation_signal(same_math)
    s_other = g._equation_signal(other_math)
    assert s_same is not None and s_other is not None
    assert s_same > s_other
    # and it moves the fused score in the same direction
    assert g.score_candidate(same_math) > g.score_candidate(other_math)


def test_equation_signal_masked_when_absent():
    """Abstract-only candidates (no equations) must not be penalised — the
    same masking discipline as sections."""
    g, base, eq_base = _eq_graph()
    abstract_only = PaperMeta(paper_id="c3", title="Abstract only",
                              abstract="x", year=2026, embedding=base)
    assert g._equation_signal(abstract_only) is None
    assert g.score_candidate(abstract_only) >= 0.30      # not floored to ~0


def test_embed_equations_sets_and_clears(monkeypatch):
    from tests.conftest import _fake_embed
    import researchbuddy.core.graph_model as gm
    monkeypatch.setattr(gm, "embed", _fake_embed)
    g = HierarchicalResearchGraph()
    m = PaperMeta(paper_id="e1", title="T", abstract="a",
                  equations=["a = b + c", "dx/dt = -x"])
    g.embed_equations(m)
    assert m.equation_embedding is not None
    m2 = PaperMeta(paper_id="e2", title="T2", abstract="a", equations=[])
    g.embed_equations(m2)
    assert m2.equation_embedding is None


def test_old_pickle_gets_equation_fields():
    import pickle
    g, _, _ = _eq_graph(n=6)
    victim = g.all_papers()[0]
    del victim.__dict__["equation_embedding"]      # simulate an old pickle
    del g.__dict__["_equation_context_vec"]
    g2 = pickle.loads(pickle.dumps(g))
    assert hasattr(g2.all_papers()[0], "equation_embedding")
    assert hasattr(g2, "_equation_context_vec")
