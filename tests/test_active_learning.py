"""
Tests for score uncertainty (bootstrap weight ensemble) and the
active-learning rating queue — the acquisition function that completes the
Bayesian loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.config import SCORED_SECTION_TYPES
from researchbuddy.core.graph_model import (
    HierarchicalResearchGraph, PaperMeta, EXTRA_SIGNAL_TYPES,
)


def _learned_graph(n=24, dim=384, seed=0):
    """Graph with enough well-separated ratings to fit weights + ensemble."""
    rng = np.random.RandomState(seed)
    liked = rng.randn(dim); liked /= np.linalg.norm(liked)
    disliked = rng.randn(dim); disliked /= np.linalg.norm(disliked)
    g = HierarchicalResearchGraph(alpha=0.6)
    for i in range(n):
        base = liked if i % 2 == 0 else disliked
        v = base + 0.08 * rng.randn(dim); v /= np.linalg.norm(v)
        m = PaperMeta(paper_id=f"p{i}", title=f"Paper {i}", abstract="x",
                      source="seed", year=2022, embedding=v)
        g.add_paper(m, v)
        g.rate_paper(m.paper_id, 9.0 if i % 2 == 0 else 2.0)
    g.rebuild_hierarchy()
    g.learn_signal_weights()
    return g, liked, disliked


# ── Ensemble fit ──────────────────────────────────────────────────────────────

def test_learning_fits_ensemble():
    g, _, _ = _learned_graph()
    ens = g._weight_ensemble
    assert ens is not None
    assert ens.shape[0] >= 2
    assert ens.shape[1] == 7 + len(SCORED_SECTION_TYPES) + len(EXTRA_SIGNAL_TYPES)
    # replicas differ (bootstrap actually resampled)
    assert float(np.std(ens, axis=0).sum()) > 0.0


def test_no_ensemble_below_threshold(graph_with_papers):
    g = graph_with_papers
    for i, m in enumerate(g.all_papers()[:4]):
        g.rate_paper(m.paper_id, 9.0 if i % 2 == 0 else 2.0)
    g.learn_signal_weights()                       # too few → no ensemble
    assert getattr(g, "_weight_ensemble", None) is None


# ── Uncertainty ───────────────────────────────────────────────────────────────

def test_score_with_uncertainty_shape_and_bounds():
    g, liked, _ = _learned_graph()
    cand = PaperMeta(paper_id="c1", title="Cand", abstract="x", year=2026,
                     embedding=liked)
    score, sigma = g.score_with_uncertainty(cand)
    assert 0.0 <= score <= 1.0
    assert sigma >= 0.0
    # matches the plain scorer
    assert score == pytest.approx(g.score_candidate(cand))


def test_uncertainty_zero_without_ensemble(graph_with_papers):
    g = graph_with_papers
    cand = PaperMeta(paper_id="c", title="C", abstract="x",
                     embedding=g.all_papers()[0].embedding)
    score, sigma = g.score_with_uncertainty(cand)
    assert sigma == 0.0            # "no error bar available", not "certain"


def test_ambiguous_paper_is_more_uncertain_than_prototype():
    """A paper halfway between the liked and disliked clusters should carry
    more ensemble spread than one sitting on the liked prototype."""
    g, liked, disliked = _learned_graph()
    proto = PaperMeta(paper_id="proto", title="Prototype", abstract="x",
                      year=2026, embedding=liked)
    mid_v = liked + disliked
    mid_v = mid_v / np.linalg.norm(mid_v)
    mid = PaperMeta(paper_id="mid", title="Ambiguous", abstract="x",
                    year=2026, embedding=mid_v)
    _, sigma_proto = g.score_with_uncertainty(proto)
    _, sigma_mid = g.score_with_uncertainty(mid)
    assert sigma_mid >= sigma_proto


# ── Acquisition + queue ───────────────────────────────────────────────────────

def test_acquisition_combines_uncertainty_and_relevance():
    g, liked, _ = _learned_graph()
    cand = PaperMeta(paper_id="a1", title="A", abstract="x", year=2026,
                     embedding=liked)
    score, sigma = g.score_with_uncertainty(cand)
    acq = g.acquisition_score(cand)
    assert acq == pytest.approx(sigma * (max(score, 0.0) ** 0.5), rel=1e-6)
    # zero uncertainty ⇒ nothing to learn ⇒ zero acquisition
    g._weight_ensemble = None
    assert g.acquisition_score(cand) == 0.0


def test_rating_queue_only_unrated_and_ordered():
    g, liked, _ = _learned_graph()
    rng = np.random.RandomState(5)
    for i in range(6):                       # add unrated candidates
        v = liked + 0.3 * rng.randn(len(liked)); v /= np.linalg.norm(v)
        m = PaperMeta(paper_id=f"u{i}", title=f"Unrated {i}", abstract="x",
                      year=2026, embedding=v)
        g.add_paper(m, v)

    queue = g.rating_queue(n=5)
    assert len(queue) == 5
    ids = [m.paper_id for m, _, _, _ in queue]
    assert all(i.startswith("u") for i in ids), "rated papers must not appear"
    accs = [a for _, _, _, a in queue]
    assert accs == sorted(accs, reverse=True), "must rank by information gain"


def test_rating_queue_accepts_explicit_candidates():
    g, liked, _ = _learned_graph()
    cands = [PaperMeta(paper_id=f"x{i}", title=f"X{i}", abstract="x",
                       year=2026, embedding=liked) for i in range(3)]
    queue = g.rating_queue(n=2, candidates=cands)
    assert len(queue) == 2
    assert {m.paper_id for m, _, _, _ in queue} <= {"x0", "x1", "x2"}


# ── Migration ─────────────────────────────────────────────────────────────────

def test_stale_ensemble_dropped_on_unpickle():
    """A bootstrap replica must match the current feature space exactly, so a
    shorter ensemble from an older schema is dropped, not padded."""
    import pickle
    g, _, _ = _learned_graph(n=12)
    g._weight_ensemble = np.ones((4, 5))          # wrong width
    g2 = pickle.loads(pickle.dumps(g))
    assert g2._weight_ensemble is None
    # and scoring still works (falls back to no error bar)
    cand = PaperMeta(paper_id="z", title="Z", abstract="x",
                     embedding=g2.all_papers()[0].embedding)
    assert g2.score_with_uncertainty(cand)[1] == 0.0


# ── API ───────────────────────────────────────────────────────────────────────

def test_rating_queue_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv

    g, liked, _ = _learned_graph()
    m = PaperMeta(paper_id="unrated_1", title="Unrated one", abstract="x",
                  year=2026, embedding=liked)
    g.add_paper(m, m.embedding)
    client = TestClient(srv.create_app(graph=g, autosave=False,
                                       scheduler=False))
    body = client.get("/api/rating_queue?n=3").json()
    assert body["ensemble_ready"] is True
    assert body["queue"]
    row = body["queue"][0]
    assert {"token", "score", "sigma", "acquisition"} <= set(row)
    assert row["token"] == "unrated_1"


def test_search_results_carry_sigma(graph_with_papers, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    from tests.conftest import _fake_embed
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    fake = [PaperMeta(paper_id="cand_a", title="Candidate A",
                      abstract="Decision making.", year=2026,
                      embedding=_fake_embed(graph_with_papers.all_papers()[0].title))]
    monkeypatch.setattr("researchbuddy.core.searcher.find_candidates",
                        lambda g, extra_keywords=None, query=None: (fake, None))
    client = TestClient(srv.create_app(graph=graph_with_papers,
                                       autosave=False, scheduler=False))
    body = client.post("/api/search", json={"n": 3}).json()
    assert body["results"]
    assert "sigma" in body["results"][0]      # key present even when None
