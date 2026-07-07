"""Tests for the autotune self-experimentation loop (Karpathy loop)."""

from __future__ import annotations

import random

import numpy as np
import pytest

import researchbuddy.core.graph_model as gm
from researchbuddy.core import autotune as at
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


@pytest.fixture(autouse=True)
def _tmp_audit(monkeypatch, tmp_path):
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "prisma.jsonl")


@pytest.fixture(autouse=True)
def _restore_globals(monkeypatch):
    """Autotune mutates graph_model globals — auto-restore after each test."""
    for attr in ("PPR_DAMPING", "SIMILARITY_THRESHOLD",
                 "WEIGHT_LEARNING_REGULARIZATION", "RATING_HALF_LIFE_DAYS"):
        monkeypatch.setattr(gm, attr, getattr(gm, attr))


def _rated_graph(n=12, dim=384, seed=0):
    """Graph with enough pos/neg ratings for quality_report to be ready."""
    rng = np.random.RandomState(seed)
    base = rng.randn(dim); base /= np.linalg.norm(base)
    g = HierarchicalResearchGraph(alpha=0.6)
    for i in range(n):
        jitter = 0.05 if i % 2 == 0 else 0.6      # likes cluster tight
        v = base + jitter * rng.randn(dim); v /= np.linalg.norm(v)
        m = PaperMeta(paper_id=f"p{i}", title=f"Paper {i}", abstract="x",
                      source="seed", year=2020, embedding=v)
        g.add_paper(m, v)
        g.rate_paper(m.paper_id, 9.0 if i % 2 == 0 else 2.0)
    g.rebuild_hierarchy()
    return g


# ── Objective + proposals ──────────────────────────────────────────────────────

def test_objective_none_without_ratings(graph_with_papers):
    assert at._objective(graph_with_papers) is None     # nothing rated


def test_objective_returns_bounded_float():
    g = _rated_graph()
    score = at._objective(g)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_propose_clamps_to_bounds():
    rng = random.Random(0)
    spec = at.PARAM_SPECS["alpha"]
    for _ in range(50):
        v = at.propose(spec, spec["hi"], rng)
        assert spec["lo"] <= v <= spec["hi"]
        v2 = at.propose(spec, spec["lo"], rng)
        assert spec["lo"] <= v2 <= spec["hi"]


# ── Session mechanics ──────────────────────────────────────────────────────────

def test_session_refuses_without_ratings(graph_with_papers, tmp_path):
    rep = at.run_session(graph_with_papers, rounds=3,
                         log_path=tmp_path / "log.tsv",
                         tuning_path=tmp_path / "tuning.json")
    assert rep["ready"] is False


def test_session_keep_discard_and_persist(tmp_path, monkeypatch):
    """Scripted objective: baseline 0.50, then 0.60 (keep), then 0.55
    (discard → revert). Verifies TSV log, persisted tuning, reverts."""
    g = _rated_graph()
    scores = iter([0.50, 0.60, 0.55])
    monkeypatch.setattr(at, "_objective", lambda graph: next(scores))
    # deterministic: only tune alpha, both rounds
    monkeypatch.setattr(at, "PARAM_SPECS",
                        {"alpha": at.PARAM_SPECS["alpha"]})

    alpha_before = g.alpha
    rep = at.run_session(g, rounds=2, seed=1,
                         log_path=tmp_path / "log.tsv",
                         tuning_path=tmp_path / "tuning.json")
    assert rep["ready"] and rep["baseline"] == 0.50 and rep["best"] == 0.60
    assert list(rep["kept"]) == ["alpha"]
    kept_alpha = rep["kept"]["alpha"]
    # second experiment was discarded → reverted to the KEPT value
    assert g.alpha == pytest.approx(kept_alpha)
    assert g.alpha != alpha_before

    # persisted for the next session
    assert at.load_tuning(tmp_path / "tuning.json")["alpha"] == kept_alpha
    # every experiment logged
    rows = at.read_log(tmp_path / "log.tsv")
    assert [r["status"] for r in rows] == ["baseline", "keep", "discard"]


def test_apply_saved_tuning(tmp_path):
    g = _rated_graph(n=6)
    at.save_tuning({"alpha": 0.8, "unknown_param": 1.0},
                   tmp_path / "tuning.json")
    applied = at.apply_saved_tuning(g, tmp_path / "tuning.json")
    assert applied == ["alpha"]
    assert g.alpha == 0.8


def test_real_session_never_degrades_objective(tmp_path):
    """End-to-end with the REAL objective: best must be >= baseline."""
    g = _rated_graph()
    rep = at.run_session(g, rounds=4, seed=7,
                         log_path=tmp_path / "log.tsv",
                         tuning_path=tmp_path / "tuning.json")
    assert rep["ready"]
    assert rep["best"] >= rep["baseline"]
    final = at._objective(g)
    # graph state must correspond to the best (kept) configuration
    assert final == pytest.approx(rep["best"], abs=0.02)
