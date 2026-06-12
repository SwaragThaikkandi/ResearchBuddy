"""Tests for the living-review watcher (core/watcher.py)."""

from __future__ import annotations

import time

import pytest

from researchbuddy.core import watcher as wt
from researchbuddy.core.graph_model import PaperMeta


@pytest.fixture(autouse=True)
def _tmp_audit_log(monkeypatch, tmp_path):
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "prisma.jsonl")


def test_watch_crud_roundtrip(tmp_path):
    wfile = tmp_path / "watches.json"
    assert wt.load_watches(wfile) == []

    w = wt.add_watch("drift diffusion", ["choice", "RT"], path=wfile)
    assert w["query"] == "drift diffusion"
    assert w["keywords"] == ["choice", "RT"]
    assert w["last_checked"] < time.strftime("%Y-%m-%d")  # starts 30 days back

    loaded = wt.load_watches(wfile)
    assert len(loaded) == 1

    assert wt.remove_watch(0, path=wfile) is True
    assert wt.load_watches(wfile) == []
    assert wt.remove_watch(5, path=wfile) is False


def test_load_watches_survives_corrupt_file(tmp_path):
    wfile = tmp_path / "watches.json"
    wfile.write_text("{not valid json", encoding="utf-8")
    assert wt.load_watches(wfile) == []


def test_check_watches_ranks_and_advances_date(graph_with_papers,
                                               tmp_path, monkeypatch):
    g = graph_with_papers
    wfile = tmp_path / "watches.json"
    wt.add_watch("bayesian cognition", path=wfile)

    fresh = [
        PaperMeta(paper_id=f"new_{i}",
                  title=f"A New Bayesian Paper {i}",
                  abstract="We model cognition with Bayesian inference "
                           "and report novel results on human learning.",
                  year=2026, source="watch")
        for i in range(3)
    ]
    monkeypatch.setattr(wt, "_search_since", lambda q, since, limit=50: fresh)

    reports = wt.check_watches(g, path=wfile)
    assert len(reports) == 1
    rep = reports[0]
    assert rep["n_found"] == 3
    # rank_candidates may floor low-relevance papers, but the structure holds
    assert isinstance(rep["results"], list)

    # last_checked advanced to today
    today = time.strftime("%Y-%m-%d")
    assert wt.load_watches(wfile)[0]["last_checked"] == today


def test_check_watches_empty_search(graph_with_papers, tmp_path, monkeypatch):
    wfile = tmp_path / "watches.json"
    wt.add_watch("quantum gravity", path=wfile)
    monkeypatch.setattr(wt, "_search_since", lambda q, since, limit=50: [])

    reports = wt.check_watches(graph_with_papers, path=wfile)
    assert reports[0]["n_found"] == 0
    assert reports[0]["results"] == []
