"""Tests for Sentinel — continuous literature surveillance."""

from __future__ import annotations

import time

import pytest

from researchbuddy.core import sentinel as sn
from researchbuddy.core.graph_model import PaperMeta


@pytest.fixture(autouse=True)
def _tmp_audit(monkeypatch, tmp_path):
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "prisma.jsonl")


# ── Config + scheduling ────────────────────────────────────────────────────────

def test_config_roundtrip(tmp_path):
    p = tmp_path / "sentinel.json"
    cfg = sn.load_config(p)
    assert cfg["enabled"] is False                 # safe default: off
    cfg["enabled"] = True
    cfg["interval_hours"] = 6
    sn.save_config(cfg, p)
    again = sn.load_config(p)
    assert again["enabled"] is True and again["interval_hours"] == 6


def test_is_due_logic():
    now = time.time()
    assert sn.is_due({"enabled": False, "interval_hours": 1,
                      "last_run": 0}, now) is False       # disabled → never
    assert sn.is_due({"enabled": True, "interval_hours": 1,
                      "last_run": 0}, now) is True         # never ran → due
    assert sn.is_due({"enabled": True, "interval_hours": 1,
                      "last_run": now - 30}, now) is False # just ran
    assert sn.is_due({"enabled": True, "interval_hours": 1,
                      "last_run": now - 3700}, now) is True


# ── Inbox ─────────────────────────────────────────────────────────────────────

def test_inbox_roundtrip_and_remove(tmp_path):
    p = tmp_path / "inbox.jsonl"
    assert sn.inbox_list(p) == []
    sn._inbox_write([{"token": "a", "title": "A"},
                     {"token": "b", "title": "B"}], p)
    assert len(sn.inbox_list(p)) == 2
    popped = sn.inbox_remove("a", p)
    assert popped["title"] == "A"
    assert [e["token"] for e in sn.inbox_list(p)] == ["b"]
    assert sn.inbox_remove("ghost", p) is None


def test_entry_to_meta():
    m = sn.entry_to_meta({"token": "t1", "title": "T", "abstract": "A",
                          "year": "2024", "doi": "10.1/x"})
    assert isinstance(m, PaperMeta)
    assert m.year == 2024                          # coercion applies here too
    assert m.source == "sentinel"


# ── The scan ──────────────────────────────────────────────────────────────────

def _fake_reports(graph, metas_scores):
    return [{"watch": {"query": "topic q"},
             "n_found": len(metas_scores),
             "results": [(m, s, "relevant") for m, s in metas_scores]}]


def test_run_scan_triage_dedupe_digest(graph_with_papers, tmp_path,
                                       monkeypatch):
    from tests.conftest import _fake_embed
    g = graph_with_papers
    hi = PaperMeta(paper_id="hi1", title="Highly Relevant",
                   abstract="x", year=2026, doi="10.1/hi",
                   embedding=_fake_embed("hi"))
    lo = PaperMeta(paper_id="lo1", title="Barely Relevant",
                   abstract="x", year=2026, embedding=_fake_embed("lo"))
    dup = g.all_papers()[0]                        # already in graph

    import researchbuddy.core.watcher as wt
    monkeypatch.setattr(wt, "check_watches",
                        lambda graph, progress=None: _fake_reports(
                            graph, [(hi, 0.8), (lo, 0.1), (dup, 0.9)]))

    inbox = tmp_path / "inbox.jsonl"
    cfgp = tmp_path / "sentinel.json"
    report = sn.run_scan(g, config={"min_score": 0.35},
                         inbox_path=inbox, config_path=cfgp,
                         digest_dir=tmp_path / "digests")

    entries = sn.inbox_list(inbox)
    assert [e["token"] for e in entries] == ["hi1"]   # lo filtered, dup skipped
    assert report["new"] == 1
    assert report["per_watch"][0]["kept"] == 1
    # digest written and mentions the keeper
    from pathlib import Path
    digest = Path(report["digest"])
    assert digest.exists()
    assert "Highly Relevant" in digest.read_text(encoding="utf-8")
    # last_run stamped → not due immediately
    assert sn.load_config(cfgp)["last_run"] > 0

    # second scan with same finds → nothing new (inbox dedupe)
    report2 = sn.run_scan(g, config={"min_score": 0.35},
                          inbox_path=inbox, config_path=cfgp,
                          digest_dir=tmp_path / "digests")
    assert report2["new"] == 0


# ── API surface ────────────────────────────────────────────────────────────────

def test_sentinel_endpoints(graph_with_papers, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import researchbuddy.ui.server as srv
    monkeypatch.setattr(srv, "save_graph", lambda g: None)
    monkeypatch.setattr(sn, "SENTINEL_FILE", tmp_path / "sentinel.json")
    monkeypatch.setattr(sn, "INBOX_FILE", tmp_path / "inbox.jsonl")
    client = TestClient(srv.create_app(graph=graph_with_papers,
                                       autosave=False, scheduler=False))

    st = client.get("/api/sentinel").json()
    assert st["config"]["enabled"] is False and st["inbox_count"] == 0

    client.post("/api/sentinel", json={"enabled": True, "interval_hours": 6,
                                       "min_score": 0.5})
    st2 = client.get("/api/sentinel").json()
    assert st2["config"]["enabled"] is True
    assert st2["config"]["interval_hours"] == 6

    # accept flow: seed the inbox, then accept into the graph with a rating
    from tests.conftest import _fake_embed
    sn._inbox_write([{"token": "nb1", "title": "Inbox Paper",
                      "abstract": "abs", "year": 2026, "doi": "10.1/n",
                      "authors": [], "url": "", "venue": ""}],
                    tmp_path / "inbox.jsonl")
    before = len(graph_with_papers.all_papers())
    r = client.post("/api/sentinel/inbox/accept",
                    json={"token": "nb1", "rating": 8})
    assert r.status_code == 200
    assert len(graph_with_papers.all_papers()) == before + 1
    assert graph_with_papers.get_paper("nb1").user_rating == 8.0
    assert client.get("/api/sentinel/inbox").json() == []

    assert client.post("/api/sentinel/inbox/accept",
                       json={"token": "ghost"}).status_code == 404
