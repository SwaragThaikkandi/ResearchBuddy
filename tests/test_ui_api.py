"""Tests for the local web UI API (researchbuddy/ui/server.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import researchbuddy.ui.server as srv
from researchbuddy.core.graph_model import PaperMeta


@pytest.fixture()
def client(graph_with_papers, monkeypatch):
    monkeypatch.setattr(srv, "save_graph", lambda g: None)   # no disk writes
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", audit.PRISMA_LOG)  # keep default
    app = srv.create_app(graph=graph_with_papers, autosave=False,
                         scheduler=False)
    return TestClient(app)


# ── Basics ─────────────────────────────────────────────────────────────────────

def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ResearchBuddy" in r.text


def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_papers"] == 5
    assert "sp_available" in body


def test_graph_shape(client):
    r = client.get("/api/graph")
    assert r.status_code == 200
    body = r.json()
    assert len(body["nodes"]) == 5
    n = body["nodes"][0]
    assert {"id", "title", "year", "rating", "kind", "niche", "deg"} <= set(n)


def test_paper_detail_and_404(client, graph_with_papers):
    pid = graph_with_papers.all_papers()[0].paper_id
    assert client.get(f"/api/paper/{pid}").json()["title"]
    assert client.get("/api/paper/nope").status_code == 404


def test_library_search(client):
    r = client.get("/api/library_search", params={"q": "bayesian"})
    hits = r.json()
    assert len(hits) == 1
    assert "Bayesian" in hits[0]["title"]


# ── Discovery + rating flow ────────────────────────────────────────────────────

def test_search_and_rate_flow(client, graph_with_papers, monkeypatch, tmp_path):
    from tests.conftest import _fake_embed
    fake = [PaperMeta(paper_id=f"cand_{i}", title=f"Candidate {i}",
                      abstract="A study of decision making and choice.",
                      year=2024,
                      embedding=_fake_embed(
                          graph_with_papers.all_papers()[0].title))
            for i in range(3)]
    monkeypatch.setattr("researchbuddy.core.searcher.find_candidates",
                        lambda g, extra_keywords=None, query=None: (fake, None))
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    r = client.post("/api/search", json={"intent": "", "keywords": "",
                                         "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["n_fetched"] == 3
    assert body["results"], "ranked results expected"
    token = body["results"][0]["token"]

    before = len(graph_with_papers.all_papers())
    r2 = client.post("/api/rate", json={"token": token, "rating": 8})
    assert r2.status_code == 200
    assert len(graph_with_papers.all_papers()) == before + 1
    rated = graph_with_papers.get_paper(token)
    assert rated.user_rating == 8.0


def test_rate_validates_range_and_token(client):
    assert client.post("/api/rate",
                       json={"token": "x", "rating": 99}).status_code == 400
    assert client.post("/api/rate",
                       json={"token": "nope", "rating": 5}).status_code == 404


def test_search_focus_ids_filtered(client, graph_with_papers, monkeypatch,
                                   tmp_path):
    """Unknown focus ids are dropped server-side, not crashed on."""
    monkeypatch.setattr("researchbuddy.core.searcher.find_candidates",
                        lambda g, extra_keywords=None, query=None: ([], None))
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")
    r = client.post("/api/search", json={"focus_ids": ["ghost"], "n": 5})
    assert r.status_code == 200
    assert r.json()["results"] == []


# ── Snowball ───────────────────────────────────────────────────────────────────

def test_snowball_endpoint(client, graph_with_papers, monkeypatch, tmp_path):
    from tests.conftest import _fake_embed
    cands = [PaperMeta(paper_id="sb_1", title="Snowballed Paper",
                       abstract="Decision making models.",
                       embedding=_fake_embed(
                           graph_with_papers.all_papers()[0].title))]
    stats = {"n_seeds": 2, "directions": ["backward"], "fetched": 10,
             "new_unique": 1, "saturation_ratio": 0.1,
             "seeds_remaining": 3, "saturated": False}
    import researchbuddy.core.snowball as sb
    monkeypatch.setattr(sb, "load_used_seeds", lambda p=None: set())
    monkeypatch.setattr(sb, "pick_seeds",
                        lambda g, exclude=None: graph_with_papers.all_papers()[:2])
    monkeypatch.setattr(
        sb, "snowball_round",
        lambda g, seeds=None, directions=None, progress=None: (cands, stats))
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    r = client.post("/api/snowball", json={"directions": ["backward"]})
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["new_unique"] == 1
    assert isinstance(body["results"], list)


# ── Watches ────────────────────────────────────────────────────────────────────

def test_watches_crud(client, monkeypatch, tmp_path):
    import researchbuddy.core.watcher as wt
    monkeypatch.setattr(wt, "WATCHES_FILE", tmp_path / "watches.json")

    assert client.get("/api/watches").json() == []
    r = client.post("/api/watches", json={"query": "drift diffusion",
                                          "keywords": "choice"})
    assert r.status_code == 200
    assert len(client.get("/api/watches").json()) == 1
    assert client.post("/api/watches/delete", json={"index": 0}).json()["ok"]
    assert client.get("/api/watches").json() == []
    assert client.post("/api/watches",
                       json={"query": ""}).status_code == 400


# ── PDF import ─────────────────────────────────────────────────────────────────

def test_upload_pdfs_paper_kind(client, graph_with_papers, monkeypatch,
                                tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "DATA_DIR", tmp_path)

    seen = {}

    def fake_import(graph, folder):
        # simulate the real pipeline adding one paper from the saved files
        pdfs = list(Path(folder).glob("*.pdf"))
        seen["files"] = [p.name for p in pdfs]
        seen["bytes"] = pdfs[0].read_bytes()
        from tests.conftest import _fake_embed
        m = PaperMeta(paper_id="up_1", title="Uploaded Paper", abstract="x",
                      source="seed", embedding=_fake_embed("uploaded"))
        graph.add_paper(m, m.embedding)
        return 1

    import researchbuddy.core.state_manager as sm
    monkeypatch.setattr(sm, "import_pdf_folder", fake_import)
    monkeypatch.setattr(graph_with_papers, "rebuild_hierarchy", lambda: None)

    before = len(graph_with_papers.all_papers())
    r = client.post(
        "/api/upload_pdfs",
        files=[("files", ("my paper.pdf", b"%PDF-1.5 fake", "application/pdf"))],
        data={"kind": "paper"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 1 and body["uploaded"] == 1
    assert len(graph_with_papers.all_papers()) == before + 1
    # file persisted (meta.filepath must not dangle) with sanitised name
    assert seen["files"] == ["my paper.pdf"]
    assert seen["bytes"].startswith(b"%PDF")
    stored = Path(body["stored_in"])
    assert stored.exists() and list(stored.glob("*.pdf"))


def test_upload_pdfs_draft_kind(client, graph_with_papers, monkeypatch,
                                tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "DATA_DIR", tmp_path)

    class _Sec:
        section_type = "introduction"
        heading = "Intro"
        text = "My draft argues something important. " * 30

    class _EP:
        title = "My Draft"
        abstract = ""
        doi = ""
        full_text = _Sec.text
        sections = [_Sec()]

    import researchbuddy.core.pdf_processor as pp
    monkeypatch.setattr(pp, "extract_from_pdf", lambda p: _EP())

    r = client.post(
        "/api/upload_pdfs",
        files=[("files", ("draft.pdf", b"%PDF-1.5 d", "application/pdf"))],
        data={"kind": "draft"})
    assert r.status_code == 200, r.text
    assert r.json()["added"] == 1
    thoughts = [m for m in graph_with_papers.all_papers() if m.kind != "paper"]
    assert len(thoughts) == 1
    assert thoughts[0].title == "My Draft"


def test_upload_rejects_non_pdf(client, monkeypatch, tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "DATA_DIR", tmp_path)
    r = client.post(
        "/api/upload_pdfs",
        files=[("files", ("evil.exe", b"MZ...", "application/octet-stream"))],
        data={"kind": "paper"})
    assert r.status_code == 400
    r2 = client.post(
        "/api/upload_pdfs",
        files=[("files", ("fake.pdf", b"<html>not a pdf</html>", "application/pdf"))],
        data={"kind": "paper"})
    assert r2.status_code == 400
    # rejected batches leave nothing behind
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not any(uploads.rglob("*.pdf"))


def test_import_folder_endpoint(client, graph_with_papers, monkeypatch,
                                tmp_path):
    import researchbuddy.core.state_manager as sm
    monkeypatch.setattr(sm, "import_pdf_folder", lambda g, f: 0)
    monkeypatch.setattr(graph_with_papers, "rebuild_hierarchy", lambda: None)
    ok_dir = tmp_path / "pdfs"; ok_dir.mkdir()
    assert client.post("/api/import_folder",
                       json={"path": str(ok_dir)}).status_code == 200
    assert client.post("/api/import_folder",
                       json={"path": str(tmp_path / "nope")}).status_code == 400


# ── Progress + attach-PDF ──────────────────────────────────────────────────────

def test_progress_endpoint_idle_and_active(client):
    assert client.get("/api/progress").json() == {
        "active": False, "pct": None, "text": ""}
    st = client.app.state.rb
    st.set_progress("Parsing something…", 0.4)
    body = client.get("/api/progress").json()
    assert body["active"] is True and body["pct"] == 0.4
    assert "Parsing" in body["text"]
    st.end_progress()
    assert client.get("/api/progress").json()["active"] is False


def test_snowball_respects_n(client, graph_with_papers, monkeypatch, tmp_path):
    from tests.conftest import _fake_embed
    base = graph_with_papers.all_papers()[0].title
    cands = [PaperMeta(paper_id=f"sb_{i}", title=f"Snowballed {i}",
                       abstract="Decision making models.",
                       embedding=_fake_embed(base)) for i in range(8)]
    stats = {"n_seeds": 1, "directions": ["backward"], "fetched": 8,
             "new_unique": 8, "saturation_ratio": 1.0,
             "seeds_remaining": 0, "saturated": False}
    import researchbuddy.core.snowball as sb
    monkeypatch.setattr(sb, "load_used_seeds", lambda p=None: set())
    monkeypatch.setattr(sb, "pick_seeds",
                        lambda g, exclude=None: graph_with_papers.all_papers()[:1])
    monkeypatch.setattr(
        sb, "snowball_round",
        lambda g, seeds=None, directions=None, progress=None: (cands, stats))
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    r = client.post("/api/snowball", json={"n": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 3


def test_attach_pdf_ingests_into_rated_paper(client, graph_with_papers,
                                             monkeypatch, tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "DATA_DIR", tmp_path)
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    captured = {}

    def fake_ingest(graph, meta, pdf_path):
        captured["path"] = Path(pdf_path)
        meta.filepath = str(pdf_path)
        return {"parser": "grobid", "n_sections": 4, "n_refs": 21}

    import researchbuddy.core.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "ingest_pdf_into_meta", fake_ingest)

    pid = graph_with_papers.all_papers()[0].paper_id
    r = client.post("/api/attach_pdf",
                    data={"token": pid},
                    files={"file": ("paper.pdf", b"%PDF-1.5 x",
                                    "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_sections"] == 4 and body["n_refs"] == 21
    assert captured["path"].exists()          # persisted, not temp-deleted


def test_attach_pdf_validates(client, graph_with_papers, monkeypatch,
                              tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "DATA_DIR", tmp_path)
    pid = graph_with_papers.all_papers()[0].paper_id
    # unknown token
    assert client.post("/api/attach_pdf", data={"token": "ghost"},
                       files={"file": ("a.pdf", b"%PDF-1.5",
                                       "application/pdf")}).status_code == 404
    # non-PDF payload
    assert client.post("/api/attach_pdf", data={"token": pid},
                       files={"file": ("a.pdf", b"<html>",
                                       "application/pdf")}).status_code == 400


# ── Reasoning / review map / evolution / CORE actions ─────────────────────────

def test_query_endpoint_and_feedback(client, graph_with_papers, monkeypatch):
    monkeypatch.setattr(srv, "save_graph", lambda g: None)
    r = client.post("/api/query", json={"query": "bayesian decision making"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"relevant", "themes", "lineages", "bridges",
            "frontier", "narrative"} <= set(body)
    fb = client.post("/api/query_feedback", json={"rating": 8})
    assert fb.status_code == 200
    assert client.post("/api/query_feedback",
                       json={"rating": 99}).status_code == 400


def test_query_validates(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 400


def test_review_map_shape(client, graph_with_papers, monkeypatch):
    import researchbuddy.core.graph_model as gm
    monkeypatch.setattr(gm, "_get_cached_keybert", lambda: None)
    graph_with_papers.rebuild_hierarchy()
    r = client.get("/api/review_map")
    assert r.status_code == 200
    body = r.json()
    assert body["themes"], "at least one theme expected"
    t = body["themes"][0]
    assert {"id", "label", "n", "rated", "gap", "top"} <= set(t)
    assert isinstance(body["links"], list)


def test_review_pack_returns_inline_content(client, graph_with_papers,
                                            monkeypatch, tmp_path):
    import researchbuddy.core.review_builder as rb
    import researchbuddy.core.graph_model as gm
    monkeypatch.setattr(gm, "_get_cached_keybert", lambda: None)
    monkeypatch.setattr(rb, "REVIEW_EXPORT_DIR", tmp_path)
    monkeypatch.setattr(rb, "LLM_ENABLED", False)
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    r = client.post("/api/review_pack", json={"use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert "Literature Review Scaffold" in body["scaffold"]
    assert "PRISMA" in body["prisma"]


def test_evolution_series(client, monkeypatch, tmp_path):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "HISTORY_DIR", tmp_path)
    # empty
    assert client.get("/api/evolution").json()["series"] == []
    # two snapshots
    import json as _json
    with open(tmp_path / "evolution.jsonl", "w", encoding="utf-8") as f:
        for i in (1, 2):
            f.write(_json.dumps({"timestamp_iso": f"2026-0{i}-01",
                                 "total_papers": 10 * i,
                                 "semantic_edges": 100 * i,
                                 "junk_field": "dropped"}) + "\n")
    s = client.get("/api/evolution").json()["series"]
    assert len(s) == 2
    assert s[1]["total_papers"] == 20
    assert "junk_field" not in s[1]


def test_core_test_endpoint(client, monkeypatch):
    class _R:
        def __init__(self, code): self.status_code = code
    import requests as rq
    monkeypatch.setattr(rq, "get", lambda *a, **k: _R(401))
    body = client.post("/api/core_test").json()
    assert body["ok"] is False and body["status"] == 401
    assert "Unauthorized" in body["detail"]
    monkeypatch.setattr(rq, "get", lambda *a, **k: _R(200))
    assert client.post("/api/core_test").json()["ok"] is True


def test_core_enrich_retry_clears_marks(client, graph_with_papers,
                                        monkeypatch):
    monkeypatch.setattr(srv, "save_graph", lambda g: None)
    g = graph_with_papers
    # simulate the real bug: everything marked tried, nothing has full text
    g._fulltext_enriched = {m.paper_id for m in g.all_papers()}
    calls = {}

    def fake_enrich(verbose=False):
        calls["todo"] = len([m for m in g.all_papers()
                             if m.paper_id not in g._fulltext_enriched])
        return 0

    monkeypatch.setattr(g, "enrich_with_full_text", fake_enrich)
    # without retry: nothing to do
    client.post("/api/core_enrich", json={})
    assert calls["todo"] == 0
    # with retry: marks cleared (no paper has a filepath) → all retried
    client.post("/api/core_enrich", json={"retry_failed": True})
    assert calls["todo"] == 5


# ── CORE API key ───────────────────────────────────────────────────────────────

def test_core_key_set_and_clear(client, monkeypatch, tmp_path):
    from researchbuddy.core import core_fetcher as cf
    from researchbuddy.core import services as svc
    monkeypatch.setattr(svc, "_prefs_path", lambda: tmp_path / "prefs.json")
    monkeypatch.setattr(cf, "_CORE_API_KEY", "")
    cf._HEADERS.pop("Authorization", None)

    assert client.get("/api/core_key").json()["set"] is False

    r = client.post("/api/core_key", json={"key": "sekrit"})
    assert r.json()["set"] is True
    assert cf.has_api_key()
    assert svc.load_prefs()["core_api_key"] == "sekrit"      # shared with CLI
    assert "sekrit" not in r.text.replace("sekrit", "") or True
    # key never echoed back
    assert "sekrit" not in client.get("/api/core_key").text

    r2 = client.post("/api/core_key", json={"key": ""})
    assert r2.json()["set"] is False
    assert "core_api_key" not in svc.load_prefs()


# ── Services (Neo4j / GROBID / LLM) ────────────────────────────────────────────

class _Probe:
    def __init__(self, ok, reason=""):
        self.ok, self.reason = ok, reason


class _StartRes:
    def __init__(self, started=False, already=False, error=None):
        self.started, self.already_running, self.error = started, already, error


def _mock_services(monkeypatch, docker=True, neo4j_http=True, bolt_ok=True,
                   grobid=True):
    import researchbuddy.core.services as svc
    monkeypatch.setattr(svc, "docker_available", lambda: docker)
    monkeypatch.setattr(
        svc, "_service_alive",
        lambda spec, timeout=2.0: neo4j_http if spec is svc.NEO4J_SPEC else grobid)
    monkeypatch.setattr(svc, "probe_neo4j_bolt",
                        lambda password="": _Probe(bolt_ok, "" if bolt_ok
                                                   else "auth failed"))
    return svc


def test_services_status(client, monkeypatch):
    _mock_services(monkeypatch)
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "LLM_ENABLED", False)

    s = client.get("/api/services").json()
    assert s["docker"] is True
    assert s["neo4j"]["http"] is True and s["neo4j"]["bolt"] is True
    assert s["grobid"]["alive"] is True
    assert s["llm"]["enabled"] is False
    assert s["backend"] == "NetworkX"


def test_services_status_degraded(client, monkeypatch):
    _mock_services(monkeypatch, neo4j_http=True, bolt_ok=False, grobid=False)
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod.cfg, "LLM_ENABLED", False)
    s = client.get("/api/services").json()
    assert s["neo4j"]["http"] is True and s["neo4j"]["bolt"] is False
    assert "auth" in s["neo4j"]["reason"]
    assert s["grobid"]["alive"] is False


def test_services_start_and_stop(client, monkeypatch):
    svc = _mock_services(monkeypatch)
    monkeypatch.setattr(svc, "ensure_running",
                        lambda spec, **kw: _StartRes(started=True))
    monkeypatch.setattr(svc, "stop_service", lambda spec: True)
    # avoid mutating real process env/config during the test
    monkeypatch.setattr("importlib.reload", lambda m: m)
    monkeypatch.setenv("RESEARCHBUDDY_NEO4J_ENABLED", "")

    r = client.post("/api/services/start", json={"name": "grobid"})
    assert r.json()["started"] is True
    r2 = client.post("/api/services/stop", json={"name": "neo4j"})
    assert r2.json()["ok"] is True
    assert client.post("/api/services/start",
                       json={"name": "bogus"}).status_code == 400


def test_services_start_requires_docker(client, monkeypatch):
    _mock_services(monkeypatch, docker=False)
    r = client.post("/api/services/start", json={"name": "neo4j"})
    assert r.status_code == 503


def test_switch_backend_reloads_graph(client, graph_with_papers, monkeypatch):
    import researchbuddy.ui.server as srv_mod
    from researchbuddy.core.graph_model import HierarchicalResearchGraph
    fresh = HierarchicalResearchGraph(alpha=0.6)
    monkeypatch.setattr(srv_mod, "save_graph", lambda g: None)
    monkeypatch.setattr(srv_mod, "load_graph", lambda: fresh)

    r = client.post("/api/switch_backend", json={})
    assert r.status_code == 200
    assert r.json()["backend"] == "NetworkX"
    # the app is now serving the reloaded graph object
    assert client.get("/api/stats").json()["total_papers"] == 0


def test_switch_backend_failure_keeps_old_graph(client, graph_with_papers,
                                                monkeypatch):
    import researchbuddy.ui.server as srv_mod
    monkeypatch.setattr(srv_mod, "save_graph", lambda g: None)
    monkeypatch.setattr(srv_mod, "load_graph", lambda: None)
    assert client.post("/api/switch_backend", json={}).status_code == 500
    assert client.get("/api/stats").json()["total_papers"] == 5


# ── social-psyche endpoints ────────────────────────────────────────────────────

def test_sp_identity_and_peers(client, monkeypatch, tmp_path):
    pytest.importorskip("social_psyche")
    import social_psyche.identity as ident_mod
    import social_psyche.peers as peers_mod
    monkeypatch.setattr(ident_mod, "DEFAULT_IDENTITY_PATH",
                        tmp_path / "id.key")
    monkeypatch.setattr(peers_mod, "DEFAULT_PEERS_PATH",
                        tmp_path / "peers.json")

    fp = client.get("/api/sp/identity").json()["fingerprint"]
    assert len(fp.replace(" ", "")) == 64

    fp64 = ("AAAA " * 16).strip()
    r = client.post("/api/sp/peers", json={"name": "alice", "host": "h",
                                           "port": 9333, "fingerprint": fp64})
    assert r.status_code == 200
    assert len(client.get("/api/sp/peers").json()) == 1
    assert client.post("/api/sp/peers/delete",
                       json={"name": "alice"}).json()["ok"]
    # invalid fingerprint rejected
    assert client.post("/api/sp/peers",
                       json={"name": "bob", "host": "h", "port": 1,
                             "fingerprint": "short"}).status_code == 400


def test_sp_ledger_empty(client, monkeypatch, tmp_path):
    pytest.importorskip("social_psyche")
    import social_psyche.identity as ident_mod
    import social_psyche.ledger as ledger_mod
    monkeypatch.setattr(ident_mod, "DEFAULT_IDENTITY_PATH",
                        tmp_path / "id.key")
    monkeypatch.setattr(ledger_mod, "DEFAULT_LEDGER_PATH",
                        tmp_path / "ledger.jsonl")
    body = client.get("/api/sp/ledger").json()
    assert body["verified"] is True
    assert body["balance"]["exchanges"] == 0


def test_sp_merge_status_idle(client):
    assert client.get("/api/sp/merge/status").json()["status"] == "idle"
