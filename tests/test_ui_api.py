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
    app = srv.create_app(graph=graph_with_papers, autosave=False)
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
    monkeypatch.setattr(sb, "snowball_round",
                        lambda g, seeds=None, directions=None: (cands, stats))
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
