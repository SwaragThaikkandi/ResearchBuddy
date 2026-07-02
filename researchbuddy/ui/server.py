"""
ResearchBuddy web UI server.

A local-first, single-user web app: `researchbuddy-ui` starts a FastAPI
server bound to 127.0.0.1 and opens the browser. Everything runs on your
machine — no cloud, no account, no telemetry (the mission demands it).

The frontend is dependency-free vanilla JS served from ./static — no npm,
no CDN, works fully offline. social-psyche features (identity, ledger,
peers, live merge) light up automatically when that package is installed.

Endpoints are plain `def` (not async), so FastAPI runs them in a thread
pool — long operations (search, harvest) don't block the event loop. A
process-wide lock serialises graph mutations.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import researchbuddy.config as cfg
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.state_manager import save as save_graph, load as load_graph
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class UIState:
    """Mutable server state: the live graph + candidate cache + merge job."""

    def __init__(self, graph: HierarchicalResearchGraph):
        self.graph = graph
        self.lock = threading.Lock()
        self.candidates: dict[str, PaperMeta] = {}   # token -> pending candidate
        self.merge_job: dict = {"status": "idle"}
        # Live progress for the browser's progress bar. pct=None means
        # indeterminate (spinner-style bar); text explains the current step.
        self.progress: dict = {"active": False, "pct": None, "text": ""}

    def remember(self, metas) -> None:
        for m in metas:
            self.candidates[m.paper_id] = m

    def set_progress(self, text: str, pct: Optional[float] = None) -> None:
        self.progress = {"active": True,
                         "pct": None if pct is None else round(float(pct), 3),
                         "text": str(text)[:200]}

    def end_progress(self) -> None:
        self.progress = {"active": False, "pct": None, "text": ""}


# ── Serialization helpers ─────────────────────────────────────────────────────

def _paper_json(m: PaperMeta, score: Optional[float] = None,
                label: str = "") -> dict:
    return {
        "token": m.paper_id,
        "title": m.title,
        "authors": m.authors[:4],
        "year": m.year,
        "abstract": (m.abstract or "")[:400],
        "doi": m.doi,
        "url": m.url,
        "venue": m.venue,
        "rating": m.user_rating,
        "kind": m.kind,
        "source": m.source,
        "peer_reviewed": m.is_peer_reviewed,
        "cited_by": getattr(m, "cited_by_count", None),
        "has_fulltext": bool(m.filepath),
        "score": round(score, 3) if score is not None else None,
        "label": label,
    }


def _results_json(results) -> list[dict]:
    return [_paper_json(m, score=s, label=lab) for m, s, lab in results]


def _ingest_draft_pdfs(graph: HierarchicalResearchGraph,
                       paths: list[Path]) -> int:
    """Ingest the user's own PDFs as thought/draft nodes (strong implicit
    weight on the context vector) — mirrors the CLI's option 14."""
    from researchbuddy.core.pdf_processor import extract_from_pdf

    added = 0
    for p in paths:
        try:
            ep = extract_from_pdf(p)
        except Exception as e:
            logger.warning("draft extraction failed for %s: %s", p.name, e)
            continue
        if ep is None:
            continue
        joined = "\n\n".join(s.text for s in ep.sections if s.text) \
                 or ep.full_text or ep.abstract
        section_text_map: dict[str, list[str]] = {}
        for s in ep.sections:
            if s.section_type and s.section_type != "other" and s.text:
                section_text_map.setdefault(s.section_type, []).append(s.text)
        meta = graph.add_thought_from_text(
            joined, title=ep.title or p.stem, kind="draft",
            section_text_map={k: "\n\n".join(v)
                              for k, v in section_text_map.items()},
        )
        if meta is not None:
            if ep.doi and not meta.doi:
                meta.doi = ep.doi
            added += 1
    return added


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(graph: Optional[HierarchicalResearchGraph] = None,
               autosave: bool = True) -> FastAPI:
    app = FastAPI(title="ResearchBuddy", docs_url=None, redoc_url=None)

    if graph is None:
        graph = load_graph() or HierarchicalResearchGraph()
    state = UIState(graph)
    app.state.rb = state

    # Reapply a saved CORE API key (core_fetcher reads env only at import;
    # the key lives in service prefs, shared with the CLI).
    try:
        from researchbuddy.core import core_fetcher
        from researchbuddy.core import services as svc
        _saved_key = (svc.load_prefs().get("core_api_key") or "").strip()
        if _saved_key and not core_fetcher.has_api_key():
            core_fetcher.set_api_key(_saved_key)
    except Exception as e:                            # pragma: no cover
        logger.debug("CORE key pref load skipped: %s", e)

    def _save():
        if autosave:
            try:
                save_graph(state.graph)
            except Exception as e:      # pragma: no cover - defensive
                logger.warning("autosave failed: %s", e)

    # ── Static frontend ──────────────────────────────────────────────────
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
                  name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    # ── Graph + stats ────────────────────────────────────────────────────
    @app.get("/api/stats")
    def stats():
        s = state.graph.stats()
        s["sp_available"] = _sp_available()
        s["backend"] = getattr(state.graph._backend, "backend_name",
                               "NetworkX")
        return s

    @app.get("/api/graph")
    def graph_data(min_weight: float = 0.45, max_nodes: int = 400,
                   max_links: int = 2000):
        g = state.graph
        papers = [m for m in g.all_papers()][:max_nodes]
        ids = {m.paper_id for m in papers}
        p2n = g.paper_to_niche()
        deg: dict[str, int] = {}
        links = []
        for layer, G in (("semantic", g.G_semantic),
                         ("citation", g.G_citation)):
            try:
                for u, v, d in G.edges(data=True):
                    w = float(d.get("weight", 1.0) or 1.0)
                    if u in ids and v in ids and (layer == "citation"
                                                  or w >= min_weight):
                        links.append({"s": u, "t": v, "w": round(w, 3),
                                      "layer": layer})
                        deg[u] = deg.get(u, 0) + 1
                        deg[v] = deg.get(v, 0) + 1
            except Exception as e:
                logger.debug("graph layer %s skipped: %s", layer, e)
        # Dense graphs would swamp the canvas: keep the strongest edges.
        if len(links) > max_links:
            links.sort(key=lambda l: -l["w"])
            links = links[:max_links]
            deg.clear()
            for l in links:
                deg[l["s"]] = deg.get(l["s"], 0) + 1
                deg[l["t"]] = deg.get(l["t"], 0) + 1
        nodes = [{
            "id": m.paper_id, "title": m.title[:90], "year": m.year,
            "rating": m.user_rating, "kind": m.kind,
            "niche": p2n.get(m.paper_id, ""),
            "deg": deg.get(m.paper_id, 0),
            "fulltext": bool(m.filepath),
        } for m in papers]
        return {"nodes": nodes, "links": links}

    @app.get("/api/paper/{pid}")
    def paper_detail(pid: str):
        m = state.graph.get_paper(pid) or state.candidates.get(pid)
        if m is None:
            raise HTTPException(404, "unknown paper")
        d = _paper_json(m)
        d["abstract"] = (m.abstract or "")[:2000]
        d["authors"] = m.authors
        return d

    # ── Discovery ────────────────────────────────────────────────────────
    @app.get("/api/library_search")
    def library_search(q: str = ""):
        ql = q.strip().lower()
        hits = [m for m in state.graph.all_papers()
                if m.kind == "paper" and ql and ql in m.title.lower()][:12]
        return [_paper_json(m) for m in hits]

    @app.get("/api/progress")
    def progress():
        return state.progress

    @app.post("/api/search")
    def search(body: dict):
        from researchbuddy.core.searcher import find_candidates
        intent = (body.get("intent") or "").strip() or None
        keywords = [k.strip() for k in (body.get("keywords") or "").split(",")
                    if k.strip()]
        focus_ids = [pid for pid in (body.get("focus_ids") or [])
                     if state.graph.get_paper(pid)]
        n = max(1, min(int(body.get("n", 10)), 30))

        with state.lock:
            try:
                state.set_progress(
                    "Querying OpenAlex, CrossRef, Semantic Scholar and arXiv "
                    "(plus LLM query expansion when available)…", 0.15)
                candidates, hyde = find_candidates(
                    state.graph, extra_keywords=keywords, query=intent)
                audit.log_event("search", query=intent or "",
                                keywords=keywords, n_results=len(candidates))
                state.set_progress(
                    f"Ranking {len(candidates)} candidates — fused semantic + "
                    "citation + PageRank scoring with your learned weights…",
                    0.75)
                results = state.graph.rank_candidates(
                    candidates, n=n, exploration_ratio=cfg.EXPLORATION_RATIO,
                    hyde_embedding=hyde, focus_ids=focus_ids or None)
                state.remember([m for m, _, _ in results])
            finally:
                state.end_progress()
        return {"results": _results_json(results),
                "n_fetched": len(candidates)}

    @app.post("/api/rate")
    def rate(body: dict):
        token = body.get("token", "")
        rating = float(body.get("rating", 0))
        if not (1 <= rating <= 10):
            raise HTTPException(400, "rating must be 1-10")
        with state.lock:
            g = state.graph
            meta = g.get_paper(token) or state.candidates.get(token)
            if meta is None:
                raise HTTPException(404, "unknown paper/candidate")
            if meta.paper_id not in {p.paper_id for p in g.all_papers()}:
                if meta.embedding is None:
                    g.embed_abstract(meta)
                g.add_paper(meta, meta.embedding)
            meta.times_shown += 1
            meta.last_shown = time.time()
            g.rate_paper(meta.paper_id, rating)
            audit.log_event("screen", paper_id=meta.paper_id,
                            title=meta.title[:200], doi=meta.doi,
                            rating=rating,
                            decision=audit.screen_decision(rating))
            _save()
        return {"ok": True, "paper_id": meta.paper_id}

    @app.post("/api/rebuild")
    def rebuild():
        with state.lock:
            state.graph.rebuild_hierarchy()
            learned = state.graph.learn_signal_weights()
            _save()
        return {"ok": True, "learned_weights": bool(learned)}

    # ── Snowball ─────────────────────────────────────────────────────────
    @app.post("/api/snowball")
    def snowball(body: dict):
        from researchbuddy.core import snowball as sb
        directions = tuple(body.get("directions")
                           or ("backward", "forward"))
        n = max(1, min(int(body.get("n", 10)), 30))
        with state.lock:
            try:
                used = sb.load_used_seeds()
                if body.get("reset_frontier"):
                    sb.reset_used_seeds()
                    used = set()
                seeds = sb.pick_seeds(state.graph, exclude=used)
                if not seeds:
                    return {"results": [],
                            "stats": {"error": "no fresh seeds"}}
                n_seeds = len(seeds)
                done = {"i": 0}

                def _prog(msg: str) -> None:
                    # snowball_round reports one line per seed — turn that
                    # into a real fraction for the progress bar.
                    if msg.startswith("["):
                        done["i"] += 1
                    state.set_progress(
                        msg, 0.05 + 0.75 * (done["i"] / max(n_seeds, 1)))

                state.set_progress(
                    f"Walking the citation frontier from {n_seeds} seed "
                    "paper(s)…", 0.05)
                cands, stats_d = sb.snowball_round(
                    state.graph, seeds=seeds, directions=directions,
                    progress=_prog)
                audit.log_event("snowball", directions=list(directions),
                                n_seeds=stats_d["n_seeds"],
                                fetched=stats_d["fetched"],
                                new_unique=stats_d["new_unique"],
                                saturation=stats_d["saturation_ratio"])
                state.set_progress(
                    f"Ranking {len(cands)} new candidates with your learned "
                    "weights…", 0.9)
                results = state.graph.rank_candidates(cands, n=n,
                                                      exploration_ratio=0.0)
                state.remember([m for m, _, _ in results])
            finally:
                state.end_progress()
        return {"results": _results_json(results), "stats": stats_d}

    # ── Attach a PDF to a rated/known paper (CLI parity: richer node) ────
    @app.post("/api/attach_pdf")
    def attach_pdf(token: str = Form(...), file: UploadFile = File(...)):
        """
        Optional PDF ingest for a paper you just rated (or any graph paper):
        GROBID parses it into section embeddings + local references, turning
        an abstract-only node into a full one — exactly the CLI's
        "PDF path to ingest as full graph node" prompt, without the typing.
        """
        import re as _re
        import shutil
        from researchbuddy.core.ingest import ingest_pdf_into_meta, IngestError

        with state.lock:
            meta = state.graph.get_paper(token) or state.candidates.get(token)
            if meta is None:
                raise HTTPException(404, "unknown paper — rate it first")
            in_graph = state.graph.get_paper(token) is not None
        name = Path(file.filename or "paper.pdf").name
        name = _re.sub(r"[^A-Za-z0-9 ._-]+", "_", name) or "paper.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, f"not a PDF: {name}")
        dest_dir = cfg.DATA_DIR / "uploads" / "attached"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{token[:12]}_{name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        if not dest.read_bytes().startswith(b"%PDF"):
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "invalid PDF payload")

        with state.lock:
            try:
                state.set_progress(
                    f"Parsing {name} with GROBID (section embeddings + "
                    "reference extraction — first request loads models, "
                    "can take a minute)…")
                if not in_graph:
                    # rate-later flow: add as unrated node so ingest sticks
                    if meta.embedding is None:
                        state.graph.embed_abstract(meta)
                    state.graph.add_paper(meta, meta.embedding)
                info = ingest_pdf_into_meta(state.graph, meta, dest)
                audit.log_event("fulltext", paper_id=meta.paper_id,
                                title=meta.title[:200], doi=meta.doi,
                                provider="user-supplied-ui", license="",
                                url="")
                _save()
            except IngestError as e:
                raise HTTPException(422, str(e))
            finally:
                state.end_progress()
        return {"ok": True, "parser": info["parser"],
                "n_sections": info["n_sections"], "n_refs": info["n_refs"]}

    # ── Harvest ──────────────────────────────────────────────────────────
    @app.post("/api/harvest")
    def harvest(body: dict):
        from researchbuddy.core import oa_harvester as oh
        n = max(1, min(int(body.get("n", 10)), 100))
        log: list[str] = []

        def _prog(msg: str) -> None:
            log.append(msg)
            # harvester emits "[i/n] Title…" lines — derive a fraction
            pct = None
            if msg.startswith("[") and "/" in msg.split("]")[0]:
                try:
                    i, tot = msg[1:].split("]")[0].split("/")
                    pct = 0.05 + 0.85 * (int(i) - 1) / max(int(tot), 1)
                except ValueError:
                    pass
            state.set_progress(msg, pct)

        with state.lock:
            try:
                state.set_progress(
                    "Resolving legal open-access copies (arXiv → Unpaywall → "
                    "OpenAlex → Europe PMC)…", 0.02)
                report = oh.harvest(state.graph, max_papers=n,
                                    progress=_prog)
                if report.ingested:
                    state.set_progress(
                        "Rebuilding the hierarchy with the new full-text "
                        "content…", 0.95)
                    state.graph.rebuild_hierarchy()
                _save()
            finally:
                state.end_progress()
        return {"checked": report.checked, "resolved": report.resolved,
                "downloaded": report.downloaded, "ingested": report.ingested,
                "no_oa": report.no_oa, "by_provider": report.by_provider,
                "errors": report.errors[:10], "log": log[-50:]}

    # ── Review pack ──────────────────────────────────────────────────────
    @app.post("/api/review_pack")
    def review_pack(body: dict):
        from researchbuddy.core.review_builder import export_review_pack
        with state.lock:
            pack = export_review_pack(state.graph,
                                      use_llm=bool(body.get("use_llm")))
        return {"path": str(pack),
                "files": sorted(p.name for p in Path(pack).iterdir())}

    # ── Watches (living review) ──────────────────────────────────────────
    @app.get("/api/watches")
    def watches_list():
        from researchbuddy.core import watcher as wt
        return wt.load_watches()

    @app.post("/api/watches")
    def watches_add(body: dict):
        from researchbuddy.core import watcher as wt
        q = (body.get("query") or "").strip()
        if not q:
            raise HTTPException(400, "query required")
        kws = [k.strip() for k in (body.get("keywords") or "").split(",")
               if k.strip()]
        return wt.add_watch(q, kws)

    @app.post("/api/watches/delete")
    def watches_delete(body: dict):
        from researchbuddy.core import watcher as wt
        ok = wt.remove_watch(int(body.get("index", -1)))
        return {"ok": ok}

    @app.post("/api/watches/check")
    def watches_check():
        from researchbuddy.core import watcher as wt
        with state.lock:
            reports = wt.check_watches(state.graph)
            out = []
            for rep in reports:
                state.remember([m for m, _, _ in rep["results"]])
                out.append({"watch": rep["watch"],
                            "n_found": rep["n_found"],
                            "results": _results_json(rep["results"])})
        return out

    @app.post("/api/save")
    def save_now():
        with state.lock:
            save_graph(state.graph)
        return {"ok": True}

    # ── PDF import (browser upload + server-side folder) ─────────────────
    @app.post("/api/upload_pdfs")
    def upload_pdfs(files: list[UploadFile] = File(...),
                    kind: str = Form("paper")):
        """
        Drag-and-drop / file-picker import. `kind`:
          paper — seed papers: full GROBID extraction, section embeddings,
                  parsed references (same pipeline as the CLI folder import)
          draft — the user's OWN writing: becomes a strongly-weighted
                  thought node that re-shapes future recommendations
        """
        import re as _re
        import shutil

        if kind not in ("paper", "draft"):
            raise HTTPException(400, "kind must be 'paper' or 'draft'")
        # Persist uploads: imported papers keep meta.filepath pointing here,
        # so the library dir must outlive the request (NOT a temp dir).
        batch_dir = (cfg.DATA_DIR / "uploads"
                     / time.strftime("%Y%m%d_%H%M%S"))
        batch_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        try:
            for uf in files:
                name = Path(uf.filename or "upload.pdf").name
                name = _re.sub(r"[^A-Za-z0-9 ._-]+", "_", name) or "upload.pdf"
                if not name.lower().endswith(".pdf"):
                    raise HTTPException(400, f"not a PDF: {name}")
                dest = batch_dir / name
                with open(dest, "wb") as out:
                    shutil.copyfileobj(uf.file, out)
                if dest.stat().st_size == 0 or \
                        not dest.read_bytes().startswith(b"%PDF"):
                    raise HTTPException(400, f"invalid PDF payload: {name}")
                saved.append(dest)
        except HTTPException:
            shutil.rmtree(batch_dir, ignore_errors=True)   # reject whole batch
            raise

        with state.lock:
            try:
                state.set_progress(
                    f"Extracting {len(saved)} PDF(s) with GROBID — sections, "
                    "references, figures (falls back to pdfplumber; first "
                    "GROBID request loads models, ~30 s)…")
                if kind == "paper":
                    from researchbuddy.core.state_manager import import_pdf_folder
                    before = len(state.graph.all_papers())
                    import_pdf_folder(state.graph, batch_dir)
                    added = len(state.graph.all_papers()) - before
                    if added:
                        state.set_progress(
                            "Rebuilding hierarchy with the new papers…", 0.9)
                        state.graph.rebuild_hierarchy()
                else:
                    added = _ingest_draft_pdfs(state.graph, saved)
                _save()
            finally:
                state.end_progress()
        return {"ok": True, "kind": kind, "uploaded": len(saved),
                "added": added, "stored_in": str(batch_dir),
                "note": ("" if added == len(saved) else
                         f"{len(saved) - added} skipped "
                         "(duplicates or no extractable text)")}

    @app.post("/api/import_folder")
    def import_folder(body: dict):
        """Bulk import of a folder that already lives on this machine."""
        from researchbuddy.core.state_manager import import_pdf_folder
        folder = Path((body.get("path") or "").strip()).expanduser()
        if not folder.is_dir():
            raise HTTPException(400, f"not a folder: {folder}")
        with state.lock:
            before = len(state.graph.all_papers())
            import_pdf_folder(state.graph, folder)
            added = len(state.graph.all_papers()) - before
            if added:
                state.graph.rebuild_hierarchy()
            _save()
        return {"ok": True, "added": added}

    # ── Services (Neo4j / GROBID / LLM) — status + one-click control ─────
    @app.get("/api/services")
    def services_status():
        from researchbuddy.core import services as svc
        out: dict = {
            "docker": svc.docker_available(),
            "backend": getattr(state.graph._backend, "backend_name",
                               "NetworkX"),
        }
        # Neo4j: HTTP liveness + bolt auth probe (the one that matters)
        n_http = svc._service_alive(svc.NEO4J_SPEC)
        bolt_ok, bolt_reason = False, ""
        if n_http:
            pw = os.environ.get("RESEARCHBUDDY_NEO4J_PASSWORD",
                                "researchbuddy")
            probe = svc.probe_neo4j_bolt(password=pw)
            bolt_ok, bolt_reason = probe.ok, (probe.reason or "")
        out["neo4j"] = {"http": n_http, "bolt": bolt_ok,
                        "reason": bolt_reason,
                        "browser": "http://localhost:7474"}
        # GROBID
        out["grobid"] = {"alive": svc._service_alive(svc.GROBID_SPEC)}
        # LLM (Ollama)
        llm_info = {"enabled": cfg.LLM_ENABLED, "available": False,
                    "model": cfg.LLM_MODEL}
        if cfg.LLM_ENABLED:
            try:
                from researchbuddy.core.llm import get_llm
                st = get_llm().status()
                llm_info["available"] = bool(st.available)
                if not st.available and st.error:
                    llm_info["error"] = str(st.error)[:120]
            except Exception as e:
                llm_info["error"] = str(e)[:120]
        out["llm"] = llm_info
        return out

    @app.post("/api/services/start")
    def services_start(body: dict):
        from researchbuddy.core import services as svc
        name = body.get("name", "")
        spec = {"neo4j": svc.NEO4J_SPEC, "grobid": svc.GROBID_SPEC}.get(name)
        if spec is None:
            raise HTTPException(400, "name must be 'neo4j' or 'grobid'")
        if not svc.docker_available():
            raise HTTPException(503, "Docker not detected — install/start "
                                     "Docker Desktop first")
        res = svc.ensure_running(spec)
        if name == "neo4j" and (res.started or res.already_running):
            # Same env plumbing the CLI startup uses, so create_backend()
            # (which reads config at call time) can see Neo4j.
            os.environ.setdefault("RESEARCHBUDDY_NEO4J_ENABLED", "true")
            os.environ.setdefault("RESEARCHBUDDY_NEO4J_PASSWORD",
                                  "researchbuddy")
            importlib.reload(cfg)
        return {"started": bool(res.started),
                "already_running": bool(res.already_running),
                "error": res.error or ""}

    @app.post("/api/services/stop")
    def services_stop(body: dict):
        from researchbuddy.core import services as svc
        name = body.get("name", "")
        spec = {"neo4j": svc.NEO4J_SPEC, "grobid": svc.GROBID_SPEC}.get(name)
        if spec is None:
            raise HTTPException(400, "name must be 'neo4j' or 'grobid'")
        return {"ok": bool(svc.stop_service(spec))}

    @app.get("/api/core_key")
    def core_key_status():
        from researchbuddy.core import core_fetcher
        return {"set": core_fetcher.has_api_key()}

    @app.post("/api/core_key")
    def core_key_set(body: dict):
        """Apply a CORE API key live + persist it in service prefs (same
        storage the CLI uses, so both surfaces stay in sync). Empty key
        clears it. The key itself is never echoed back."""
        from researchbuddy.core import core_fetcher
        from researchbuddy.core import services as svc
        key = (body.get("key") or "").strip()
        core_fetcher.set_api_key(key)
        try:
            prefs = svc.load_prefs()
            if key:
                prefs["core_api_key"] = key
            else:
                prefs.pop("core_api_key", None)
            svc.save_prefs(prefs)
        except Exception as e:                       # pragma: no cover
            logger.warning("could not persist CORE key: %s", e)
        return {"set": core_fetcher.has_api_key()}

    @app.post("/api/switch_backend")
    def switch_backend():
        """Reload the graph so it migrates to whatever backend the current
        config selects (Neo4j after /api/services/start, else NetworkX).
        No UI restart needed — this is the CLI's 're-launch to switch',
        done live."""
        with state.lock:
            save_graph(state.graph)          # never lose in-memory work
            new_graph = load_graph()
            if new_graph is None:
                raise HTTPException(500, "could not reload the graph")
            state.graph = new_graph
        return {"backend": getattr(state.graph._backend, "backend_name",
                                   "NetworkX")}

    # ── social-psyche integration ────────────────────────────────────────
    def _sp_available() -> bool:
        try:
            import social_psyche  # noqa: F401
            return True
        except ImportError:
            return False

    def _sp_or_501():
        if not _sp_available():
            raise HTTPException(
                501, "social-psyche not installed — "
                     "pip install -e <path>/social-psyche[net]")

    @app.get("/api/sp/identity")
    def sp_identity():
        _sp_or_501()
        from social_psyche.identity import Identity
        ident = Identity.load_or_create()
        return {"fingerprint": ident.fingerprint()}

    @app.get("/api/sp/ledger")
    def sp_ledger():
        _sp_or_501()
        from social_psyche.identity import Identity
        from social_psyche.ledger import Ledger
        led = Ledger(Identity.load_or_create())
        problems = led.verify()
        entries = [{"height": e.height,
                    "body": e.receipt["body"]} for e in led.entries()[-20:]]
        return {"balance": led.balance(), "entries": entries,
                "verified": not problems, "problems": problems[:10]}

    @app.get("/api/sp/peers")
    def sp_peers():
        _sp_or_501()
        from social_psyche import peers as pr
        return pr.load_peers()

    @app.post("/api/sp/peers")
    def sp_peers_add(body: dict):
        _sp_or_501()
        from social_psyche import peers as pr
        try:
            return pr.add_peer(body.get("name", ""), body.get("host", ""),
                               int(body.get("port", 0)),
                               body.get("fingerprint", ""))
        except pr.PeerError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/sp/peers/delete")
    def sp_peers_delete(body: dict):
        _sp_or_501()
        from social_psyche import peers as pr
        return {"ok": pr.remove_peer(body.get("name", ""))}

    @app.post("/api/sp/export_capsule")
    def sp_export_capsule(body: dict):
        _sp_or_501()
        from researchbuddy.core import capsule as cap
        from social_psyche.identity import Identity
        from social_psyche.publish import sign_file
        cfg.CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
        out = cfg.CAPSULE_DIR / f"mygraph_{int(time.time())}.rbcapsule"
        with state.lock:
            capsule = cap.export_capsule(
                state.graph,
                share_identifiers=bool(body.get("share_ids")),
                share_ratings=bool(body.get("share_ratings")))
            path = cap.write_capsule(capsule, out)
        sig = sign_file(path, Identity.load_or_create())
        return {"capsule": str(path), "signature": str(sig),
                "stats": capsule.stats}

    @app.post("/api/sp/merge")
    def sp_merge(body: dict):
        """Live merge. connect = synchronous; serve = background job
        (poll /api/sp/merge/status)."""
        _sp_or_501()
        from social_psyche import netmerge
        from social_psyche.identity import Identity
        from social_psyche import peers as pr

        ident = Identity.load_or_create()
        mode = body.get("mode", "connect")
        host, port = body.get("host", ""), int(body.get("port", 9333))
        peer_fp = (body.get("peer_fp") or "").strip() or None
        if body.get("peer"):
            entry = pr.get_peer(body["peer"])
            host = host or entry["host"]
            port = entry["port"]
            peer_fp = peer_fp or entry["fingerprint"]
        kwargs = dict(identity=ident, expected_peer_fp=peer_fp,
                      share_identifiers=bool(body.get("share_ids", True)),
                      share_ratings=bool(body.get("share_ratings")))

        def _result_json(res):
            r = res.report
            return {"peer_fingerprint": res.peer_fingerprint,
                    "shared": len(res.shared_dois),
                    "imported": r.imported,
                    "shared_by_doi": r.shared_by_doi,
                    "novel_regions": r.novel_regions,
                    "jaccard_doi": r.jaccard_doi,
                    "spectral_distance": r.spectral_distance,
                    "deltacon": r.deltacon_similarity,
                    "gw_distortion": r.gw_distortion,
                    "notes": r.notes}

        if mode == "connect":
            if not host:
                raise HTTPException(400, "host or pinned peer required")
            with state.lock:
                res = netmerge.connect(state.graph, host, port, **kwargs)
                _save()
            return {"status": "done", "result": _result_json(res)}

        # serve mode — background thread, single job at a time
        if state.merge_job.get("status") == "waiting":
            raise HTTPException(409, "already waiting for a peer")

        def _serve():
            try:
                with state.lock:
                    res = netmerge.serve(state.graph, "0.0.0.0", port,
                                         **kwargs)
                    _save()
                state.merge_job = {"status": "done",
                                   "result": _result_json(res)}
            except Exception as e:
                state.merge_job = {"status": "error", "error": str(e)}

        state.merge_job = {"status": "waiting", "port": port}
        threading.Thread(target=_serve, daemon=True).start()
        return {"status": "waiting", "port": port,
                "fingerprint": ident.fingerprint()}

    @app.get("/api/sp/merge/status")
    def sp_merge_status():
        return state.merge_job

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    import webbrowser
    import uvicorn

    p = argparse.ArgumentParser(description="ResearchBuddy web UI (local)")
    p.add_argument("--port", type=int, default=8230)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    app = create_app()
    url = f"http://127.0.0.1:{args.port}"
    print(f"ResearchBuddy UI: {url}  (Ctrl-C to quit; graph saves on rate)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # 127.0.0.1 ONLY — single-user local app, never exposed to the network.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
