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

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
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

    def remember(self, metas) -> None:
        for m in metas:
            self.candidates[m.paper_id] = m


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


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(graph: Optional[HierarchicalResearchGraph] = None,
               autosave: bool = True) -> FastAPI:
    app = FastAPI(title="ResearchBuddy", docs_url=None, redoc_url=None)

    if graph is None:
        graph = load_graph() or HierarchicalResearchGraph()
    state = UIState(graph)
    app.state.rb = state

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
            candidates, hyde = find_candidates(
                state.graph, extra_keywords=keywords, query=intent)
            audit.log_event("search", query=intent or "", keywords=keywords,
                            n_results=len(candidates))
            results = state.graph.rank_candidates(
                candidates, n=n, exploration_ratio=cfg.EXPLORATION_RATIO,
                hyde_embedding=hyde, focus_ids=focus_ids or None)
            state.remember([m for m, _, _ in results])
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
            used = sb.load_used_seeds()
            if body.get("reset_frontier"):
                sb.reset_used_seeds()
                used = set()
            seeds = sb.pick_seeds(state.graph, exclude=used)
            if not seeds:
                return {"results": [], "stats": {"error": "no fresh seeds"}}
            cands, stats_d = sb.snowball_round(
                state.graph, seeds=seeds, directions=directions)
            audit.log_event("snowball", directions=list(directions),
                            n_seeds=stats_d["n_seeds"],
                            fetched=stats_d["fetched"],
                            new_unique=stats_d["new_unique"],
                            saturation=stats_d["saturation_ratio"])
            results = state.graph.rank_candidates(cands, n=n,
                                                  exploration_ratio=0.0)
            state.remember([m for m, _, _ in results])
        return {"results": _results_json(results), "stats": stats_d}

    # ── Harvest ──────────────────────────────────────────────────────────
    @app.post("/api/harvest")
    def harvest(body: dict):
        from researchbuddy.core import oa_harvester as oh
        n = max(1, min(int(body.get("n", 10)), 100))
        log: list[str] = []
        with state.lock:
            report = oh.harvest(state.graph, max_papers=n,
                                progress=log.append)
            if report.ingested:
                state.graph.rebuild_hierarchy()
            _save()
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
