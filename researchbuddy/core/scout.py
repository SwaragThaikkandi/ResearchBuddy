"""
Living Graph — a Bayesian scout that explores literature for you.

Architecture (the user's framing, made precise):

    PRIOR       your main graph. Its context vector μ₀ encodes what you
                have read, rated, and written — accumulated belief.
    LIKELIHOOD  a SECOND graph the scout grows and prunes autonomously.
                Each cycle it acquires abstract-only candidates seeded by
                the prior, then optimises itself with graph-theoretic
                queries: PageRank mass flowing from prior-aligned nodes,
                pruning of low-mass regions, diversity-selected slates.
    EVIDENCE    your ratings on the scout's slate.
    POSTERIOR   rated scout papers flow into the main graph; the context
                vector update is precision-weighted (existing rating
                machinery), TEMPERED by ABSTRACT_EVIDENCE_DISCOUNT because
                abstract-only evidence is weaker than full text (β < 1,
                standard tempered Bayes). Attach/harvest the PDF later and
                the discount vanishes automatically.

The scout can only strengthen the posterior through YOU — it never writes
to the main graph on its own. That is the Goodhart firewall: the optimiser
(scout) and the ground truth (your judgment) are different agents.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from researchbuddy.config import (
    DATA_DIR, SCOUT_GRAPH_FILE, SCOUT_FILE,
    SCOUT_MAX_SIZE, SCOUT_ACQUIRE, SCOUT_SLATE_SIZE,
)
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.embedder import cosine_similarity
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

DEFAULT_STATE = {
    "enabled": False,
    "interval_hours": 24,
    "last_run": 0.0,
    "cycles": 0,
    "slate": [],          # current slate entries awaiting the user's verdict
    "anchors": [],        # scout paper_ids the user rated >=7 (restart boost)
    "avoid": [],          # rated <=3 — pruned and never re-acquired
}


# ── State ─────────────────────────────────────────────────────────────────────

def load_state(path: Optional[Path] = None) -> dict:
    p = Path(path or SCOUT_FILE)
    state = dict(DEFAULT_STATE)
    if p.exists():
        try:
            state.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("scout state unreadable: %s", e)
    return state


def save_state(state: dict, path: Optional[Path] = None) -> None:
    p = Path(path or SCOUT_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULT_STATE)
    merged.update(state)
    p.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def is_due(state: dict, now: Optional[float] = None) -> bool:
    if not state.get("enabled"):
        return False
    now = time.time() if now is None else now
    interval = max(1, float(state.get("interval_hours", 24))) * 3600
    return (now - float(state.get("last_run", 0.0))) >= interval


def load_scout_graph(path: Optional[Path] = None) -> HierarchicalResearchGraph:
    p = Path(path or SCOUT_GRAPH_FILE)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("scout graph unreadable (%s) — starting fresh", e)
    return HierarchicalResearchGraph()


def save_scout_graph(graph: HierarchicalResearchGraph,
                     path: Optional[Path] = None) -> None:
    p = Path(path or SCOUT_GRAPH_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(graph, f)


# ── Graph-theoretic self-optimisation ─────────────────────────────────────────

def _prior_similarity(prior: Optional[np.ndarray],
                      meta: PaperMeta) -> float:
    if prior is None or meta.embedding is None:
        return 0.0
    return max(0.0, float(cosine_similarity(prior, meta.embedding)))


def _scout_ppr(scout: HierarchicalResearchGraph,
               prior: Optional[np.ndarray],
               anchors: set[str]) -> dict[str, float]:
    """PageRank mass over the scout's semantic graph, restarting from
    prior-aligned nodes (and hard-boosting user-confirmed anchors)."""
    import networkx as nx
    G = nx.Graph()
    papers = scout.all_papers()
    G.add_nodes_from(m.paper_id for m in papers)
    try:
        for u, v, d in scout.G_semantic.edges(data=True):
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, weight=float(d.get("weight", 1.0) or 1.0))
    except Exception as e:
        logger.debug("scout ppr edges skipped: %s", e)
    if G.number_of_edges() == 0:
        return {}
    restart = {}
    for m in papers:
        w = _prior_similarity(prior, m)
        if m.paper_id in anchors:
            w = max(w, 0.5) * 3.0          # confirmed evidence dominates
        if w > 0:
            restart[m.paper_id] = w
    total = sum(restart.values())
    if not total:
        return {}
    restart = {k: v / total for k, v in restart.items()}
    try:
        pr = nx.pagerank(G, alpha=0.85, personalization=restart,
                         weight="weight")
    except Exception as e:
        logger.debug("scout pagerank failed: %s", e)
        return {}
    peak = max(pr.values()) or 1.0
    return {k: v / peak for k, v in pr.items()}


def _prune(scout: HierarchicalResearchGraph, keep_scores: dict[str, float],
           max_size: int) -> HierarchicalResearchGraph:
    """Keep the top max_size papers by keep-score; the scout stays small,
    sharp, and cheap. (No remove-node API → rebuild a fresh graph.)"""
    papers = scout.all_papers()
    if len(papers) <= max_size:
        return scout
    papers.sort(key=lambda m: (-keep_scores.get(m.paper_id, 0.0), m.paper_id))
    fresh = HierarchicalResearchGraph()
    for m in papers[:max_size]:
        fresh.add_paper(m, m.embedding)
    return fresh


# ── The cycle ─────────────────────────────────────────────────────────────────

def run_cycle(
    main_graph: HierarchicalResearchGraph,
    scout: Optional[HierarchicalResearchGraph] = None,
    progress: Optional[Callable[..., None]] = None,
    state_path: Optional[Path] = None,
    graph_path: Optional[Path] = None,
    acquire: int = SCOUT_ACQUIRE,
    max_size: int = SCOUT_MAX_SIZE,
    slate_size: int = SCOUT_SLATE_SIZE,
) -> dict:
    """
    One living-graph cycle: acquire → self-optimise → prune → slate.
    Returns a report; persists the scout graph + state.
    """
    from researchbuddy.core.searcher import search_openalex

    say = progress or (lambda *a, **k: None)
    state = load_state(state_path)
    scout = scout if scout is not None else load_scout_graph(graph_path)
    prior = main_graph.context_vector()
    anchors = set(state.get("anchors", []))
    avoid = set(state.get("avoid", []))

    # 1. Queries derived from the prior (what you care about) ------------
    say("Deriving exploration queries from your prior…", 0.05)
    queries = main_graph.top_seed_keywords(n=6)[:3]
    top_rated = sorted(
        [m for m in main_graph.rated_papers()
         if (m.user_rating or 0) >= 7 and m.kind == "paper"],
        key=lambda m: (-(m.user_rating or 0), m.paper_id))[:2]
    queries += [m.title for m in top_rated]
    queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
    if not queries:
        return {"ok": False,
                "note": "Prior is empty — add/rate papers first so the "
                        "scout knows what to explore."}

    # 2. Acquire abstract-only candidates --------------------------------
    known_main = {m.doi.lower() for m in main_graph.all_papers() if m.doi}
    known_scout = {m.doi.lower() for m in scout.all_papers() if m.doi}
    per_q = max(5, acquire // max(len(queries), 1))
    added = 0
    for i, q in enumerate(queries):
        say(f"Acquiring literature ({i + 1}/{len(queries)}): '{q[:50]}'…",
            0.1 + 0.35 * i / len(queries))
        try:
            found = search_openalex(q, limit=per_q)
        except Exception as e:
            logger.debug("scout acquisition failed for %r: %s", q, e)
            continue
        for m in found:
            key = (m.doi or "").lower()
            if (m.paper_id in avoid or (key and (key in known_main
                                                 or key in known_scout))):
                continue
            m.source = "scout"
            if m.embedding is None:
                scout.embed_abstract(m)
            if scout.add_paper(m, m.embedding):
                known_scout.add(key)
                added += 1

    # 3. Self-optimise: structure, PageRank mass, prune ------------------
    say("Self-optimising: rebuilding structure + PageRank from the prior…",
        0.55)
    try:
        scout.rebuild_hierarchy()
    except Exception as e:
        logger.debug("scout rebuild skipped: %s", e)
    ppr = _scout_ppr(scout, prior, anchors)
    keep = {m.paper_id: 0.6 * _prior_similarity(prior, m)
            + 0.4 * ppr.get(m.paper_id, 0.0)
            for m in scout.all_papers()}
    before = len(scout.all_papers())
    scout = _prune(scout, keep, max_size)
    pruned = before - len(scout.all_papers())
    if pruned:
        try:
            scout.rebuild_hierarchy()
        except Exception as e:
            logger.debug("scout rebuild after prune skipped: %s", e)

    # 4. Slate: best finds, diversity-selected (MMR) ---------------------
    say("Selecting the slate (relevance + diversity)…", 0.85)
    in_main_titles = {m.title.lower().strip() for m in main_graph.all_papers()}
    cands = [m for m in scout.all_papers()
             if m.user_rating is None
             and m.title.lower().strip() not in in_main_titles]
    scored = []
    for m in cands:
        imp = HierarchicalResearchGraph._impact_signal(m)
        s = (0.5 * _prior_similarity(prior, m)
             + 0.3 * ppr.get(m.paper_id, 0.0)
             + 0.2 * (imp if imp is not None else 0.0))
        scored.append((m, s))
    scored.sort(key=lambda t: (-t[1], t[0].paper_id))
    slate: list[tuple[PaperMeta, float]] = []
    for m, s in scored:
        if len(slate) >= slate_size:
            break
        redundancy = max(
            (float(cosine_similarity(m.embedding, p.embedding))
             for p, _ in slate
             if m.embedding is not None and p.embedding is not None),
            default=0.0)
        if 0.7 * s - 0.3 * redundancy > 0:
            slate.append((m, s))

    state["slate"] = [{
        "token": m.paper_id, "title": m.title,
        "abstract": (m.abstract or "")[:400],
        "authors": m.authors[:4], "year": m.year, "doi": m.doi,
        "url": m.url, "venue": m.venue, "score": round(s, 3),
    } for m, s in slate]
    state["cycles"] = int(state.get("cycles", 0)) + 1
    state["last_run"] = time.time()
    save_state(state, state_path)
    save_scout_graph(scout, graph_path)
    audit.log_event("scout_cycle", acquired=added, pruned=pruned,
                    size=len(scout.all_papers()), slate=len(slate))
    # ASCII only: progress lines reach cp1252 Windows consoles via the CLI.
    say(f"Cycle done: +{added} acquired, -{pruned} pruned, "
        f"{len(slate)} on the slate.", 1.0)
    return {"ok": True, "acquired": added, "pruned": pruned,
            "size": len(scout.all_papers()), "slate": state["slate"],
            "cycles": state["cycles"]}


# ── Evidence → posterior ──────────────────────────────────────────────────────

def apply_feedback(
    main_graph: HierarchicalResearchGraph,
    token: str,
    rating: float,
    state_path: Optional[Path] = None,
    graph_path: Optional[Path] = None,
) -> dict:
    """
    The Bayesian update. A rated slate paper becomes evidence:
      - imported into the MAIN graph (source='scout' → tempered
        effective_weight via ABSTRACT_EVIDENCE_DISCOUNT) and rated there,
        which shifts the context vector = posterior update;
      - recorded as an anchor (>=7) or avoided (<=3) so the next scout
        cycle restarts its PageRank from confirmed evidence.
    """
    state = load_state(state_path)
    scout = load_scout_graph(graph_path)
    meta = scout.get_paper(token)
    if meta is None:
        entry = next((e for e in state.get("slate", [])
                      if e.get("token") == token), None)
        if entry is None:
            return {"ok": False, "note": "unknown scout paper"}
        meta = PaperMeta(paper_id=entry["token"], title=entry["title"],
                         abstract=entry.get("abstract", ""),
                         authors=list(entry.get("authors") or []),
                         year=entry.get("year"), doi=entry.get("doi", ""),
                         url=entry.get("url", ""),
                         venue=entry.get("venue", ""), source="scout")

    # evidence into the posterior (tempered by evidence_factor)
    if main_graph.get_paper(meta.paper_id) is None:
        clone = PaperMeta(
            paper_id=meta.paper_id, title=meta.title, abstract=meta.abstract,
            authors=list(meta.authors), year=meta.year, doi=meta.doi,
            url=meta.url, venue=meta.venue,
            cited_by_count=getattr(meta, "cited_by_count", None),
            source="scout")
        if meta.embedding is not None:
            clone.embedding = meta.embedding
        else:
            main_graph.embed_abstract(clone)
        main_graph.add_paper(clone, clone.embedding)
    main_graph.rate_paper(meta.paper_id, float(rating))
    audit.log_event("screen", paper_id=meta.paper_id, title=meta.title[:200],
                    doi=meta.doi, rating=float(rating),
                    decision=audit.screen_decision(float(rating)),
                    channel="scout")

    # evidence into the next likelihood cycle
    if rating >= 7:
        anchors = set(state.get("anchors", []))
        anchors.add(meta.paper_id)
        state["anchors"] = sorted(anchors)[-100:]
        if scout.get_paper(meta.paper_id) is not None:
            scout.rate_paper(meta.paper_id, float(rating))
            save_scout_graph(scout, graph_path)
    elif rating <= 3:
        av = set(state.get("avoid", []))
        av.add(meta.paper_id)
        state["avoid"] = sorted(av)[-500:]
    state["slate"] = [e for e in state.get("slate", [])
                      if e.get("token") != token]
    save_state(state, state_path)
    return {"ok": True, "paper_id": meta.paper_id,
            "evidence_factor": meta.evidence_factor}
