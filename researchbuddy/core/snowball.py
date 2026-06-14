"""
Citation snowballing — the systematic-review expansion loop.

Backward snowballing follows the reference lists of your best-rated papers;
forward snowballing finds newer papers that cite them. Both operate purely
on bibliographic metadata (titles, authors, DOIs, abstracts surfaced by the
OpenAlex and Semantic Scholar APIs) — facts, not copyrightable expression.

Two data paths feed the backward direction:
  1. local_refs   — references GROBID already parsed out of PDFs you own
                    (works fully offline), then enriched via OpenAlex
  2. OpenAlex     — referenced_works of each seed, resolved by DOI

Forward citations come from OpenAlex's `cites:` filter, with Semantic
Scholar as fallback when a seed has an S2 id but no DOI.

Saturation tracking reports the ratio of new-unique results per round —
the standard stopping criterion for a systematic search ("another round
found almost nothing new => coverage is saturated").
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from researchbuddy.config import (
    OPENALEX_URL, REQUEST_TIMEOUT, REQUEST_DELAY,
    SNOWBALL_MIN_RATING, SNOWBALL_MAX_SEEDS, SNOWBALL_PER_PAPER,
    SNOWBALL_MAX_CANDIDATES, SNOWBALL_SATURATION,
    SNOWBALL_STATE_FILE, SNOWBALL_FRONTIER_FILL,
)
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.searcher import _openalex_to_meta, fetch_forward_citations

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ResearchBuddy/0.6 (citation snowballing)"}


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def _oa_get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    import os
    params = dict(params or {})
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    try:
        r = requests.get(url, params=params, headers=_HEADERS,
                         timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug("OpenAlex request failed (%s): %s", url, e)
    return None


def _work_by_doi(doi: str) -> Optional[dict]:
    if not doi:
        return None
    return _oa_get(f"{OPENALEX_URL}/doi:{doi}")


def _short_id(openalex_url_or_id: str) -> str:
    """'https://openalex.org/W123' -> 'W123' (already-short ids pass through)."""
    return (openalex_url_or_id or "").rstrip("/").rsplit("/", 1)[-1]


def _works_by_ids(ids: list[str]) -> list[dict]:
    """Batch-fetch works by OpenAlex id (50 per request, the API maximum)."""
    out: list[dict] = []
    short = [s for s in (_short_id(i) for i in ids) if s]
    for i in range(0, len(short), 50):
        batch = short[i:i + 50]
        data = _oa_get(OPENALEX_URL, params={
            "filter":   "openalex:" + "|".join(batch),
            "per-page": 50,
        })
        if data:
            out.extend(data.get("results", []))
        time.sleep(REQUEST_DELAY)
    return out


def _works_by_dois(dois: list[str]) -> list[dict]:
    """Batch-fetch works by DOI (50 per request)."""
    out: list[dict] = []
    clean = [d.strip().lower() for d in dois if d and d.strip()]
    for i in range(0, len(clean), 50):
        batch = clean[i:i + 50]
        data = _oa_get(OPENALEX_URL, params={
            "filter":   "doi:" + "|".join(batch),
            "per-page": 50,
        })
        if data:
            out.extend(data.get("results", []))
        time.sleep(REQUEST_DELAY)
    return out


def _citing_works(openalex_id: str, limit: int) -> list[dict]:
    data = _oa_get(OPENALEX_URL, params={
        "filter":   f"cites:{_short_id(openalex_id)}",
        "per-page": min(int(limit), 200),
        "sort":     "cited_by_count:desc",
    })
    time.sleep(REQUEST_DELAY)
    return (data or {}).get("results", [])


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def _norm_title(t: str) -> str:
    return " ".join((t or "").lower().split())


def _known_keys(graph: HierarchicalResearchGraph) -> tuple[set[str], set[str]]:
    dois  = {m.doi.lower() for m in graph.all_papers() if m.doi}
    names = {_norm_title(m.title) for m in graph.all_papers()}
    return dois, names


# ── Used-seed memory (the frontier walk's bookkeeping) ────────────────────────
#
# Re-running a snowball from the same favourites re-harvests the same 1-hop
# shell — the #1 cause of premature saturation. We persist the ids we've
# already expanded so each round walks to a *new* frontier instead.

def load_used_seeds(path=None) -> set[str]:
    import json
    p = Path(path or SNOWBALL_STATE_FILE)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.get("used_seed_ids", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_used_seeds(used: set[str], path=None) -> None:
    import json
    p = Path(path or SNOWBALL_STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"used_seed_ids": sorted(used)}, indent=2),
                 encoding="utf-8")


def reset_used_seeds(path=None) -> None:
    """Forget the frontier — the next round restarts from the best papers."""
    p = Path(path or SNOWBALL_STATE_FILE)
    if p.exists():
        p.unlink()


# ── Frontier detection ────────────────────────────────────────────────────────

def _degree_map(graph: HierarchicalResearchGraph) -> dict[str, int]:
    """Combined semantic + citation degree per paper (cheap structural signal)."""
    deg: dict[str, int] = {}
    for G in (graph.G_semantic, graph.G_citation):
        try:
            for n, d in G.degree():
                deg[n] = deg.get(n, 0) + int(d)
        except Exception:
            continue
    return deg


def frontier_papers(graph: HierarchicalResearchGraph,
                    exclude: set[str],
                    limit: int) -> list[PaperMeta]:
    """
    Papers worth pushing outward from next: snowball/discovered papers with a
    DOI that we haven't expanded yet, ordered by how *recently* they entered
    the graph (insertion order) and how *underconnected* they are. Recent +
    sparse = the live edge of the search frontier.
    """
    deg = _degree_map(graph)
    # graph._papers preserves insertion order → later = more recently added.
    ordered = list(graph.all_papers())
    cands = [
        m for m in ordered
        if m.kind == "paper" and m.doi
        and m.paper_id not in exclude
        and m.source in ("snowball", "discovered")
    ]
    # Recency rank (newest first), then fewest connections first.
    recency = {m.paper_id: i for i, m in enumerate(reversed(cands))}
    cands.sort(key=lambda m: (recency[m.paper_id], deg.get(m.paper_id, 0)))
    return cands[:limit]


# ── Seed selection ────────────────────────────────────────────────────────────

def pick_seeds(graph: HierarchicalResearchGraph,
               min_rating: float = SNOWBALL_MIN_RATING,
               max_seeds: int = SNOWBALL_MAX_SEEDS,
               exclude: Optional[set[str]] = None,
               frontier_fill: bool = SNOWBALL_FRONTIER_FILL) -> list[PaperMeta]:
    """
    Seeds for the next round, in priority order:
      1. best-rated, not-yet-expanded papers (rating >= min_rating)
      2. frontier fill — recently-added, underconnected snowball papers, so
         successive rounds walk outward to new neighbourhoods (2-hop, 3-hop...)
      3. seed PDFs, only when nothing else qualifies (cold start)

    `exclude` is the set of paper_ids already expanded in prior rounds.
    """
    exclude = exclude or set()
    rated = [m for m in graph.rated_papers()
             if (m.user_rating or 0) >= min_rating and m.kind == "paper"
             and m.paper_id not in exclude]
    rated.sort(key=lambda m: (-(m.user_rating or 0), -(m.year or 0)))
    seeds = rated[:max_seeds]

    if frontier_fill and len(seeds) < max_seeds:
        picked = {m.paper_id for m in seeds} | exclude
        seeds += frontier_papers(graph, picked, max_seeds - len(seeds))

    if not seeds:
        seeds = [m for m in graph.seed_papers()
                 if m.kind == "paper" and m.paper_id not in exclude][:max_seeds]
    return seeds


# ── The snowball round ────────────────────────────────────────────────────────

def snowball_round(
    graph: HierarchicalResearchGraph,
    seeds: Optional[list[PaperMeta]] = None,
    directions: Iterable[str] = ("backward", "forward"),
    per_paper: int = SNOWBALL_PER_PAPER,
    max_candidates: int = SNOWBALL_MAX_CANDIDATES,
    progress: Optional[Callable[[str], None]] = None,
    remember: bool = True,
    state_path=None,
) -> tuple[list[PaperMeta], dict]:
    """
    One round of backward/forward snowballing from `seeds`.

    Returns (candidates, stats). Candidates are deduped against the graph
    and each other; rank them with graph.rank_candidates() like any other
    search result. stats carries the saturation diagnostics.

    When `remember` is True (default), the seeds expanded this round are
    persisted so the *next* round automatically walks outward to a fresh
    frontier instead of re-harvesting the same shell.
    """
    say = progress or (lambda s: None)
    used = load_used_seeds(state_path) if remember else set()
    if seeds is None:
        seeds = pick_seeds(graph, exclude=used)
    directions = set(directions)

    known_dois, known_titles = _known_keys(graph)
    fetched = 0
    candidates: dict[str, PaperMeta] = {}     # paper_id -> meta
    pending_ref_dois: list[str] = []          # local refs to enrich via OpenAlex
    bare_refs: list[PaperMeta] = []           # local refs without DOI

    def _consider(meta: Optional[PaperMeta]) -> None:
        nonlocal fetched
        if meta is None:
            return
        fetched += 1
        if meta.doi and meta.doi.lower() in known_dois:
            return
        nt = _norm_title(meta.title)
        if not nt or nt in known_titles:
            return
        if meta.paper_id in candidates:
            return
        known_titles.add(nt)          # dedupe within the round too
        if meta.doi:
            known_dois.add(meta.doi.lower())
        candidates[meta.paper_id] = meta

    for i, seed in enumerate(seeds, 1):
        say(f"[{i}/{len(seeds)}] snowballing from: {seed.title[:60]}")

        oa_work: Optional[dict] = None
        if seed.doi:
            oa_work = _work_by_doi(seed.doi)
            time.sleep(REQUEST_DELAY)

        # ── Backward: references ─────────────────────────────────────────
        if "backward" in directions:
            # (a) GROBID-parsed local references — free, offline
            for ref in (seed.local_refs or [])[:per_paper]:
                doi = (ref.get("doi") or "").lower()
                title = (ref.get("title") or "").strip()
                if doi and doi not in known_dois:
                    pending_ref_dois.append(doi)
                elif title and _norm_title(title) not in known_titles:
                    bare_refs.append(PaperMeta(
                        paper_id = HierarchicalResearchGraph.make_id(title, doi=doi),
                        title    = title[:250],
                        abstract = "",
                        authors  = list(ref.get("authors") or [])[:10],
                        year     = ref.get("year"),
                        doi      = doi,
                        source   = "snowball",
                    ))
            # (b) OpenAlex referenced_works
            ref_ids = (oa_work or {}).get("referenced_works") or []
            if ref_ids:
                for item in _works_by_ids(ref_ids[:per_paper]):
                    _consider(_openalex_to_meta(item))

        # ── Forward: citing papers ───────────────────────────────────────
        if "forward" in directions:
            oa_id = (oa_work or {}).get("id", "")
            if oa_id:
                for item in _citing_works(oa_id, per_paper):
                    _consider(_openalex_to_meta(item))
            elif seed.s2_id:
                for m in fetch_forward_citations(seed.s2_id, limit=per_paper):
                    _consider(m)

        if len(candidates) >= max_candidates:
            say("    candidate cap reached — stopping early")
            break

    # ── Enrich GROBID local refs via one batched OpenAlex DOI lookup ───────
    if pending_ref_dois and len(candidates) < max_candidates:
        say(f"Enriching {len(pending_ref_dois)} parsed references via OpenAlex ...")
        found_dois: set[str] = set()
        for item in _works_by_dois(pending_ref_dois):
            m = _openalex_to_meta(item)
            if m:
                found_dois.add(m.doi.lower())
                _consider(m)
        # DOIs OpenAlex couldn't resolve are still real, new references —
        # keep them as bare-metadata candidates rather than inflating the
        # `fetched` denominator (the old behaviour faked saturation).
        for doi in pending_ref_dois:
            if doi not in found_dois and doi not in known_dois:
                _consider(PaperMeta(
                    paper_id = HierarchicalResearchGraph.make_id("", doi=doi),
                    title    = f"[unresolved reference {doi}]",
                    abstract = "",
                    doi      = doi,
                    source   = "snowball",
                ))
    for m in bare_refs:
        _consider(m)

    cands = list(candidates.values())[:max_candidates]
    for m in cands:
        if m.source == "discovered":
            m.source = "snowball"

    # ── Remember expanded seeds so the next round walks a new frontier ─────
    if remember:
        used |= {s.paper_id for s in seeds}
        save_used_seeds(used, state_path)
    seeds_remaining = len(pick_seeds(graph, exclude=used)) if remember else None

    new_unique = len(cands)
    saturation_ratio = (new_unique / fetched) if fetched else 0.0
    # A round is only "saturated" if it found little new AND there's no fresh
    # frontier left to expand — otherwise the next round will keep finding work.
    truly_saturated = bool(
        fetched >= 20 and saturation_ratio < SNOWBALL_SATURATION
        and (seeds_remaining == 0 if seeds_remaining is not None else True)
    )
    stats = {
        "n_seeds":          len(seeds),
        "directions":       sorted(directions),
        "fetched":          fetched,
        "new_unique":       new_unique,
        "saturation_ratio": round(saturation_ratio, 3),
        "seeds_remaining":  seeds_remaining,
        "saturated":        truly_saturated,
    }
    return cands, stats
