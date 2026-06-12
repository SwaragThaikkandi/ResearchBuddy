"""
Living review — saved watch queries with "what's new since I last checked".

A literature review is obsolete the day it's written unless it keeps
watching. Each watch stores a query + keywords + the date it last ran;
checking a watch asks OpenAlex only for works published since then,
ranks them against the user's graph, and reports the top hits.

Metadata-only (titles, abstracts via the OpenAlex API) — no full texts
are fetched here; pipe the keepers through the OA harvester instead.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from researchbuddy.config import (
    WATCHES_FILE, OPENALEX_URL, REQUEST_TIMEOUT, REQUEST_DELAY,
)
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.searcher import _openalex_to_meta
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ResearchBuddy/0.6 (living review watcher)"}

WATCH_TOP_N = 5          # new papers reported per watch per check


# ── Persistence ───────────────────────────────────────────────────────────────

def load_watches(path: Optional[Path] = None) -> list[dict]:
    p = Path(path or WATCHES_FILE)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Could not read watches: %s", e)
        return []


def save_watches(watches: list[dict], path: Optional[Path] = None) -> None:
    p = Path(path or WATCHES_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(watches, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def add_watch(query: str, keywords: Optional[list[str]] = None,
              path: Optional[Path] = None) -> dict:
    watches = load_watches(path)
    watch = {
        "query":        query.strip(),
        "keywords":     [k.strip() for k in (keywords or []) if k.strip()],
        "created":      time.strftime("%Y-%m-%d"),
        # Start the window a month back so the first check isn't empty.
        "last_checked": time.strftime(
            "%Y-%m-%d", time.localtime(time.time() - 30 * 86400)),
    }
    watches.append(watch)
    save_watches(watches, path)
    return watch


def remove_watch(index: int, path: Optional[Path] = None) -> bool:
    watches = load_watches(path)
    if 0 <= index < len(watches):
        watches.pop(index)
        save_watches(watches, path)
        return True
    return False


# ── Checking ──────────────────────────────────────────────────────────────────

def _search_since(query: str, since: str, limit: int = 50) -> list[PaperMeta]:
    """OpenAlex search restricted to works published on/after `since`."""
    import os
    params = {
        "search":   query,
        "per-page": min(limit, 200),
        "filter":   f"from_publication_date:{since},has_abstract:true",
        "sort":     "publication_date:desc",
    }
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    try:
        r = requests.get(OPENALEX_URL, params=params, headers=_HEADERS,
                         timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        items = r.json().get("results", [])
    except Exception as e:
        logger.debug("Watch search failed: %s", e)
        return []
    out = []
    for item in items:
        m = _openalex_to_meta(item)
        if m:
            m.source = "watch"
            out.append(m)
    time.sleep(REQUEST_DELAY)
    return out


def check_watches(
    graph: HierarchicalResearchGraph,
    path: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
    top_n: int = WATCH_TOP_N,
) -> list[dict]:
    """
    Run every saved watch. Returns one report per watch:
        {watch, n_found, results: [(PaperMeta, score, label), ...]}
    Watches' last_checked dates are advanced to today on success.
    """
    say = progress or (lambda s: None)
    watches = load_watches(path)
    reports: list[dict] = []
    today = time.strftime("%Y-%m-%d")

    for w in watches:
        terms = " ".join([w["query"]] + w.get("keywords", [])).strip()
        say(f"Checking watch: '{w['query']}' (since {w['last_checked']}) ...")
        found = _search_since(terms, w["last_checked"])

        # Drop anything already in the graph, then rank with the user's model.
        ranked = graph.rank_candidates(found, n=top_n, exploration_ratio=0.0) \
                 if found else []
        reports.append({
            "watch":   dict(w),
            "n_found": len(found),
            "results": ranked,
        })
        audit.log_event("watch_check", query=w["query"], n_new=len(ranked))
        w["last_checked"] = today

    if watches:
        save_watches(watches, path)
    return reports
