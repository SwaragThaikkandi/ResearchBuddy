"""
searcher.py
Fetch candidate papers from Semantic Scholar and ArXiv (both free, no key needed).
Falls back gracefully if a source is unreachable.
"""

from __future__ import annotations

import time
import re
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from researchbuddy.config import (
    S2_SEARCH_URL, S2_REC_URL,
    ARXIV_SEARCH_URL, MAX_SEARCH_RESULTS, REQUEST_TIMEOUT, REQUEST_DELAY,
)
from researchbuddy.core.graph_model import PaperMeta, ResearchGraph


_HEADERS = {"User-Agent": "ResearchBuddy/0.1 (local research assistant)"}
S2_FIELDS = "paperId,title,abstract,authors,year,externalIds,url"


# ── Internal helpers ────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> Optional[dict | str]:
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        return r.json() if "json" in ct else r.text
    except Exception as e:
        print(f"  [searcher] Request failed: {e}")
        return None


def _post(url: str, payload: dict) -> Optional[dict]:
    try:
        r = requests.post(url, json=payload, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [searcher] POST failed: {e}")
        return None


def _s2_to_meta(item: dict) -> Optional[PaperMeta]:
    title = item.get("title", "").strip()
    if not title:
        return None
    s2_id    = item.get("paperId", "")
    abstract = item.get("abstract") or ""
    authors  = [a.get("name", "") for a in item.get("authors", [])]
    year     = item.get("year")
    ext_ids  = item.get("externalIds") or {}
    doi      = ext_ids.get("DOI", "")
    arxiv_id = ext_ids.get("ArXiv", "")
    url      = item.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")

    paper_id = ResearchGraph.make_id(title, doi=doi, s2_id=s2_id)
    return PaperMeta(
        paper_id = paper_id,
        title    = title[:250],
        abstract = abstract[:2000],
        authors  = authors[:10],
        year     = year,
        url      = url,
        doi      = doi,
        s2_id    = s2_id,
        arxiv_id = arxiv_id,
        source   = "discovered",
    )


# ── Semantic Scholar ───────────────────────────────────────────────────────────

def search_semantic_scholar(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[PaperMeta]:
    data = _get(S2_SEARCH_URL, {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS})
    if not data or not isinstance(data, dict):
        return []
    results = []
    for item in data.get("data", []):
        m = _s2_to_meta(item)
        if m:
            results.append(m)
    time.sleep(REQUEST_DELAY)
    return results


def get_s2_recommendations(
    positive_ids: list[str],
    negative_ids: list[str] | None = None,
    limit: int = MAX_SEARCH_RESULTS,
) -> list[PaperMeta]:
    """Use S2 recommendations endpoint (positive/negative paper ID lists)."""
    if not positive_ids:
        return []
    payload = {
        "positivePaperIds": positive_ids[:10],
        "negativePaperIds": (negative_ids or [])[:5],
    }
    data = _post(f"{S2_REC_URL}?fields={S2_FIELDS}&limit={limit}", payload)
    if not data:
        return []
    results = []
    for item in data.get("recommendedPapers", []):
        m = _s2_to_meta(item)
        if m:
            results.append(m)
    time.sleep(REQUEST_DELAY)
    return results


# ── ArXiv ─────────────────────────────────────────────────────────────────────

def search_arxiv(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[PaperMeta]:
    xml_text = _get(ARXIV_SEARCH_URL, {
        "search_query": f"all:{query}",
        "start"       : 0,
        "max_results" : min(limit, 100),
        "sortBy"      : "relevance",
        "sortOrder"   : "descending",
    })
    if not xml_text or not isinstance(xml_text, str):
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el    = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        id_el       = entry.find("atom:id", ns)

        title    = title_el.text.strip()    if title_el    is not None else ""
        abstract = abstract_el.text.strip() if abstract_el is not None else ""
        raw_id   = id_el.text.strip()       if id_el       is not None else ""

        arxiv_id = re.sub(r'v\d+$', '', raw_id.split("/abs/")[-1])
        url      = f"https://arxiv.org/abs/{arxiv_id}"

        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
        year_el = entry.find("atom:published", ns)
        year    = int(year_el.text[:4]) if year_el is not None else None

        paper_id = ResearchGraph.make_id(title, arxiv_id=arxiv_id)
        results.append(PaperMeta(
            paper_id = paper_id,
            title    = title[:250],
            abstract = abstract[:2000],
            authors  = authors[:10],
            year     = year,
            url      = url,
            arxiv_id = arxiv_id,
            source   = "discovered",
        ))

    time.sleep(REQUEST_DELAY)
    return results


# ── High-level orchestrator ────────────────────────────────────────────────────

def find_candidates(graph: ResearchGraph, extra_keywords: list[str] | None = None) -> list[PaperMeta]:
    """
    Run all search strategies and return deduplicated candidate papers.
    Order: S2 recommendations → S2 text search → ArXiv text search.
    """
    all_candidates: list[PaperMeta] = []
    seen_ids: set[str] = set()

    def add(papers: list[PaperMeta]):
        for p in papers:
            if p.paper_id not in seen_ids:
                seen_ids.add(p.paper_id)
                all_candidates.append(p)

    # S2 Recommendations from highly-rated / seed papers
    pos_ids = [m.s2_id for m in graph.all_papers() if m.s2_id and m.effective_weight >= 6][:10]
    neg_ids = [m.s2_id for m in graph.rated_papers() if m.s2_id and m.user_rating is not None and m.user_rating <= 3][:5]

    if pos_ids:
        print("  [search] Fetching S2 recommendations ...")
        add(get_s2_recommendations(pos_ids, neg_ids))

    # Build search queries from keywords + top-rated paper titles
    keywords = graph.top_seed_keywords(n=6)
    if extra_keywords:
        keywords = list(dict.fromkeys(extra_keywords + keywords))
    if not keywords:
        keywords = ["machine learning", "deep learning"]

    queries = []
    if keywords:
        queries.append(" ".join(keywords[:3]))
    if len(keywords) > 3:
        queries.append(" ".join(keywords[3:6]))

    top_rated = sorted(
        [m for m in graph.rated_papers() if m.user_rating and m.user_rating >= 7],
        key=lambda m: m.user_rating, reverse=True
    )[:2]
    for m in top_rated:
        queries.append(m.title)

    for query in queries[:3]:
        print(f"  [search] S2 search: '{query[:60]}' ...")
        add(search_semantic_scholar(query, limit=15))

    for query in queries[:2]:
        print(f"  [search] ArXiv search: '{query[:60]}' ...")
        add(search_arxiv(query, limit=15))

    print(f"  [search] Total candidates fetched: {len(all_candidates)}")
    return all_candidates


def resolve_s2_id(title: str) -> str:
    """Try to find a Semantic Scholar paper ID by title search."""
    results = search_semantic_scholar(title, limit=3)
    return results[0].s2_id if results else ""
