"""
searcher.py
Fetch candidate papers from Semantic Scholar and ArXiv (both free, no key needed).
Falls back gracefully if a source is unreachable.

LLM enhancements (optional, degrade gracefully when Ollama is unavailable):
  * HyDE — Hypothetical Document Embeddings for superior semantic matching
  * Query expansion — LLM generates alternative search formulations
  * LLM reranking — reranks top candidates by semantic relevance
"""

from __future__ import annotations

import json
import time
import re
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import requests

from researchbuddy.config import (
    S2_SEARCH_URL, S2_REC_URL,
    ARXIV_SEARCH_URL, MAX_SEARCH_RESULTS, REQUEST_TIMEOUT, REQUEST_DELAY,
    S2_SEARCH_QUERIES, S2_SEARCH_LIMIT, ARXIV_SEARCH_QUERIES, ARXIV_SEARCH_LIMIT,
)
from researchbuddy.core.graph_model import PaperMeta, ResearchGraph


_HEADERS = {"User-Agent": "ResearchBuddy/0.1 (local research assistant)"}
S2_FIELDS = "paperId,title,abstract,authors,year,externalIds,url,publicationVenue"


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

    # ── Publication venue / peer-review status ──────────────────────
    venue_info = item.get("publicationVenue") or {}
    venue_name = venue_info.get("name", "")
    venue_type = venue_info.get("type", "")   # "Journal", "Conference", "Book", etc.
    if venue_type in ("Journal", "Conference", "Book"):
        is_peer_reviewed = True
    elif arxiv_id and not venue_name:
        is_peer_reviewed = False
    else:
        is_peer_reviewed = None   # unknown

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
        venue    = venue_name[:200],
        is_peer_reviewed = is_peer_reviewed,
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
            venue    = "arXiv",
            is_peer_reviewed = False,
            source   = "discovered",
        ))

    time.sleep(REQUEST_DELAY)
    return results


# ── LLM-enhanced search helpers ───────────────────────────────────────────────

def _generate_hyde_abstract(query: str) -> Optional[str]:
    """
    HyDE — Hypothetical Document Embedding.

    Ask the LLM to write a short *hypothetical* abstract for an ideal paper
    that would perfectly answer ``query``.  This abstract is then embedded
    and used as a supplementary search signal, dramatically improving
    semantic matching compared to embedding the raw query string.

    Returns None when Ollama is unavailable or generation fails.
    """
    from researchbuddy.config import HYDE_ENABLED, LLM_ENABLED
    if not HYDE_ENABLED or not LLM_ENABLED:
        return None

    try:
        from researchbuddy.core.llm import get_llm
        client = get_llm()
        if not client.is_available():
            return None

        prompt = (
            f"Write a short academic abstract (100-150 words) for a hypothetical "
            f"research paper that would perfectly answer this research question:\n\n"
            f"\"{query}\"\n\n"
            f"The abstract should include typical academic language, mention "
            f"methodology, key findings, and conclusions. Output ONLY the "
            f"abstract text, no title or labels."
        )
        system = (
            "You are an academic paper abstract generator. Write realistic, "
            "concise scientific abstracts. Output only the abstract paragraph."
        )
        result = client.generate(prompt, system=system, temperature=0.5, max_tokens=256)
        if result and len(result) > 50:
            print(f"  [HyDE] Generated hypothetical abstract ({len(result)} chars)")
            return result
    except Exception as e:
        print(f"  [HyDE] Failed: {e}")
    return None


def _expand_query(query: str, keywords: list[str]) -> list[str]:
    """
    LLM query expansion — generate alternative search formulations.

    Given the user's research intent and existing keywords, generate 3
    complementary search queries that cover different facets of the topic.
    Returns an empty list when LLM is unavailable.
    """
    from researchbuddy.config import LLM_QUERY_EXPANSION, LLM_ENABLED
    if not LLM_QUERY_EXPANSION or not LLM_ENABLED:
        return []

    try:
        from researchbuddy.core.llm import get_llm
        client = get_llm()
        if not client.is_available():
            return []

        kw_str = ", ".join(keywords[:6]) if keywords else "(none)"
        prompt = (
            f"Given this research question: \"{query}\"\n"
            f"And these existing keywords: {kw_str}\n\n"
            f"Generate exactly 3 alternative academic search queries that would "
            f"find relevant papers. Each query should be 3-8 words and cover a "
            f"different aspect or use different terminology.\n\n"
            f"Output as a JSON array of 3 strings, e.g.:\n"
            f'[\"query one\", \"query two\", \"query three\"]'
        )
        result = client.generate_json(
            prompt,
            system="You generate academic search queries. Output valid JSON only.",
            temperature=0.4,
            max_tokens=128,
        )
        if isinstance(result, list) and len(result) >= 1:
            expanded = [q.strip() for q in result if isinstance(q, str) and len(q.strip()) >= 5]
            if expanded:
                print(f"  [LLM expand] +{len(expanded)} queries: "
                      f"{', '.join(q[:40] for q in expanded[:3])}")
                return expanded[:3]
    except Exception as e:
        print(f"  [LLM expand] Failed: {e}")
    return []


def _llm_rerank(query: str, candidates: list[PaperMeta], top_n: int = 15) -> list[PaperMeta]:
    """
    LLM reranking — rerank the top candidates by semantic relevance.

    Sends titles and abstract snippets of top candidates to the LLM
    and asks it to rank them by relevance to the query. Returns reordered
    candidates (or original order on failure).
    """
    from researchbuddy.config import LLM_RERANK_ENABLED, LLM_ENABLED
    if not LLM_RERANK_ENABLED or not LLM_ENABLED:
        return candidates

    if len(candidates) <= 3:
        return candidates

    try:
        from researchbuddy.core.llm import get_llm, fix_mojibake
        client = get_llm()
        if not client.is_available():
            return candidates

        # Take top candidates for reranking
        to_rerank = candidates[:top_n]
        rest = candidates[top_n:]

        # Build paper descriptions for LLM
        papers_desc = []
        for i, p in enumerate(to_rerank):
            title = fix_mojibake(p.title or "Untitled")[:100]
            abstract = fix_mojibake(p.abstract or "")[:150]
            papers_desc.append(f"{i}: \"{title}\" — {abstract}")

        prompt = (
            f"Research question: \"{query}\"\n\n"
            f"Rank these papers by relevance to the research question. "
            f"Return a JSON array of paper indices (0-based) from most to "
            f"least relevant.\n\n"
            + "\n".join(papers_desc)
            + f"\n\nOutput ONLY a JSON array of integers, e.g. [3, 0, 5, 1, ...]"
        )

        result = client.generate_json(
            prompt,
            system="You are a research paper relevance ranker. Output valid JSON only.",
            temperature=0.2,
            max_tokens=128,
        )
        if isinstance(result, list) and len(result) >= 3:
            # Validate indices
            valid_indices = [int(i) for i in result
                            if isinstance(i, (int, float)) and 0 <= int(i) < len(to_rerank)]
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in valid_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            if len(unique_indices) >= 3:
                # Add any missing indices at the end
                for i in range(len(to_rerank)):
                    if i not in seen:
                        unique_indices.append(i)

                reranked = [to_rerank[i] for i in unique_indices] + rest
                print(f"  [LLM rerank] Reranked {len(unique_indices)} candidates")
                return reranked

    except Exception as e:
        print(f"  [LLM rerank] Failed: {e}")
    return candidates


# ── High-level orchestrator ────────────────────────────────────────────────────

def find_candidates(
    graph: ResearchGraph,
    extra_keywords: list[str] | None = None,
    query: str | None = None,
) -> tuple[list[PaperMeta], Optional[np.ndarray]]:
    """
    Run all search strategies and return deduplicated candidate papers.
    Order: S2 recommendations → S2 text search → ArXiv text search.

    When ``query`` (research intent) is provided, LLM enhancements kick in:
      * HyDE: generate a hypothetical abstract, embed it for better matching
      * Query expansion: LLM generates alternative search formulations
      * LLM reranking: reranks top candidates by relevance

    Returns (candidates, hyde_embedding) where hyde_embedding is None when
    LLM is unavailable or no query is given.
    """
    all_candidates: list[PaperMeta] = []
    seen_ids: set[str] = set()
    hyde_embedding: Optional[np.ndarray] = None

    def add(papers: list[PaperMeta]):
        for p in papers:
            if p.paper_id not in seen_ids:
                seen_ids.add(p.paper_id)
                all_candidates.append(p)

    # ── HyDE: generate hypothetical abstract ─────────────────────────────
    if query:
        hyde_abstract = _generate_hyde_abstract(query)
        if hyde_abstract:
            try:
                from researchbuddy.core.embedder import embed
                hyde_embedding = embed(hyde_abstract)
            except Exception:
                hyde_embedding = None

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

    # ── LLM query expansion ──────────────────────────────────────────────
    if query:
        expanded = _expand_query(query, keywords)
        queries = expanded + queries   # LLM expansions go first

    for q in queries[:S2_SEARCH_QUERIES]:
        print(f"  [search] S2 search: '{q[:60]}' ...")
        add(search_semantic_scholar(q, limit=S2_SEARCH_LIMIT))

    for q in queries[:ARXIV_SEARCH_QUERIES]:
        print(f"  [search] ArXiv search: '{q[:60]}' ...")
        add(search_arxiv(q, limit=ARXIV_SEARCH_LIMIT))

    # ── LLM reranking ────────────────────────────────────────────────────
    if query and all_candidates:
        all_candidates = _llm_rerank(query, all_candidates)

    # ── Soft prioritization: peer-reviewed before preprints ──────────
    all_candidates.sort(
        key=lambda p: (getattr(p, "is_peer_reviewed", None) is not False),
        reverse=True,
    )

    print(f"  [search] Total candidates fetched: {len(all_candidates)}")
    return all_candidates, hyde_embedding


def resolve_s2_id(title: str) -> str:
    """Try to find a Semantic Scholar paper ID by title search."""
    results = search_semantic_scholar(title, limit=3)
    return results[0].s2_id if results else ""
