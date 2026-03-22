"""
searcher.py
Fetch candidate papers from Semantic Scholar and ArXiv (both free, no key needed).
Falls back gracefully if a source is unreachable.

LLM enhancements (optional, degrade gracefully when Ollama is unavailable):
  * HyDE â€" Hypothetical Document Embeddings for superior semantic matching
  * Query expansion â€" LLM generates alternative search formulations
  * LLM reranking â€" reranks top candidates by semantic relevance
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import time
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Optional

import numpy as np
import requests

from researchbuddy.config import (
    S2_SEARCH_URL, S2_REC_URL, S2_PAPER_URL,
    ARXIV_SEARCH_URL, MAX_SEARCH_RESULTS, REQUEST_TIMEOUT, REQUEST_DELAY,
    S2_SEARCH_QUERIES, S2_SEARCH_LIMIT, ARXIV_SEARCH_QUERIES, ARXIV_SEARCH_LIMIT,
    COLD_START_THRESHOLD,
)
from researchbuddy.core.graph_model import PaperMeta, ResearchGraph

import logging
logger = logging.getLogger(__name__)


_HEADERS = {"User-Agent": "ResearchBuddy/0.1 (local research assistant)"}
_S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
if _S2_API_KEY:
    _HEADERS["x-api-key"] = _S2_API_KEY

S2_FIELDS = "paperId,title,abstract,authors,year,externalIds,url,publicationVenue"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_HTTP_RETRIES = 4
_BACKOFF_BASE_SECONDS = max(float(REQUEST_DELAY), 0.5)
_BACKOFF_MAX_SECONDS = 20.0
_BACKOFF_JITTER_SECONDS = 0.25
_S2_COOLDOWN_SECONDS = float(os.getenv("S2_RATE_LIMIT_COOLDOWN_SECONDS", "90"))
_s2_backoff_until = 0.0


def _is_s2_url(url: str) -> bool:
    return "api.semanticscholar.org" in url


def _s2_cooldown_remaining() -> float:
    return max(0.0, _s2_backoff_until - time.time())


def _activate_s2_cooldown(response: Optional[requests.Response] = None):
    global _s2_backoff_until
    delay = _S2_COOLDOWN_SECONDS
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
    _s2_backoff_until = max(_s2_backoff_until, time.time() + delay)
    logger.warning(
        f"S2 cooldown active for {delay:.0f}s after rate-limit. "
        "Set SEMANTIC_SCHOLAR_API_KEY to improve limits."
    )


def _retry_delay_seconds(response: Optional[requests.Response], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))
    try:
        from researchbuddy.config import DETERMINISTIC_MODE
    except Exception:
        DETERMINISTIC_MODE = False
    if DETERMINISTIC_MODE:
        return delay
    return delay + random.uniform(0.0, _BACKOFF_JITTER_SECONDS)


def _cache_file(namespace: str, payload: dict) -> Optional[Path]:
    """Return cache file path for a deterministic payload hash."""
    try:
        from researchbuddy.config import (
            SEARCH_CACHE_DIR,
            SEARCH_CACHE_ENABLED,
            SEARCH_CACHE_VERSION,
        )
    except Exception:
        return None

    if not SEARCH_CACHE_ENABLED:
        return None

    try:
        cache_dir = Path(SEARCH_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_key = json.dumps(
            {
                "version": SEARCH_CACHE_VERSION,
                "namespace": namespace,
                "payload": payload,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return cache_dir / f"{namespace}_{digest}.json"
    except Exception:
        return None


def _cache_load(namespace: str, payload: dict) -> Optional[Any]:
    path = _cache_file(namespace, payload)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_save(namespace: str, payload: dict, value: Any):
    path = _cache_file(namespace, payload)
    if path is None:
        return
    try:
        path.write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        pass

# â"€â"€ Internal helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _get(url: str, params: dict) -> Optional[dict | str]:
    if _is_s2_url(url):
        cooldown = _s2_cooldown_remaining()
        if cooldown > 0:
            logger.debug(f"S2 cooldown active ({cooldown:.0f}s), skipping request.")
            return None

    response: Optional[requests.Response] = None
    for attempt in range(_MAX_HTTP_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_HTTP_RETRIES:
                delay = _retry_delay_seconds(response, attempt)
                logger.debug(
                    f"GET retry {attempt + 1}/{_MAX_HTTP_RETRIES} "
                    f"(status={response.status_code}) in {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            ct = response.headers.get("Content-Type", "")
            return response.json() if "json" in ct else response.text
        except requests.RequestException as e:
            if attempt < _MAX_HTTP_RETRIES:
                delay = _retry_delay_seconds(response, attempt)
                logger.debug(
                    f"GET error (attempt {attempt + 1}/{_MAX_HTTP_RETRIES}): {e}; "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            if response is not None and response.status_code == 429 and _is_s2_url(url):
                _activate_s2_cooldown(response)
            logger.warning(f"Request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return None
    return None

def _post(url: str, payload: dict) -> Optional[dict]:
    if _is_s2_url(url):
        cooldown = _s2_cooldown_remaining()
        if cooldown > 0:
            logger.debug(f"S2 cooldown active ({cooldown:.0f}s), skipping POST request.")
            return None

    response: Optional[requests.Response] = None
    for attempt in range(_MAX_HTTP_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=_HEADERS, timeout=REQUEST_TIMEOUT)

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_HTTP_RETRIES:
                delay = _retry_delay_seconds(response, attempt)
                logger.debug(
                    f"POST retry {attempt + 1}/{_MAX_HTTP_RETRIES} "
                    f"(status={response.status_code}) in {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < _MAX_HTTP_RETRIES:
                delay = _retry_delay_seconds(response, attempt)
                logger.debug(
                    f"POST error (attempt {attempt + 1}/{_MAX_HTTP_RETRIES}): {e}; "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            if response is not None and response.status_code == 429 and _is_s2_url(url):
                _activate_s2_cooldown(response)
            logger.warning(f"POST failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"POST failed: {e}")
            return None
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

    # â"€â"€ Publication venue / peer-review status â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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


# â"€â"€ Semantic Scholar â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# ── Forward citations ──────────────────────────────────────────────────────────

def fetch_forward_citations(s2_id: str, limit: int = 20) -> list[PaperMeta]:
    """
    Fetch papers that CITE the given S2 paper (forward/incoming citations).

    This is complementary to recommendations: recommendations find similar
    papers, while forward citations find papers that *build on* this one —
    often the most directly relevant follow-up work.
    """
    if not s2_id:
        return []
    url = f"{S2_PAPER_URL}/{s2_id}/citations"
    data = _get(url, {"fields": S2_FIELDS, "limit": min(limit, 100)})
    if not data or not isinstance(data, dict):
        return []
    results = []
    for item in data.get("data", []):
        citing = item.get("citingPaper", {})
        m = _s2_to_meta(citing)
        if m:
            results.append(m)
    time.sleep(REQUEST_DELAY)
    return results


# â"€â"€ ArXiv â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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


# â"€â"€ LLM-enhanced search helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _generate_hyde_abstract(query: str) -> Optional[str]:
    """
    HyDE - Hypothetical Document Embedding.

    Ask the LLM to write a short hypothetical abstract for an ideal paper
    that would perfectly answer ``query``. This abstract is then embedded
    and used as a supplementary search signal.

    Returns None when Ollama is unavailable or generation fails.
    """
    from researchbuddy.config import (
        DETERMINISTIC_MODE,
        HYDE_ENABLED,
        LLM_ENABLED,
        LLM_MODEL,
    )
    if not HYDE_ENABLED or not LLM_ENABLED:
        return None

    cache_payload = {"query": query.strip(), "model": LLM_MODEL}
    cached = _cache_load("hyde", cache_payload)
    if isinstance(cached, str) and len(cached) > 50:
        logger.debug(f"[HyDE] Cache hit ({len(cached)} chars)")
        return cached

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
        temperature = 0.0 if DETERMINISTIC_MODE else 0.5
        result = client.generate(prompt, system=system, temperature=temperature, max_tokens=256)
        if result and len(result) > 50:
            _cache_save("hyde", cache_payload, result)
            logger.info(f"[HyDE] Generated hypothetical abstract ({len(result)} chars)")
            return result
    except Exception as e:
        logger.warning(f"[HyDE] Failed: {e}")
    return None


def _expand_query(query: str, keywords: list[str]) -> list[str]:
    """
    LLM query expansion - generate alternative search formulations.

    Given the user's research intent and existing keywords, generate 3
    complementary search queries that cover different facets of the topic.
    Returns an empty list when LLM is unavailable.
    """
    from researchbuddy.config import (
        DETERMINISTIC_MODE,
        LLM_ENABLED,
        LLM_MODEL,
        LLM_QUERY_EXPANSION,
    )
    if not LLM_QUERY_EXPANSION or not LLM_ENABLED:
        return []

    cache_payload = {
        "query": query.strip(),
        "keywords": [k.strip().lower() for k in keywords if k and k.strip()],
        "model": LLM_MODEL,
    }
    cached = _cache_load("expand_query", cache_payload)
    if isinstance(cached, list):
        expanded = [
            q.strip() for q in cached
            if isinstance(q, str) and len(q.strip()) >= 5
        ]
        if expanded:
            logger.debug(f"[LLM expand] Cache hit (+{len(expanded)} queries)")
            return expanded[:3]

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
            f'["query one", "query two", "query three"]'
        )
        temperature = 0.0 if DETERMINISTIC_MODE else 0.4
        result = client.generate_json(
            prompt,
            system="You generate academic search queries. Output valid JSON only.",
            temperature=temperature,
            max_tokens=128,
        )
        if isinstance(result, list) and len(result) >= 1:
            expanded = [
                q.strip() for q in result
                if isinstance(q, str) and len(q.strip()) >= 5
            ]
            # De-duplicate while preserving order.
            expanded = list(dict.fromkeys(expanded))
            if expanded:
                _cache_save("expand_query", cache_payload, expanded[:3])
                logger.info(f"[LLM expand] +{len(expanded)} queries: "
                      f"{', '.join(q[:40] for q in expanded[:3])}")
                return expanded[:3]
    except Exception as e:
        logger.warning(f"[LLM expand] Failed: {e}")
    return []


def _llm_rerank(query: str, candidates: list[PaperMeta], top_n: int = 15) -> list[PaperMeta]:
    """
    LLM reranking - rerank the top candidates by semantic relevance.

    Sends titles and abstract snippets of top candidates to the LLM
    and asks it to rank them by relevance to the query. Returns reordered
    candidates (or original order on failure).
    """
    from researchbuddy.config import (
        DETERMINISTIC_MODE,
        LLM_ENABLED,
        LLM_MODEL,
        LLM_RERANK_ENABLED,
    )
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

        def _normalize_indices(raw_indices: list[Any]) -> list[int]:
            valid = [
                int(i)
                for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < len(to_rerank)
            ]
            seen: set[int] = set()
            ordered: list[int] = []
            for idx in valid:
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
            if len(ordered) < 3:
                return []
            for i in range(len(to_rerank)):
                if i not in seen:
                    ordered.append(i)
            return ordered

        cache_payload = {
            "query": query.strip(),
            "top_n": int(top_n),
            "candidate_ids": [p.paper_id for p in to_rerank],
            "model": LLM_MODEL,
        }
        cached = _cache_load("rerank", cache_payload)
        if isinstance(cached, list):
            ordered = _normalize_indices(cached)
            if ordered:
                reranked = [to_rerank[i] for i in ordered] + rest
                logger.debug(f"[LLM rerank] Cache hit ({len(ordered)} candidates)")
                return reranked

        # Build paper descriptions for LLM
        papers_desc = []
        for i, p in enumerate(to_rerank):
            title = fix_mojibake(p.title or "Untitled")[:100]
            abstract = fix_mojibake(p.abstract or "")[:150]
            papers_desc.append(f"{i}: \"{title}\" - {abstract}")

        prompt = (
            f"Research question: \"{query}\"\n\n"
            f"Rank these papers by relevance to the research question. "
            f"Return a JSON array of paper indices (0-based) from most to "
            f"least relevant.\n\n"
            + "\n".join(papers_desc)
            + f"\n\nOutput ONLY a JSON array of integers, e.g. [3, 0, 5, 1, ...]"
        )

        temperature = 0.0 if DETERMINISTIC_MODE else 0.2
        result = client.generate_json(
            prompt,
            system="You are a research paper relevance ranker. Output valid JSON only.",
            temperature=temperature,
            max_tokens=128,
        )
        if isinstance(result, list) and len(result) >= 3:
            ordered = _normalize_indices(result)
            if ordered:
                _cache_save("rerank", cache_payload, ordered)
                reranked = [to_rerank[i] for i in ordered] + rest
                logger.info(f"[LLM rerank] Reranked {len(ordered)} candidates")
                return reranked

    except Exception as e:
        logger.warning(f"[LLM rerank] Failed: {e}")
    return candidates


# â"€â"€ High-level orchestrator â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def find_candidates(
    graph: ResearchGraph,
    extra_keywords: list[str] | None = None,
    query: str | None = None,
) -> tuple[list[PaperMeta], Optional[np.ndarray]]:
    """
    Run all search strategies and return deduplicated candidate papers.
    Order: S2 recommendations â†’ S2 text search â†’ ArXiv text search.

    When ``query`` (research intent) is provided, LLM enhancements kick in:
      * HyDE: generate a hypothetical abstract, embed it for better matching
      * Query expansion: LLM generates alternative search formulations
      * LLM reranking: reranks top candidates by relevance

    Returns (candidates, hyde_embedding) where hyde_embedding is None when
    LLM is unavailable or no query is given.
    """
    all_candidates: list[PaperMeta] = []
    seen_ids: set[str] = set()
    seen_arxiv_ids: set[str] = set()
    seen_dois: set[str] = set()
    hyde_embedding: Optional[np.ndarray] = None
    from researchbuddy.config import DETERMINISTIC_MODE

    def add(papers: list[PaperMeta]):
        for p in papers:
            # Primary dedup by paper_id
            if p.paper_id in seen_ids:
                continue
            # Cross-source dedup: same ArXiv preprint may appear from S2 and ArXiv
            if p.arxiv_id and p.arxiv_id in seen_arxiv_ids:
                continue
            # Cross-source dedup: same DOI from different S2 IDs
            if p.doi and p.doi in seen_dois:
                continue
            seen_ids.add(p.paper_id)
            if p.arxiv_id:
                seen_arxiv_ids.add(p.arxiv_id)
            if p.doi:
                seen_dois.add(p.doi)
            all_candidates.append(p)

    # ── HyDE: generate hypothetical abstract ─────────────────────────────────
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
        logger.info("Fetching S2 recommendations ...")
        add(get_s2_recommendations(pos_ids, neg_ids))

    # ── Forward citation expansion: papers that build on highly-rated work ───
    # Cap at 3 papers to limit API calls; skip if S2 cooldown is active
    if pos_ids and not _s2_cooldown_remaining():
        for s2_id in pos_ids[:3]:
            logger.info(f"Fetching forward citations for S2:{s2_id[:12]} ...")
            add(fetch_forward_citations(s2_id, limit=15))

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
        key=lambda m: (-(m.user_rating or 0), m.paper_id),
    )[:2]
    for m in top_rated:
        queries.append(m.title)

    # â"€â"€ LLM query expansion â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if query:
        expanded = _expand_query(query, keywords)
        queries = expanded + queries   # LLM expansions go first

    queries = [q.strip() for q in queries if q and q.strip()]
    queries = list(dict.fromkeys(queries))

    # In cold-start mode, expand search breadth to compensate for thin graph
    is_cold_start = len(graph.all_papers()) < COLD_START_THRESHOLD
    s2_cap = S2_SEARCH_QUERIES if _S2_API_KEY else min(S2_SEARCH_QUERIES, 3)
    arxiv_cap = ARXIV_SEARCH_QUERIES
    if is_cold_start:
        s2_cap = min(s2_cap + 3, len(queries))      # more text queries
        arxiv_cap = min(arxiv_cap + 2, len(queries)) # broader ArXiv coverage
        logger.info("Cold-start mode: expanded search queries (S2=%d, ArXiv=%d)",
                     s2_cap, arxiv_cap)

    for q in queries[:s2_cap]:
        logger.info(f"S2 search: '{q[:60]}' ...")
        add(search_semantic_scholar(q, limit=S2_SEARCH_LIMIT))

    for q in queries[:arxiv_cap]:
        logger.info(f"ArXiv search: '{q[:60]}' ...")
        add(search_arxiv(q, limit=ARXIV_SEARCH_LIMIT))

    # â"€â"€ LLM reranking â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if query and all_candidates:
        all_candidates = _llm_rerank(query, all_candidates)

    # â"€â"€ Soft prioritization: peer-reviewed before preprints â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if DETERMINISTIC_MODE:
        all_candidates.sort(
            key=lambda p: (
                0 if getattr(p, "is_peer_reviewed", None) is not False else 1,
                (p.title or "").lower(),
                p.paper_id,
            )
        )
    else:
        all_candidates.sort(
            key=lambda p: (getattr(p, "is_peer_reviewed", None) is not False),
            reverse=True,
        )

    logger.info(f"Total candidates fetched: {len(all_candidates)}")
    return all_candidates, hyde_embedding


def resolve_s2_id(title: str) -> str:
    """Try to find a Semantic Scholar paper ID by title search."""
    results = search_semantic_scholar(title, limit=3)
    return results[0].s2_id if results else ""
