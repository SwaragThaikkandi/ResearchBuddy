"""
core_fetcher.py
Fetch full text for papers from CORE (core.ac.uk).

CORE is a free aggregator of open-access full texts covering 220M+ works.
It is especially useful for ArXiv papers and gold/green open-access journals.

Setup (optional but recommended):
  1. Register for a free API key at https://core.ac.uk/services/api
  2. Set the CORE_API_KEY environment variable before running ResearchBuddy.
     Anonymous access works but is rate-limited to ~1 req/s.

What this module does:
  - Queries CORE by DOI → ArXiv ID → title (in order of reliability)
  - Strips equation-dense lines from returned plain text
  - Caches results to disk so each paper is only fetched once
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
import logging

logger = logging.getLogger(__name__)

_CORE_API_KEY = os.getenv("CORE_API_KEY", "").strip()

_HEADERS = {"User-Agent": "ResearchBuddy/0.4 (research assistant; "
                           "mailto:researchbuddy@example.com)"}
if _CORE_API_KEY:
    _HEADERS["Authorization"] = f"Bearer {_CORE_API_KEY}"

_REQUEST_TIMEOUT = 20
# Polite rate: 1.1 s without key, 0.15 s with key
_REQUEST_DELAY = 0.15 if _CORE_API_KEY else 1.1

# ── Equation / noise stripping ─────────────────────────────────────────────────

# Lines where fewer than 35% of characters are alphabetic are likely equations,
# tables, or reference lists — all low-value noise for semantic embeddings.
_MIN_ALPHA_RATIO = 0.35

# Lines matching this pattern are almost certainly standalone math expressions.
_MATH_LINE_RE = re.compile(
    r'^[\s\d\+\-\*\/\=\(\)\[\]\{\}\^\|_<>,.;:\'"\\~`@#%&]+$'
)


def strip_equations(text: str) -> str:
    """
    Remove equation-dense and symbol-heavy lines from plain text.

    CORE returns text extracted from PDFs. Equations appear as lines packed
    with digits, operators, and special characters. These lines add noise
    to semantic embeddings without conveying conceptual meaning.
    """
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")   # preserve paragraph breaks
            continue
        # Skip very short lines that are just symbols
        if len(s) <= 4:
            continue
        alpha = sum(c.isalpha() for c in s)
        if alpha < len(s) * _MIN_ALPHA_RATIO:
            continue
        if _MATH_LINE_RE.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


# ── Disk cache ─────────────────────────────────────────────────────────────────

_CACHE_MISS = object()   # sentinel: file not in cache (different from "tried, not found")


def _cache_dir() -> Optional[Path]:
    try:
        from researchbuddy.config import DATA_DIR, CORE_FULL_TEXT
        if not CORE_FULL_TEXT:
            return None
        d = Path(DATA_DIR) / "cache" / "core"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        return None


def _cache_key(doi: str, arxiv_id: str, title: str) -> str:
    raw = "|".join(filter(None, [
        doi.strip().lower(),
        arxiv_id.strip().lower(),
        title.strip().lower()[:80],
    ]))
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_load(key: str):
    """Returns _CACHE_MISS (not cached), None (cached: not found), or str (cached: found)."""
    d = _cache_dir()
    if d is None:
        return _CACHE_MISS
    path = d / f"{key}.json"
    if not path.exists():
        return _CACHE_MISS
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fulltext")
    except Exception:
        return _CACHE_MISS


def _cache_save(key: str, fulltext: Optional[str]) -> None:
    d = _cache_dir()
    if d is None:
        return
    try:
        (d / f"{key}.json").write_text(
            json.dumps({"fulltext": fulltext}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# ── CORE API calls ─────────────────────────────────────────────────────────────

def _search_core(query: str) -> Optional[str]:
    """Send one CORE search query and return fullText of the first hit, or None."""
    try:
        from researchbuddy.config import CORE_API_URL
    except Exception:
        CORE_API_URL = "https://api.core.ac.uk/v3"

    try:
        r = requests.get(
            f"{CORE_API_URL}/search/works",
            params={"q": query, "limit": 1},
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        if r.status_code == 401:
            logger.warning("[CORE] 401 Unauthorized — is CORE_API_KEY set correctly?")
            return None
        if r.status_code == 429:
            logger.warning("[CORE] 429 Rate limited — consider registering for a free API key")
            return None
        if r.status_code != 200:
            logger.debug("[CORE] HTTP %d for query: %s", r.status_code, query[:60])
            return None
        results = r.json().get("results", [])
        if not results:
            return None
        ft = (results[0].get("fullText") or "").strip()
        return ft if ft else None
    except Exception as e:
        logger.debug("[CORE] Request error: %s", e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_fulltext(
    doi: str = "",
    arxiv_id: str = "",
    title: str = "",
) -> Optional[str]:
    """
    Fetch and return cleaned full text for a paper, or None if unavailable.

    Tries in order: DOI → ArXiv ID → title.
    Results are cached; each paper is only fetched from CORE once.

    Parameters
    ----------
    doi       : paper DOI  (e.g. "10.1038/s41586-021-03819-2")
    arxiv_id  : ArXiv ID   (e.g. "2106.09685")
    title     : paper title (fallback when identifiers are missing)

    Returns
    -------
    str   — cleaned full text (equations stripped), ≥200 chars
    None  — full text unavailable or CORE_FULL_TEXT is disabled
    """
    try:
        from researchbuddy.config import CORE_FULL_TEXT
        if not CORE_FULL_TEXT:
            return None
    except Exception:
        pass

    if not any([doi, arxiv_id, title]):
        return None

    key = _cache_key(doi, arxiv_id, title)
    cached = _cache_load(key)
    if cached is not _CACHE_MISS:
        if cached:
            logger.debug("[CORE] Cache hit (%d chars): %s", len(cached), key[:16])
        return cached   # None means "tried before, not found"

    fulltext: Optional[str] = None

    if doi:
        fulltext = _search_core(f"doi:{doi}")
        time.sleep(_REQUEST_DELAY)

    if not fulltext and arxiv_id:
        fulltext = _search_core(f"arxivId:{arxiv_id}")
        time.sleep(_REQUEST_DELAY)

    if not fulltext and title and len(title) > 15:
        # Quote the title to narrow the search
        fulltext = _search_core(f'title:"{title[:120]}"')
        time.sleep(_REQUEST_DELAY)

    if fulltext:
        fulltext = strip_equations(fulltext)
        if len(fulltext) < 200:   # too short / mostly stripped
            fulltext = None

    _cache_save(key, fulltext)   # cache even None to avoid redundant API calls

    if fulltext:
        logger.info("[CORE] Full text fetched (%d chars): %s",
                    len(fulltext), (doi or arxiv_id or title[:40]))
    else:
        logger.debug("[CORE] No full text available: %s",
                     (doi or arxiv_id or title[:40]))

    return fulltext
