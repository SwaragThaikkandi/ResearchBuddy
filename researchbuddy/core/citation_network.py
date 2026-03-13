"""
citation_network.py
Build a citation-based similarity network.

Resolution strategy (tried in order for each paper):
  1. Extract DOI from title/abstract text via regex
  2. CrossRef by DOI       → cited DOIs (most reliable for published papers)
  3. CrossRef by bibliographic query (multiple cleaned text variants, fuzzy)
  4. OpenAlex by DOI       → cited OpenAlex Work IDs
  5. OpenAlex by title     → cited OpenAlex Work IDs
  6. Semantic Scholar by S2 ID  (legacy fallback)

Similarity measures
-------------------
Bibliographic coupling (Kessler 1963):
  coupling(A, B) = |refs(A) ∩ refs(B)| / sqrt(|refs(A)| × |refs(B)|)

Two papers are strongly coupled if they cite many of the same external works
— this holds even for small corpora where direct intra-corpus citations are rare.
"""

from __future__ import annotations

import re
import time
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

import logging
import requests

from researchbuddy.config import (
    S2_PAPER_URL, REQUEST_TIMEOUT, REQUEST_DELAY, OPENALEX_URL,
)

_HEADERS = {
    "User-Agent": "ResearchBuddy/0.3 (research assistant; "
                  "https://github.com/SwaragThaikkandi/ResearchBuddy)"
}

logger = logging.getLogger(__name__)

_CROSSREF_URL = "https://api.crossref.org/works"

# Minimum DOI length to avoid matching truncated DOIs
_MIN_DOI_LEN = 12


# ── Cross-validation data model ──────────────────────────────────────────────

@dataclass
class RefResult:
    """Citation references found by a single data source."""
    source: str          # "crossref_doi", "crossref_query", "openalex_doi", etc.
    ref_ids: set = field(default_factory=set)   # set of reference IDs found


# ── DOI utilities ──────────────────────────────────────────────────────────────

def extract_doi_from_text(text: str) -> str:
    """
    Find the first plausible DOI in any text string.
    Handles 'doi:10.xxx', 'https://doi.org/10.xxx', plain '10.xxx/yyy'.
    """
    m = re.search(r'10\.\d{4,9}/[^\s"\'<>\[\]{},;]+', text)
    if m:
        doi = m.group(0).strip(".,;:)(")
        if len(doi) >= _MIN_DOI_LEN:
            return doi
    return ""


# ── OCR / ligature fixes ──────────────────────────────────────────────────────

def _fix_ligatures(text: str) -> str:
    """Fix common OCR ligature artifacts that break CrossRef matching."""
    # Unicode ligature codepoints → plain ASCII
    text = text.replace('\ufb01', 'fi')
    text = text.replace('\ufb02', 'fl')
    text = text.replace('\ufb00', 'ff')
    text = text.replace('\ufb03', 'ffi')
    text = text.replace('\ufb04', 'ffl')
    # Split ligatures ("fi " → "fi") — pdfplumber sometimes inserts a space
    text = re.sub(r'\bfi\s', 'fi', text)
    text = re.sub(r'\bfl\s', 'fl', text)
    text = re.sub(r'\bff\s', 'ff', text)
    return text


# ── Inline metadata stripping (works on single-line collapsed text) ───────────

# Each pair: (compiled regex, replacement)
_INLINE_STRIP: list[tuple[re.Pattern, str]] = [
    # Journal + vol(year)pages at start: "JournalName91(2019)1437–1460"
    (re.compile(r'^[A-Za-z][A-Za-z\s,&]+\d{1,4}\s*\(\d{4}\)\s*[\d\-–]+\s*'),
     ''),
    # "Contents lists available at ScienceDirect"
    (re.compile(r'Contents\s+lists?\s+available\s+at\s+\S+', re.I), ''),
    # "journal homepage: www.xxx.com/yyy"
    (re.compile(r'journal\s+homepage\s*:?\s*\S+', re.I), ''),
    # "Available online at www..."
    (re.compile(r'Available\s+online\s+at\s+\S+', re.I), ''),
    # DOI strings
    (re.compile(r'https?://doi\.org/\S+'), ''),
    (re.compile(r'doi\s*:\s*10\.\S+', re.I), ''),
    # Full URL
    (re.compile(r'https?://\S+'), ''),
    (re.compile(r'www\.\S+'), ''),
    # ISSN like "0270-6474/82/..." or "0270~6474"
    (re.compile(r'\d{4}[-~/]\d{3}[\dxX][\S]*'), ''),
    # Price: "$02.00/O"
    (re.compile(r'\$\d+\.\d{2}\S*'), ''),
    # "Vol. 81, No. 5, 338-364"
    (re.compile(r'Vol\.\s*\d+\s*,?\s*No\.\s*\d+[^.]*', re.I), ''),
    # "pp. 338–364"
    (re.compile(r'pp\.\s*\d+\s*[-–]\s*\d+', re.I), ''),
    # Publisher names
    (re.compile(r'Elsevier\s*(B\.V\.|Science|Ltd)?|Pergamon(\s+Press)?|'
                r'Academic\s+Press|Springer[-\s]Verlag|'
                r'Cambridge\s+University\s+Press|Wiley[-\s]Liss', re.I), ''),
    # Copyright line
    (re.compile(r'©\s*\d{4}[^.]*\.?'), ''),
    # Received/Accepted dates
    (re.compile(r'Received\s+\w+\s+\d+.*?(?:Accepted|Published)\s+\w+\s+\d+[^.]*'
                r'\.?', re.I), ''),
    # "This article was downloaded by..."
    (re.compile(r'This\s+article\s+was\s+download\w*[^.]*\.?', re.I), ''),
    # "Article in press"
    (re.compile(r'Article\s+in\s+press', re.I), ''),
    # Proc. Natl. Acad. Sci. preamble
    (re.compile(r'Proc\.\s*Natl\.\s*Acad\.\s*Sci\.\s*USA\s*'
                r'Vol\.\s*\d+,?\s*pp\.\s*\d+[-–]\d+,?\s*'
                r'(?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December)?\s*\d{0,4}', re.I), ''),
    # "Psychological Review 1974, Vol. 81, ..." or "The Journal of Neuroscience, July 20, 2011 31(29)"
    (re.compile(r'^(?:The\s+)?(?:Journal\s+of\s+\w[\w\s,]+|'
                r'Psychological\s+(?:Review|Bulletin|Science)|'
                r'Neuroscience|Cogniti\w+\s+Psychology|'
                r'Annual\s+Review\s+of\s+\w+)'
                r'[\s,]*\d{4}[^A-Z]*', re.I), ''),
    # ALL-CAPS section headers: "ABSTRACT", "ARTICLES", etc.
    (re.compile(r'\b[A-Z]{4,}\b\s*'), ''),
    # Month day, year pattern: "July 20, 2011"
    (re.compile(r'(?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December)\s+\d{1,2},?\s*\d{4}',
                re.I), ''),
]


def _smart_clean_query(text: str) -> str:
    """
    Strip journal-metadata noise from abstract/title text using inline regex.
    Works on single-line collapsed text (after pdfplumber _clean()).
    """
    t = _fix_ligatures(text)
    for pat, repl in _INLINE_STRIP:
        t = pat.sub(repl, t)
    # Collapse whitespace
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t


# ── Title extraction from garbled abstract ────────────────────────────────────

def _looks_like_journal_header(text: str) -> bool:
    """Return True if text looks like a journal/publisher header, not a title."""
    t = text.strip()
    if not t:
        return True
    # "JournalName91(2019)1437..."
    if re.match(r'[A-Za-z][\w\s,]+\d{1,4}\s*\(\d{4}\)', t):
        return True
    # "The Journal of Neuroscience, July 20, 2011 31(29)..."
    if re.search(r'(?:January|February|March|April|May|June|July|August|'
                 r'September|October|November|December)\s+\d{1,2},?\s*\d{4}', t):
        return True
    # ISSN at start
    if re.match(r'\d{4}[-~/]\d{3}[\dxX]', t):
        return True
    # Price pattern
    if re.search(r'\$\d+\.\d{2}', t):
        return True
    # "Vol." or "Proc."
    if re.search(r'\bVol\.\s*\d+|\bProc\.\s*Natl', t, re.I):
        return True
    # "Psychological Review 1974, Vol..."
    if re.match(r'(?:Psychological|Cognitive|Journal|Annual)\s+\w+\s+\d{4}',
                t, re.I):
        return True
    return False


def _try_extract_title(abstract: str) -> str:
    """
    Try to find the real paper title buried inside a garbled abstract.
    After stripping metadata, the first substantial text segment is likely
    the title or the first meaningful sentence.
    """
    if not abstract or len(abstract) < 30:
        return ""
    cleaned = _smart_clean_query(abstract)
    if not cleaned or len(cleaned) < 15:
        return ""
    # Take the first sentence-like chunk (before a period followed by space)
    m = re.match(r'^(.{15,180}?)(?:\.\s|$)', cleaned)
    if m:
        return m.group(1).strip()
    return cleaned[:150].strip()


def _prepare_queries(title: str, abstract: str) -> list[str]:
    """
    Return an ordered list of query strings to try with CrossRef, best first.
    Handles garbled titles by extracting real titles from abstract text.
    """
    candidates: list[str] = []
    title_is_garbage = _looks_like_journal_header(title)

    # 1. Cleaned abstract (inline metadata stripped + ligatures fixed)
    if abstract and len(abstract) > 40:
        q = _smart_clean_query(abstract)
        if len(q) > 40:
            candidates.append(q[:250])

    # 2. Skip first 80 chars (often journal header) + clean remainder
    if abstract and len(abstract) > 120:
        mid = _smart_clean_query(abstract[80:380])
        if len(mid) > 40:
            candidates.append(mid[:250])

    # 3. Skip even more (120 chars) for deeply buried content
    if abstract and len(abstract) > 200:
        deep = _smart_clean_query(abstract[120:420])
        if len(deep) > 40:
            candidates.append(deep[:250])

    # 4. Extracted title from abstract (when stored title is garbage)
    if title_is_garbage and abstract:
        extracted = _try_extract_title(abstract)
        if extracted and len(extracted) > 15:
            candidates.append(extracted[:200])

    # 5. Cleaned title (if not garbage)
    if title and len(title) > 15 and not title_is_garbage:
        t = _smart_clean_query(title)
        if len(t) > 15:
            candidates.append(t[:200])

    # 6. Raw title (last resort — sometimes CrossRef handles noise)
    if title and len(title) > 20:
        candidates.append(title[:200])

    # Deduplicate preserving order (compare first 50 chars as key)
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        key = c[:50].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ── CrossRef (primary source — returns cited DOIs directly) ───────────────────

def _crossref_by_doi(doi: str) -> list[str]:
    """
    Fetch the reference list for a paper by its DOI via CrossRef.
    Returns a list of cited DOIs.
    """
    try:
        r = requests.get(
            f"{_CROSSREF_URL}/{doi}",
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            msg = r.json().get("message", {})
            return [ref["DOI"].lower().strip()
                    for ref in msg.get("reference", [])
                    if "DOI" in ref and ref["DOI"]]
    except Exception:
        pass
    return []


def _crossref_by_query(
    queries: list[str],
    min_score: float = 15.0,
) -> tuple[str, list[str]]:
    """
    Try each query candidate in order against CrossRef bibliographic search.
    Returns (found_doi, cited_dois) for the first successful hit.

    Uses a tiered acceptance strategy:
      - score ≥ 40 → accept with ≥1 ref  (high confidence)
      - score ≥ 15 → accept with ≥3 refs (lower score needs more evidence)
    """
    for query in queries:
        if not query or len(query) < 15:
            continue
        try:
            r = requests.get(
                _CROSSREF_URL,
                params={
                    "query.bibliographic": query[:250],
                    "rows": 1,
                    "select": "DOI,title,reference,score",
                },
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                if items:
                    item  = items[0]
                    score = item.get("score", 0)
                    refs  = [ref["DOI"].lower().strip()
                             for ref in item.get("reference", [])
                             if "DOI" in ref and ref["DOI"]]
                    # Tiered acceptance: higher confidence → fewer refs needed
                    if score >= 40 and refs:
                        doi = item.get("DOI", "").lower().strip()
                        return doi, refs
                    if score >= min_score and len(refs) >= 3:
                        doi = item.get("DOI", "").lower().strip()
                        return doi, refs
        except Exception:
            pass
        time.sleep(0.15)   # small gap between retry attempts
    return "", []


# ── OpenAlex (secondary — returns OpenAlex Work IDs) ──────────────────────────

def _normalise_oa_id(raw: str) -> str:
    """'https://openalex.org/W123' → 'W123'"""
    return raw.split("/")[-1] if raw else ""


def _openalex_by_doi(doi: str) -> list[str]:
    """Fetch referenced OpenAlex Work IDs by DOI."""
    doi_clean = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    url = f"{OPENALEX_URL}/https://doi.org/{doi_clean}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            raw = r.json().get("referenced_works", [])
            return [_normalise_oa_id(x) for x in raw if x]
    except Exception:
        pass
    return []


def _openalex_by_title(title: str) -> list[str]:
    """Search OpenAlex by title and return referenced Work IDs."""
    try:
        r = requests.get(
            OPENALEX_URL,
            params={
                "filter": f"title.search:{title[:120]}",
                "per-page": 1,
                "select": "id,doi,referenced_works",
            },
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            items = r.json().get("results", [])
            if items:
                raw = items[0].get("referenced_works", [])
                return [_normalise_oa_id(x) for x in raw if x]
    except Exception:
        pass
    return []


# ── S2 fallback ────────────────────────────────────────────────────────────────

def _s2_refs(s2_id: str, limit: int = 150) -> list[str]:
    url = f"{S2_PAPER_URL}/{s2_id}/references"
    try:
        r = requests.get(
            url,
            params={"fields": "paperId", "limit": limit},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return [
            item["citedPaper"]["paperId"]
            for item in r.json().get("data", [])
            if item.get("citedPaper") and item["citedPaper"].get("paperId")
        ]
    except Exception:
        return []


# ── Per-paper reference fetching ───────────────────────────────────────────────

def fetch_refs_for_paper(
    paper_id: str,
    doi: str = "",
    title: str = "",
    abstract: str = "",
    s2_id: str = "",
    verbose: bool = True,
) -> tuple[str, set[str]]:
    """
    Fetch the reference list for a single paper, trying every available
    identifier.  Returns (source_label, set_of_ref_ids).

    Strategy:
      1. Extract DOI from title/abstract text if doi param is empty
      2. CrossRef by DOI  (most reliable, returns cited DOIs)
      3. CrossRef bibliographic query  (multiple cleaned variants, fuzzy)
      4. OpenAlex by DOI
      5. OpenAlex by title
      6. Semantic Scholar (s2_id fallback)
    """
    combined_text = f"{title} {abstract}".strip()

    # Step 1: try to extract DOI from the paper's own text
    if not doi and combined_text:
        doi = extract_doi_from_text(combined_text)

    # Step 2: CrossRef by DOI
    if doi:
        refs = _crossref_by_doi(doi)
        if refs:
            if verbose:
                logger.info(f"  [citation] CrossRef/DOI  {len(refs):>4} refs  doi={doi[:50]}")
            return "crossref_doi", set(refs)

    # Step 3: CrossRef bibliographic query (multiple cleaned query candidates)
    queries = _prepare_queries(title, abstract)
    if queries:
        found_doi, refs = _crossref_by_query(queries)
        if refs:
            if verbose:
                logger.info(f"  [citation] CrossRef/query {len(refs):>4} refs  "
                            f"found_doi={found_doi[:40]}")
            return "crossref_query", set(refs)

    # Step 4: OpenAlex by DOI
    if doi:
        refs = _openalex_by_doi(doi)
        if refs:
            if verbose:
                logger.info(f"  [citation] OpenAlex/DOI  {len(refs):>4} refs  doi={doi[:50]}")
            return "openalex_doi", set(refs)

    # Step 5: OpenAlex by title (try cleaned/extracted title too)
    title_candidates = []
    if title and len(title) > 20:
        cleaned_t = _smart_clean_query(title)
        if cleaned_t and len(cleaned_t) > 15:
            title_candidates.append(cleaned_t)
        if title not in title_candidates:
            title_candidates.append(title)
    # Also try title extracted from abstract
    if abstract:
        extracted_t = _try_extract_title(abstract)
        if extracted_t and len(extracted_t) > 15:
            title_candidates.append(extracted_t)
    for tc in title_candidates:
        refs = _openalex_by_title(tc)
        if refs:
            if verbose:
                logger.info(f"  [citation] OpenAlex/title {len(refs):>4} refs  "
                            f"title={tc[:50]}")
            return "openalex_title", set(refs)

    # Step 6: Semantic Scholar
    if s2_id:
        refs = _s2_refs(s2_id)
        if refs:
            if verbose:
                logger.info(f"  [citation] S2           {len(refs):>4} refs  s2={s2_id}")
            return "s2", set(refs)

    if verbose:
        label = doi or title[:40] or s2_id or paper_id
        logger.debug(f"  [citation] Not found: {label[:60]}")
    return "none", set()


def fetch_refs_all_sources(
    paper_id: str,
    doi: str = "",
    title: str = "",
    abstract: str = "",
    s2_id: str = "",
    verbose: bool = True,
) -> list[RefResult]:
    """
    Try ALL available reference sources for a single paper.
    Unlike fetch_refs_for_paper(), does NOT stop at first success —
    queries every source to enable cross-validation.

    Returns a list of RefResult objects (one per source that found refs).
    """
    results: list[RefResult] = []
    combined_text = f"{title} {abstract}".strip()

    # Extract DOI from text if not provided
    if not doi and combined_text:
        doi = extract_doi_from_text(combined_text)

    # ── CrossRef by DOI ──────────────────────────────────────────────
    if doi:
        try:
            refs = _crossref_by_doi(doi)
            if refs:
                results.append(RefResult("crossref_doi", set(refs)))
                if verbose:
                    logger.debug(f"  [xval] CrossRef/DOI   {len(refs):>4} refs")
        except Exception:
            pass
        time.sleep(0.15)

    # ── CrossRef by query ────────────────────────────────────────────
    queries = _prepare_queries(title, abstract)
    if queries:
        try:
            found_doi, refs = _crossref_by_query(queries)
            if refs:
                results.append(RefResult("crossref_query", set(refs)))
                if verbose:
                    logger.debug(f"  [xval] CrossRef/query {len(refs):>4} refs")
                # If we found a DOI via query and didn't have one, use it
                if found_doi and not doi:
                    doi = found_doi
        except Exception:
            pass
        time.sleep(0.15)

    # ── OpenAlex by DOI ──────────────────────────────────────────────
    if doi:
        try:
            refs = _openalex_by_doi(doi)
            if refs:
                results.append(RefResult("openalex_doi", set(refs)))
                if verbose:
                    logger.debug(f"  [xval] OpenAlex/DOI  {len(refs):>4} refs")
        except Exception:
            pass
        time.sleep(0.15)

    # ── OpenAlex by title ────────────────────────────────────────────
    title_candidates = []
    if title and len(title) > 20:
        cleaned_t = _smart_clean_query(title)
        if cleaned_t and len(cleaned_t) > 15:
            title_candidates.append(cleaned_t)
    if abstract:
        extracted_t = _try_extract_title(abstract)
        if extracted_t and len(extracted_t) > 15:
            title_candidates.append(extracted_t)
    for tc in title_candidates[:1]:   # just try best candidate
        try:
            refs = _openalex_by_title(tc)
            if refs:
                results.append(RefResult("openalex_title", set(refs)))
                if verbose:
                    logger.debug(f"  [xval] OpenAlex/title {len(refs):>4} refs")
                break
        except Exception:
            pass
        time.sleep(0.15)

    # ── Semantic Scholar ─────────────────────────────────────────────
    if s2_id:
        try:
            refs = _s2_refs(s2_id)
            if refs:
                results.append(RefResult("s2", set(refs)))
                if verbose:
                    logger.debug(f"  [xval] S2            {len(refs):>4} refs")
        except Exception:
            pass

    return results


def compute_edge_confidence(
    n_sources_a: int,
    n_sources_b: int,
    base_weight: float,
) -> float:
    """
    Compute confidence for a bibliographic coupling edge.

    Higher confidence when multiple independent sources found refs
    for BOTH papers involved in the edge.

    Multiplier:
      min(sources) == 1: 1.0  (baseline)
      min(sources) == 2: 1.3  (two sources agree)
      min(sources) >= 3: 1.5  (three+ sources agree)
    """
    min_sources = min(n_sources_a, n_sources_b)
    if min_sources >= 3:
        multiplier = 1.5
    elif min_sources >= 2:
        multiplier = 1.3
    else:
        multiplier = 1.0
    return min(1.0, base_weight * multiplier)

def _normalise_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    if not d:
        return ""
    d = re.sub(r'^https?://(dx\.)?doi\.org/', '', d)
    return d.strip()


def _guess_ref_namespace(source: str, ref_id: str) -> str:
    src = (source or "").strip().lower()
    rid = (ref_id or "").strip().lower()
    if not rid:
        return ""
    if src.startswith("crossref"):
        return "doi"
    if src.startswith("openalex"):
        return "openalex"
    if src in {"s2", "semantic_scholar"}:
        return "s2"
    if rid.startswith("10.") or "doi.org/" in rid:
        return "doi"
    if re.match(r'^(?:https?://openalex\.org/)?w\d+$', rid):
        return "openalex"
    return "s2"


def fetch_all_refs(
    papers,
    existing: dict[str, set[str]] | None = None,
    verbose: bool = True,
    ref_sources_out: dict | None = None,
) -> dict[str, set[str]]:
    """
    Fetch references for every paper in `papers` that is not already in `existing`.
    Keyed by internal paper_id.

    When ``ref_sources_out`` dict is provided, per-paper multi-source results
    are stored there (paper_id → list[RefResult]) for cross-validation.
    """
    refs = dict(existing) if existing else {}
    need = [m for m in papers if m.paper_id not in refs]
    if not need:
        return refs

    if verbose:
        logger.info(f"[citation] Fetching references for {len(need)} papers "
                    f"(cross-validating: CrossRef + OpenAlex + S2) ...")

    for meta in need:
        all_results = fetch_refs_all_sources(
            paper_id = meta.paper_id,
            doi      = getattr(meta, "doi", "") or "",
            title    = meta.title or "",
            abstract = getattr(meta, "abstract", "") or "",
            s2_id    = getattr(meta, "s2_id", "") or "",
            verbose  = verbose,
        )
        # Merge: union of all ref_ids across sources
        merged = set()
        for rr in all_results:
            merged |= rr.ref_ids
        if merged:
            refs[meta.paper_id] = merged
        # Store per-source results for confidence scoring
        if ref_sources_out is not None and all_results:
            ref_sources_out[meta.paper_id] = all_results
        time.sleep(REQUEST_DELAY)

    n_ok = sum(1 for pid in [m.paper_id for m in need] if pid in refs)
    n_multi = (sum(1 for v in ref_sources_out.values() if len(v) >= 2)
               if ref_sources_out else 0)
    if verbose:
        logger.info(f"[citation] Got references for {n_ok}/{len(need)} papers"
                    f" ({n_multi} cross-validated by 2+ sources).")
    return refs


# ── Legacy shims (kept for old callers) ───────────────────────────────────────

def fetch_references(s2_id: str, limit: int = 100) -> list[str]:
    return _s2_refs(s2_id, limit)


def fetch_all_references(
    s2_ids: list[str],
    existing_refs: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    refs = dict(existing_refs) if existing_refs else {}
    for s2_id in s2_ids:
        if s2_id not in refs:
            refs[s2_id] = set(_s2_refs(s2_id))
            time.sleep(REQUEST_DELAY)
    return refs


# ── Similarity matrices ────────────────────────────────────────────────────────

def bibliographic_coupling_matrix(
    paper_ids: list[str],
    refs: dict[str, set[str]],
) -> np.ndarray:
    """
    Symmetric (n×n) bibliographic coupling matrix.
    refs[paper_id] = set of external reference IDs (DOIs, OA IDs, S2 IDs).
    Works regardless of ID type — only set intersections matter.
    """
    n = len(paper_ids)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        refs_i = refs.get(paper_ids[i], set())
        if not refs_i:
            continue
        for j in range(i + 1, n):
            refs_j = refs.get(paper_ids[j], set())
            if not refs_j:
                continue
            overlap = len(refs_i & refs_j)
            if overlap:
                coupling = overlap / np.sqrt(len(refs_i) * len(refs_j))
                W[i, j] = coupling
                W[j, i] = coupling
    return W


def citation_similarity_matrix(
    paper_ids: list[str],
    s2_ids: list[str],          # kept for API compat; not used internally
    refs: dict[str, set[str]],  # keyed by paper_id
    bib_weight: float = 1.0,    # only bib coupling now (co-cit needs citing data)
) -> np.ndarray:
    return bibliographic_coupling_matrix(paper_ids, refs)


# ── Citation graph (directed) ──────────────────────────────────────────────────

def build_citation_graph(
    paper_ids: list[str],
    s2_ids: list[str],
    refs: dict[str, set[str]],  # keyed by paper_id
    ref_sources: dict[str, list] | None = None,  # paper_id -> list[RefResult]
    paper_metas: dict[str, object] | None = None,
) -> nx.DiGraph:
    """
    Build a directed citation graph with confidence scores.

    Two edge families are represented:
    1. citation: A -> B when A directly cites B (preferred for causal orientation)
    2. bib_coupling: symmetric overlap signal when two papers share references

    Direct citation edges are constructed first and kept even when a
    bibliographic-coupling relation also exists between the same pair.
    """
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)

    # Pre-compute source counts for confidence scoring.
    source_counts: dict[str, int] = {}
    if ref_sources:
        for pid, results in ref_sources.items():
            source_counts[pid] = len(results or [])

    # Build internal ID maps (S2 and DOI are usually available).
    s2_to_internal: dict[str, str] = {}
    doi_to_internal: dict[str, str] = {}
    openalex_to_internal: dict[str, str] = {}

    for pid, s2 in zip(paper_ids, s2_ids):
        s2_norm = (s2 or "").strip()
        if s2_norm and s2_norm not in s2_to_internal:
            s2_to_internal[s2_norm] = pid

    if paper_metas:
        for pid in paper_ids:
            meta = paper_metas.get(pid)
            if meta is None:
                continue

            doi_norm = _normalise_doi(getattr(meta, "doi", "") or "")
            if doi_norm and doi_norm not in doi_to_internal:
                doi_to_internal[doi_norm] = pid

            s2_norm = (getattr(meta, "s2_id", "") or "").strip()
            if s2_norm and s2_norm not in s2_to_internal:
                s2_to_internal[s2_norm] = pid

            oa_raw = (getattr(meta, "openalex_id", "") or "")
            if not oa_raw:
                url = (getattr(meta, "url", "") or "")
                m = re.search(r'openalex\.org/(W\d+)', url, flags=re.I)
                if m:
                    oa_raw = m.group(1)
            oa_norm = _normalise_oa_id(oa_raw)
            if oa_norm and oa_norm not in openalex_to_internal:
                openalex_to_internal[oa_norm] = pid

    def _map_ref_to_internal(ref_id: str, source_hint: str = "") -> Optional[str]:
        rid = (ref_id or "").strip()
        if not rid:
            return None

        ns = _guess_ref_namespace(source_hint, rid)
        if ns == "doi":
            return doi_to_internal.get(_normalise_doi(rid))
        if ns == "openalex":
            return openalex_to_internal.get(_normalise_oa_id(rid))
        return s2_to_internal.get(rid)

    # Direct citations: preserve evidence source(s) per directed pair.
    direct_evidence: dict[tuple[str, str], set[str]] = {}

    for pid in paper_ids:
        used_source_rows = False

        if ref_sources and pid in ref_sources and ref_sources.get(pid):
            used_source_rows = True
            for rr in ref_sources.get(pid, []):
                if isinstance(rr, dict):
                    source_label = (rr.get("source", "") or "").strip()
                    ref_ids = rr.get("ref_ids", set()) or set()
                else:
                    source_label = (getattr(rr, "source", "") or "").strip()
                    ref_ids = getattr(rr, "ref_ids", set()) or set()

                for rid in ref_ids:
                    target = _map_ref_to_internal(str(rid), source_label)
                    if target and target != pid:
                        key = (pid, target)
                        direct_evidence.setdefault(key, set()).add(source_label or "source")

        # Fallback for older states that only have merged refs.
        if not used_source_rows:
            for rid in refs.get(pid, set()):
                rid_s = str(rid)
                target = _map_ref_to_internal(rid_s)
                if target and target != pid:
                    ns = _guess_ref_namespace("", rid_s) or "unknown"
                    key = (pid, target)
                    direct_evidence.setdefault(key, set()).add(f"merged_{ns}")

    for (src, tgt), evidence in direct_evidence.items():
        n_src = len(evidence)
        n_src_a = source_counts.get(src, 1)
        n_src_b = source_counts.get(tgt, 1)
        base_conf = compute_edge_confidence(n_src_a, n_src_b, 0.85)
        conf = min(0.99, base_conf + 0.03 * max(0, n_src - 1))

        G.add_edge(
            src,
            tgt,
            weight=1.0,
            etype="citation",
            edge_confidence=round(conf, 3),
            citation_source_count=n_src,
            citation_sources=sorted(evidence),
        )

    # Bibliographic coupling edges are secondary and do not overwrite direct citations.
    n = len(paper_ids)
    for i in range(n):
        pid_i = paper_ids[i]
        refs_i = refs.get(pid_i, set())
        if not refs_i:
            continue

        for j in range(i + 1, n):
            pid_j = paper_ids[j]
            refs_j = refs.get(pid_j, set())
            if not refs_j:
                continue

            overlap = len(refs_i & refs_j)
            if not overlap:
                continue

            w = overlap / np.sqrt(len(refs_i) * len(refs_j))
            n_src_i = source_counts.get(pid_i, 1)
            n_src_j = source_counts.get(pid_j, 1)
            conf = compute_edge_confidence(n_src_i, n_src_j, round(w, 4))

            has_direct_ij = (pid_i, pid_j) in direct_evidence
            has_direct_ji = (pid_j, pid_i) in direct_evidence

            # Attach coupling metadata to direct edges when present.
            if has_direct_ij and G.has_edge(pid_i, pid_j):
                G[pid_i][pid_j]["bib_coupling_weight"] = round(w, 4)
                G[pid_i][pid_j]["shared_refs"] = overlap
            if has_direct_ji and G.has_edge(pid_j, pid_i):
                G[pid_j][pid_i]["bib_coupling_weight"] = round(w, 4)
                G[pid_j][pid_i]["shared_refs"] = overlap

            # Do not add symmetric coupling edges when direct citation exists.
            if has_direct_ij or has_direct_ji:
                continue

            G.add_edge(
                pid_i,
                pid_j,
                weight=round(w, 4),
                etype="bib_coupling",
                shared_refs=overlap,
                edge_confidence=round(conf, 3),
            )
            G.add_edge(
                pid_j,
                pid_i,
                weight=round(w, 4),
                etype="bib_coupling",
                shared_refs=overlap,
                edge_confidence=round(conf, 3),
            )

    return G
