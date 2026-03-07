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
from typing import Optional

import requests

from researchbuddy.config import (
    S2_PAPER_URL, REQUEST_TIMEOUT, REQUEST_DELAY, OPENALEX_URL,
)

_HEADERS = {
    "User-Agent": "ResearchBuddy/0.3 (research assistant; "
                  "https://github.com/SwaragThaikkandi/ResearchBuddy)"
}

_CROSSREF_URL = "https://api.crossref.org/works"

# Minimum DOI length to avoid matching truncated DOIs
_MIN_DOI_LEN = 12


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
                print(f"  [citation] CrossRef/DOI  {len(refs):>4} refs  doi={doi[:50]}")
            return "crossref_doi", set(refs)

    # Step 3: CrossRef bibliographic query (multiple cleaned query candidates)
    queries = _prepare_queries(title, abstract)
    if queries:
        found_doi, refs = _crossref_by_query(queries)
        if refs:
            if verbose:
                print(f"  [citation] CrossRef/query {len(refs):>4} refs  "
                      f"found_doi={found_doi[:40]}")
            return "crossref_query", set(refs)

    # Step 4: OpenAlex by DOI
    if doi:
        refs = _openalex_by_doi(doi)
        if refs:
            if verbose:
                print(f"  [citation] OpenAlex/DOI  {len(refs):>4} refs  doi={doi[:50]}")
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
                print(f"  [citation] OpenAlex/title {len(refs):>4} refs  "
                      f"title={tc[:50]}")
            return "openalex_title", set(refs)

    # Step 6: Semantic Scholar
    if s2_id:
        refs = _s2_refs(s2_id)
        if refs:
            if verbose:
                print(f"  [citation] S2           {len(refs):>4} refs  s2={s2_id}")
            return "s2", set(refs)

    if verbose:
        label = doi or title[:40] or s2_id or paper_id
        print(f"  [citation] Not found: {label[:60]}")
    return "none", set()


def fetch_all_refs(
    papers,
    existing: dict[str, set[str]] | None = None,
    verbose: bool = True,
) -> dict[str, set[str]]:
    """
    Fetch references for every paper in `papers` that is not already in `existing`.
    Keyed by internal paper_id.
    """
    refs = dict(existing) if existing else {}
    need = [m for m in papers if m.paper_id not in refs]
    if not need:
        return refs

    if verbose:
        print(f"[citation] Fetching references for {len(need)} papers "
              f"(CrossRef → OpenAlex → S2) ...")

    for meta in need:
        source, ref_set = fetch_refs_for_paper(
            paper_id = meta.paper_id,
            doi      = getattr(meta, "doi", "") or "",
            title    = meta.title or "",
            abstract = getattr(meta, "abstract", "") or "",
            s2_id    = getattr(meta, "s2_id", "") or "",
            verbose  = verbose,
        )
        if ref_set:
            refs[meta.paper_id] = ref_set
        time.sleep(REQUEST_DELAY)

    n_ok = sum(1 for pid in [m.paper_id for m in need] if pid in refs)
    if verbose:
        print(f"[citation] Got references for {n_ok}/{len(need)} papers.")
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
) -> nx.DiGraph:
    """
    Build a directed citation graph.

    Two types of edges:
    1. Bibliographic coupling (bib_coupling): undirected strength between any
       two papers that share external references. Always populated when refs exist.
    2. Direct citation (citation): A → B when A cites B directly (only when
       both papers are in the corpus AND identified by the same ID namespace).
    """
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)

    # ── Bibliographic coupling edges ──────────────────────────────────────────
    n = len(paper_ids)
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
                w = overlap / np.sqrt(len(refs_i) * len(refs_j))
                G.add_edge(paper_ids[i], paper_ids[j],
                           weight=round(w, 4), etype="bib_coupling",
                           shared_refs=overlap)
                G.add_edge(paper_ids[j], paper_ids[i],
                           weight=round(w, 4), etype="bib_coupling",
                           shared_refs=overlap)

    # ── Direct citation edges (S2 namespace only) ─────────────────────────────
    s2_to_internal = {s2: pid for pid, s2 in zip(paper_ids, s2_ids) if s2}
    for pid, s2 in zip(paper_ids, s2_ids):
        if not s2:
            continue
        for cited_s2 in refs.get(pid, set()):
            target = s2_to_internal.get(cited_s2)
            if target and target != pid:
                # Only add if not already a stronger bib_coupling edge
                if not G.has_edge(pid, target):
                    G.add_edge(pid, target, weight=1.0, etype="citation")

    return G
