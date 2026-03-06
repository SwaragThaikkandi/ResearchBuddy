"""
citation_network.py
Build a citation-based similarity network using OpenAlex (primary) and
Semantic Scholar (fallback).

Resolution strategy
-------------------
For each paper, we attempt to find its reference list via:
  1. OpenAlex by DOI  (fastest, most reliable for DOI-bearing papers)
  2. OpenAlex by title search  (fallback if no DOI or DOI lookup fails)
  3. Semantic Scholar by S2 ID  (legacy fallback)

References are stored as normalised OpenAlex Work IDs ("W1234567890")
or Semantic Scholar paper IDs, whichever source succeeded.  Bibliographic
coupling and co-citation still work correctly as long as all refs for a
given paper come from the same namespace — set intersections are key.

Citation graph
--------------
Directed: A → B means paper A cites paper B (B is inside our corpus).

Bibliographic coupling (Kessler 1963)
--------------------------------------
  coupling(A, B) = |refs(A) ∩ refs(B)| / sqrt(|refs(A)| × |refs(B)|)

Co-citation similarity
-----------------------
  co_cite(A, B) = |citers(A) ∩ citers(B)| / sqrt(|citers(A)| × |citers(B)|)
"""

from __future__ import annotations

import time
import re
import numpy as np
import networkx as nx
from typing import Optional

import requests

from researchbuddy.config import (
    S2_PAPER_URL, REQUEST_TIMEOUT, REQUEST_DELAY,
    OPENALEX_URL,
)

_HEADERS = {
    "User-Agent": "ResearchBuddy/0.3 (literature search assistant; "
                  "https://github.com/SwaragThaikkandi/ResearchBuddy)"
}


# ── OpenAlex helpers ───────────────────────────────────────────────────────────

def _oa_work_by_doi(doi: str) -> Optional[dict]:
    """Fetch a single OpenAlex Work object by DOI."""
    doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    url = f"{OPENALEX_URL}/https://doi.org/{doi_clean}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _oa_work_by_title(title: str) -> Optional[dict]:
    """Fetch best OpenAlex Work match by title search."""
    try:
        r = requests.get(
            OPENALEX_URL,
            params={"filter": f"title.search:{title[:120]}", "per-page": 1,
                    "select": "id,doi,title,referenced_works"},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0]
    except Exception:
        pass
    return None


def _normalise_oa_id(raw: str) -> str:
    """'https://openalex.org/W123' → 'W123'"""
    return raw.split("/")[-1] if raw else ""


def fetch_openalex_refs(doi: str = "", title: str = "") -> tuple[str, list[str]]:
    """
    Look up a paper on OpenAlex and return its reference list.

    Returns
    -------
    (canonical_doi, [normalised_openalex_work_ids])
    canonical_doi is empty string if not found.
    """
    work = None
    if doi:
        work = _oa_work_by_doi(doi)
    if work is None and title:
        work = _oa_work_by_title(title)
    if work is None:
        return "", []

    canonical = (work.get("doi") or "").replace("https://doi.org/", "")
    raw_refs  = work.get("referenced_works", [])
    refs      = [_normalise_oa_id(r) for r in raw_refs if r]
    return canonical, refs


# ── Semantic Scholar fallback ──────────────────────────────────────────────────

def _s2_refs(s2_id: str, limit: int = 150) -> list[str]:
    """Return S2 paper IDs cited by s2_id."""
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
    s2_id: str = "",
    verbose: bool = True,
) -> tuple[str, set[str]]:
    """
    Fetch the reference list for a single paper, trying all available identifiers.

    Returns
    -------
    (source_label, set_of_ref_ids)
    source_label is one of "openalex", "s2", or "none".
    """
    # 1. OpenAlex (DOI first, then title)
    if doi or title:
        canonical, refs = fetch_openalex_refs(doi=doi, title=title)
        if refs:
            if verbose:
                tag = f"doi={doi}" if doi else f"title={title[:50]}"
                print(f"  [citation] OpenAlex  {len(refs):>4} refs  ({tag})")
            return "openalex", set(refs)

    # 2. Semantic Scholar fallback
    if s2_id:
        refs = _s2_refs(s2_id)
        if refs:
            if verbose:
                print(f"  [citation] S2        {len(refs):>4} refs  (s2={s2_id})")
            return "s2", set(refs)

    if verbose:
        label = doi or title[:40] or s2_id or paper_id
        print(f"  [citation] No refs found for: {label}")
    return "none", set()


def fetch_all_refs(
    papers,                          # list of PaperMeta
    existing: dict[str, set[str]] | None = None,
    verbose: bool = True,
) -> dict[str, set[str]]:
    """
    Fetch references for every paper in `papers` that does not already have
    an entry in `existing`.

    `existing` is keyed by `paper_id` (internal), values are sets of external
    reference IDs (OpenAlex Work IDs or S2 paper IDs).

    Returns the updated dict.
    """
    refs = dict(existing) if existing else {}
    for meta in papers:
        if meta.paper_id in refs:
            continue
        source, ref_set = fetch_refs_for_paper(
            paper_id = meta.paper_id,
            doi      = getattr(meta, "doi", "") or "",
            title    = meta.title or "",
            s2_id    = getattr(meta, "s2_id", "") or "",
            verbose  = verbose,
        )
        if ref_set:
            refs[meta.paper_id] = ref_set
        time.sleep(REQUEST_DELAY)
    return refs


# ── Legacy S2-keyed fetch (kept for backward compatibility) ───────────────────

def fetch_references(s2_id: str, limit: int = 100) -> list[str]:
    """Compatibility shim: fetch via S2, return list of S2 paper IDs."""
    return _s2_refs(s2_id, limit)


def fetch_all_references(
    s2_ids: list[str],
    existing_refs: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Compatibility shim for old callers using S2-keyed refs."""
    refs = dict(existing_refs) if existing_refs else {}
    for s2_id in s2_ids:
        if s2_id not in refs:
            print(f"  [citation] Fetching S2 refs for {s2_id} …")
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
    `refs[paper_id]` = set of external ref IDs for that paper.
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


def co_citation_matrix(
    paper_ids: list[str],
    refs: dict[str, set[str]],
) -> np.ndarray:
    """
    Symmetric (n×n) co-citation matrix.
    co_cite(A, B) = how many papers in our set cite both A and B.
    """
    n = len(paper_ids)
    # Build inverted index: ref_id → set of paper_ids (in our set) that cite it
    citers: dict[str, set[str]] = {}
    for pid in paper_ids:
        for ref in refs.get(pid, set()):
            citers.setdefault(ref, set()).add(pid)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        citing_i = {ref for ref in refs.get(paper_ids[i], set())}
        c_i = {pid for ref in citing_i for pid in citers.get(ref, set())} - {paper_ids[i]}
        if not c_i:
            continue
        for j in range(i + 1, n):
            citing_j = {ref for ref in refs.get(paper_ids[j], set())}
            c_j = {pid for ref in citing_j for pid in citers.get(ref, set())} - {paper_ids[j]}
            overlap = len(c_i & c_j)
            if overlap:
                score = overlap / np.sqrt(len(c_i) * len(c_j))
                W[i, j] = score
                W[j, i] = score
    return W


def citation_similarity_matrix(
    paper_ids: list[str],
    s2_ids: list[str],          # kept for API compat; not used internally now
    refs: dict[str, set[str]],  # keyed by paper_id
    bib_weight: float = 0.6,
) -> np.ndarray:
    """
    Combined citation similarity: bib_weight × bib_coupling + (1-bib_weight) × co_citation.
    """
    W_bib = bibliographic_coupling_matrix(paper_ids, refs)
    W_coc = co_citation_matrix(paper_ids, refs)
    return bib_weight * W_bib + (1.0 - bib_weight) * W_coc


# ── Citation graph (directed) ──────────────────────────────────────────────────

def build_citation_graph(
    paper_ids: list[str],
    s2_ids: list[str],          # kept for API compat
    refs: dict[str, set[str]],  # keyed by paper_id
) -> nx.DiGraph:
    """
    Directed citation graph: edge A → B means A cites B.
    Only edges where both A and B are in our corpus are included.

    Note: since refs now store external IDs (OpenAlex/S2), we cannot directly
    match them to internal paper_ids.  Instead we use cross-corpus bibliographic
    coupling strength as a proxy edge weight.
    """
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)

    # Direct citation edges (only possible when refs are S2 paper IDs and
    # papers also have S2 IDs — legacy path)
    s2_to_internal = {s2: pid for pid, s2 in zip(paper_ids, s2_ids) if s2}
    for pid, s2 in zip(paper_ids, s2_ids):
        if not s2:
            continue
        for cited_s2 in refs.get(pid, set()):
            if cited_s2 in s2_to_internal and s2_to_internal[cited_s2] != pid:
                G.add_edge(pid, s2_to_internal[cited_s2],
                           weight=1.0, etype="citation")

    # Bibliographic coupling edges (works for all papers regardless of ID type)
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
                coupling = overlap / np.sqrt(len(refs_i) * len(refs_j))
                G.add_edge(paper_ids[i], paper_ids[j],
                           weight=coupling, etype="bib_coupling")
                G.add_edge(paper_ids[j], paper_ids[i],
                           weight=coupling, etype="bib_coupling")

    return G
