"""
citation_network.py
Build a citation-based similarity network from Semantic Scholar reference data.

Citation graph
--------------
Directed: A → B  means  paper A cites paper B.
B is therefore a conceptual "ancestor" of A.

Bibliographic coupling (Kessler 1963)
--------------------------------------
Two papers are coupled if they cite the same references.
    coupling(A, B) = |refs(A) ∩ refs(B)| / sqrt(|refs(A)| × |refs(B)|)

This produces a symmetric similarity matrix in [0, 1].

Co-citation similarity (optional)
-----------------------------------
Two papers are co-cited if a third paper cites both of them.
    co_cite(A, B) = |citers(A) ∩ citers(B)| / sqrt(|citers(A)| × |citers(B)|)

We combine both signals: cit_sim = 0.5 * bib_coupling + 0.5 * co_citation
"""

from __future__ import annotations

import time
import numpy as np
import networkx as nx
from typing import Optional

import requests

from researchbuddy.config import S2_PAPER_URL, REQUEST_TIMEOUT, REQUEST_DELAY

_HEADERS = {"User-Agent": "ResearchBuddy/0.2 (local research assistant)"}


# ── S2 reference fetching ──────────────────────────────────────────────────────

def fetch_references(s2_id: str, limit: int = 100) -> list[str]:
    """
    Return a list of Semantic Scholar paper IDs that `s2_id` cites.
    Returns [] if the fetch fails or the paper has no references.
    """
    url = f"{S2_PAPER_URL}/{s2_id}/references"
    try:
        r = requests.get(
            url,
            params={"fields": "paperId", "limit": limit},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        refs = [
            item["citedPaper"]["paperId"]
            for item in data.get("data", [])
            if item.get("citedPaper") and item["citedPaper"].get("paperId")
        ]
        time.sleep(REQUEST_DELAY)
        return refs
    except Exception as e:
        print(f"  [citation] Could not fetch refs for {s2_id}: {e}")
        return []


def fetch_all_references(
    s2_ids: list[str],
    existing_refs: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """
    Fetch references for a list of S2 IDs.
    `existing_refs` (paper_internal_id → set of cited S2 IDs) is updated in place.
    Returns the updated dict.
    """
    refs = existing_refs or {}
    for s2_id in s2_ids:
        if s2_id not in refs:
            print(f"  [citation] Fetching references for {s2_id} …")
            refs[s2_id] = set(fetch_references(s2_id))
    return refs


# ── Similarity matrices ────────────────────────────────────────────────────────

def bibliographic_coupling_matrix(
    paper_ids: list[str],
    s2_ids: list[str],
    refs: dict[str, set[str]],
) -> np.ndarray:
    """
    Build a symmetric (n × n) bibliographic coupling matrix.
    Rows/cols correspond to paper_ids (in order).
    s2_ids[i] is the Semantic Scholar ID for paper_ids[i] (may be empty string).
    refs[s2_id] = set of cited S2 IDs.
    """
    n = len(paper_ids)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        s2_i = s2_ids[i]
        if not s2_i or s2_i not in refs:
            continue
        refs_i = refs[s2_i]
        if not refs_i:
            continue
        for j in range(i + 1, n):
            s2_j = s2_ids[j]
            if not s2_j or s2_j not in refs:
                continue
            refs_j = refs[s2_j]
            if not refs_j:
                continue
            overlap = len(refs_i & refs_j)
            if overlap > 0:
                coupling = overlap / np.sqrt(len(refs_i) * len(refs_j))
                W[i, j] = coupling
                W[j, i] = coupling
    return W


def co_citation_matrix(
    paper_ids: list[str],
    s2_ids: list[str],
    refs: dict[str, set[str]],
) -> np.ndarray:
    """
    Build a symmetric (n × n) co-citation matrix.
    co_cite(A, B) = how many other papers in our set cite both A and B.
    Normalised to [0, 1] using Salton's cosine.
    """
    n = len(paper_ids)
    # citers[s2_id] = set of s2_ids (in our set) that cite it
    citers: dict[str, set[str]] = {s2: set() for s2 in s2_ids if s2}
    for s2_i in s2_ids:
        if not s2_i or s2_i not in refs:
            continue
        for cited in refs[s2_i]:
            if cited in citers:
                citers[cited].add(s2_i)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        s2_i = s2_ids[i]
        if not s2_i:
            continue
        c_i = citers.get(s2_i, set())
        if not c_i:
            continue
        for j in range(i + 1, n):
            s2_j = s2_ids[j]
            if not s2_j:
                continue
            c_j = citers.get(s2_j, set())
            if not c_j:
                continue
            overlap = len(c_i & c_j)
            if overlap > 0:
                score = overlap / np.sqrt(len(c_i) * len(c_j))
                W[i, j] = score
                W[j, i] = score
    return W


def citation_similarity_matrix(
    paper_ids: list[str],
    s2_ids: list[str],
    refs: dict[str, set[str]],
    bib_weight: float = 0.6,
) -> np.ndarray:
    """
    Combined citation similarity: bib_weight * bib_coupling + (1-bib_weight) * co_citation.
    """
    W_bib = bibliographic_coupling_matrix(paper_ids, s2_ids, refs)
    W_coc = co_citation_matrix(paper_ids, s2_ids, refs)
    return bib_weight * W_bib + (1.0 - bib_weight) * W_coc


# ── Citation graph (directed) ──────────────────────────────────────────────────

def build_citation_graph(
    paper_ids: list[str],
    s2_ids: list[str],
    refs: dict[str, set[str]],
) -> nx.DiGraph:
    """
    Build a directed citation graph.
    Edge A → B means paper A (internal ID) cites paper B (if B is in our set).
    """
    G = nx.DiGraph()
    G.add_nodes_from(paper_ids)

    s2_to_internal = {s2: pid for pid, s2 in zip(paper_ids, s2_ids) if s2}

    for pid, s2 in zip(paper_ids, s2_ids):
        if not s2 or s2 not in refs:
            continue
        for cited_s2 in refs[s2]:
            if cited_s2 in s2_to_internal:
                G.add_edge(pid, s2_to_internal[cited_s2], weight=1.0, etype="citation")

    return G
