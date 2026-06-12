"""Tests for citation snowballing (core/snowball.py)."""

from __future__ import annotations

import pytest

from researchbuddy.core import snowball as sb
from researchbuddy.core.graph_model import PaperMeta


def _oa_work(title, doi="", year=2020, oa_id="W1"):
    return {
        "id": f"https://openalex.org/{oa_id}",
        "display_name": title,
        "doi": f"https://doi.org/{doi}" if doi else "",
        "publication_year": year,
        "abstract_inverted_index": {"This": [0], "abstract": [1]},
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "J. Test",
                                        "type": "journal"}},
    }


# ── Seed selection ─────────────────────────────────────────────────────────────

def test_pick_seeds_prefers_high_ratings(graph_with_papers):
    g = graph_with_papers
    papers = g.all_papers()
    papers[0].user_rating = 9.0
    papers[1].user_rating = 4.0     # below SNOWBALL_MIN_RATING
    papers[2].user_rating = 8.0

    seeds = sb.pick_seeds(g)
    ids = {m.paper_id for m in seeds}
    assert papers[0].paper_id in ids
    assert papers[2].paper_id in ids
    assert papers[1].paper_id not in ids
    assert seeds[0].user_rating == 9.0


def test_pick_seeds_falls_back_to_seed_pdfs(graph_with_papers):
    seeds = sb.pick_seeds(graph_with_papers)   # nothing rated
    assert seeds                                # sample papers are source="seed"
    assert all(m.source == "seed" for m in seeds)


# ── Helpers ────────────────────────────────────────────────────────────────────

def test_short_id():
    assert sb._short_id("https://openalex.org/W123") == "W123"
    assert sb._short_id("W123") == "W123"
    assert sb._short_id("") == ""


# ── Snowball round ─────────────────────────────────────────────────────────────

def test_snowball_dedupes_against_graph(graph_with_papers, monkeypatch):
    g = graph_with_papers
    seed = g.all_papers()[0]
    seed.user_rating = 9.0
    seed.doi = "10.1/seed"

    existing_title = g.all_papers()[1].title   # already in graph

    monkeypatch.setattr(sb, "REQUEST_DELAY", 0)
    monkeypatch.setattr(sb, "_work_by_doi", lambda doi: {
        "id": "https://openalex.org/W_seed",
        "referenced_works": ["https://openalex.org/W10",
                             "https://openalex.org/W11"],
    })
    monkeypatch.setattr(sb, "_works_by_ids", lambda ids: [
        _oa_work("A Brand New Reference", doi="10.1/new", oa_id="W10"),
        _oa_work(existing_title, doi="", oa_id="W11"),    # dup by title
    ])
    monkeypatch.setattr(sb, "_citing_works", lambda oid, limit: [
        _oa_work("A Citing Paper", doi="10.1/citing", oa_id="W20"),
        _oa_work("A Brand New Reference", doi="10.1/new", oa_id="W10"),  # dup in-round
    ])

    cands, stats = sb.snowball_round(g, seeds=[seed])
    titles = {m.title for m in cands}
    assert "A Brand New Reference" in titles
    assert "A Citing Paper" in titles
    assert existing_title not in titles
    assert stats["new_unique"] == 2
    assert stats["fetched"] == 4
    assert all(m.source == "snowball" for m in cands)


def test_snowball_backward_only(graph_with_papers, monkeypatch):
    g = graph_with_papers
    seed = g.all_papers()[0]
    seed.doi = "10.1/seed"

    calls = {"citing": 0}
    monkeypatch.setattr(sb, "REQUEST_DELAY", 0)
    monkeypatch.setattr(sb, "_work_by_doi", lambda doi: {
        "id": "https://openalex.org/W_seed",
        "referenced_works": ["https://openalex.org/W10"],
    })
    monkeypatch.setattr(sb, "_works_by_ids", lambda ids: [
        _oa_work("Ref Paper", doi="10.1/ref", oa_id="W10"),
    ])

    def no_citing(oid, limit):
        calls["citing"] += 1
        return []

    monkeypatch.setattr(sb, "_citing_works", no_citing)

    cands, stats = sb.snowball_round(g, seeds=[seed],
                                     directions=("backward",))
    assert calls["citing"] == 0
    assert stats["directions"] == ["backward"]
    assert len(cands) == 1


def test_snowball_uses_grobid_local_refs(graph_with_papers, monkeypatch):
    """References parsed from the user's own PDFs feed the backward path."""
    g = graph_with_papers
    seed = g.all_papers()[0]
    seed.doi = ""           # no OpenAlex resolution for the seed itself
    seed.local_refs = [
        {"title": "Local Ref With Doi", "doi": "10.9/localref",
         "year": 2018, "authors": ["B. Writer"]},
        {"title": "Local Ref Without Doi", "doi": "",
         "year": 2017, "authors": []},
    ]

    monkeypatch.setattr(sb, "REQUEST_DELAY", 0)
    monkeypatch.setattr(sb, "_work_by_doi", lambda doi: None)
    monkeypatch.setattr(sb, "_citing_works", lambda oid, limit: [])
    monkeypatch.setattr(sb, "_works_by_dois", lambda dois: [
        _oa_work("Local Ref With Doi", doi="10.9/localref", oa_id="W30"),
    ])

    cands, stats = sb.snowball_round(g, seeds=[seed])
    titles = {m.title for m in cands}
    assert "Local Ref With Doi" in titles        # enriched via OpenAlex
    assert "Local Ref Without Doi" in titles     # kept as bare metadata
    enriched = next(m for m in cands if m.title == "Local Ref With Doi")
    assert enriched.abstract                     # got an abstract from OpenAlex


def test_snowball_saturation_flag(graph_with_papers, monkeypatch):
    """When fetched is large but almost nothing is new, saturated=True."""
    g = graph_with_papers
    seed = g.all_papers()[0]
    seed.doi = "10.1/seed"
    in_graph_titles = [m.title for m in g.all_papers()]

    monkeypatch.setattr(sb, "REQUEST_DELAY", 0)
    monkeypatch.setattr(sb, "_work_by_doi", lambda doi: {
        "id": "https://openalex.org/W_seed",
        "referenced_works": [f"https://openalex.org/W{i}" for i in range(25)],
    })
    # Every "reference" is already in the graph
    monkeypatch.setattr(sb, "_works_by_ids", lambda ids: [
        _oa_work(in_graph_titles[i % len(in_graph_titles)], oa_id=f"W{i}")
        for i in range(25)
    ])
    monkeypatch.setattr(sb, "_citing_works", lambda oid, limit: [])

    cands, stats = sb.snowball_round(g, seeds=[seed])
    assert stats["new_unique"] == 0
    assert stats["fetched"] >= 20
    assert stats["saturated"] is True
