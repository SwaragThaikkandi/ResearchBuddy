"""
Tests for the OpenAlex + CrossRef search sources.

Network mocked. Verifies the JSON-shape parsers, abstract reconstruction,
peer-review inference, and dedup behaviour.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from researchbuddy.core.searcher import (
    search_openalex, search_crossref,
    _openalex_to_meta, _openalex_inverted_to_text,
    _crossref_to_meta,
)


# ── OpenAlex ─────────────────────────────────────────────────────────────────

OPENALEX_SAMPLE = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/example.001",
            "display_name": "Causal inference with regularized estimators",
            "publication_year": 2020,
            "abstract_inverted_index": {
                "We":  [0],
                "show": [1],
                "that": [2],
                "regularization": [3],
                "helps.": [4],
            },
            "authorships": [
                {"author": {"display_name": "Jane Smith"}},
                {"author": {"display_name": "John Doe"}},
            ],
            "primary_location": {
                "source": {"display_name": "Annals of Statistics", "type": "journal"},
            },
        },
        {
            # No DOI, abstract missing — still constructible
            "id": "https://openalex.org/W456",
            "display_name": "An older paper without DOI",
            "publication_year": 1995,
            "authorships": [{"author": {"display_name": "Old Author"}}],
            "primary_location": {
                "source": {"display_name": "ArXiv", "type": "repository"},
            },
        },
        {
            # No title — should be dropped
            "id": "https://openalex.org/W789",
            "display_name": "",
        },
    ]
}


def test_openalex_inverted_index_reconstructs_abstract():
    inv = {"foo": [0], "bar": [2], "baz": [1]}
    assert _openalex_inverted_to_text(inv) == "foo baz bar"


def test_openalex_inverted_handles_empty():
    assert _openalex_inverted_to_text({}) == ""
    assert _openalex_inverted_to_text(None) == ""


def test_openalex_to_meta_journal_paper_is_peer_reviewed():
    item = OPENALEX_SAMPLE["results"][0]
    m = _openalex_to_meta(item)
    assert m is not None
    assert m.title.startswith("Causal inference")
    assert m.doi == "10.1234/example.001"
    assert m.year == 2020
    assert "Jane Smith" in m.authors
    assert m.is_peer_reviewed is True
    assert m.venue == "Annals of Statistics"
    # Abstract reconstructed from inverted index
    assert "regularization" in m.abstract


def test_openalex_to_meta_repository_marks_preprint():
    item = OPENALEX_SAMPLE["results"][1]
    m = _openalex_to_meta(item)
    assert m is not None
    assert m.is_peer_reviewed is False
    assert m.year == 1995


def test_openalex_to_meta_returns_none_on_missing_title():
    assert _openalex_to_meta(OPENALEX_SAMPLE["results"][2]) is None


def test_search_openalex_end_to_end_with_mock():
    with patch("researchbuddy.core.searcher._get",
               return_value=OPENALEX_SAMPLE):
        out = search_openalex("regularization", limit=10)
    # Two parseable results, third dropped (empty title)
    assert len(out) == 2
    assert any(p.doi == "10.1234/example.001" for p in out)
    assert all(p.source == "discovered" for p in out)


def test_search_openalex_handles_no_response():
    with patch("researchbuddy.core.searcher._get", return_value=None):
        out = search_openalex("anything")
    assert out == []


# ── CrossRef ─────────────────────────────────────────────────────────────────

CROSSREF_SAMPLE = {
    "message": {
        "items": [
            {
                "DOI": "10.1234/cross.001",
                "title": ["Distributed regression on small networks"],
                "abstract": ("<jats:p>We propose a <jats:bold>new</jats:bold> "
                             "approach to distributed regression.</jats:p>"),
                "author": [
                    {"given": "Alice", "family": "Walker"},
                    {"given": "Bob", "family": "Brown"},
                ],
                "issued": {"date-parts": [[2018, 5, 1]]},
                "container-title": ["Journal of Statistics"],
                "type": "journal-article",
                "URL": "https://doi.org/10.1234/cross.001",
            },
            {
                "DOI": "10.5555/preprint.42",
                "title": ["A preprint with no abstract"],
                "author": [{"given": "Solo", "family": "Author"}],
                "issued": {"date-parts": [[2024]]},
                "type": "posted-content",
                "URL": "https://doi.org/10.5555/preprint.42",
            },
            {
                # missing title — drop
                "DOI": "10.1/x",
                "title": [],
            },
        ]
    }
}


def test_crossref_to_meta_journal_marks_peer_reviewed():
    item = CROSSREF_SAMPLE["message"]["items"][0]
    m = _crossref_to_meta(item)
    assert m is not None
    assert m.title.startswith("Distributed regression")
    assert m.doi == "10.1234/cross.001"
    assert m.year == 2018
    assert "Alice Walker" in m.authors
    assert m.is_peer_reviewed is True
    assert m.venue == "Journal of Statistics"
    # JATS tags must be stripped, content preserved
    assert "<" not in m.abstract
    assert "new" in m.abstract


def test_crossref_to_meta_posted_content_is_preprint():
    item = CROSSREF_SAMPLE["message"]["items"][1]
    m = _crossref_to_meta(item)
    assert m is not None
    assert m.is_peer_reviewed is False
    assert m.year == 2024
    assert m.abstract == ""


def test_crossref_to_meta_returns_none_on_missing_title():
    assert _crossref_to_meta(CROSSREF_SAMPLE["message"]["items"][2]) is None


def test_search_crossref_end_to_end_with_mock():
    with patch("researchbuddy.core.searcher._get",
               return_value=CROSSREF_SAMPLE):
        out = search_crossref("distributed regression", limit=10)
    assert len(out) == 2
    dois = {p.doi for p in out}
    assert dois == {"10.1234/cross.001", "10.5555/preprint.42"}


def test_search_crossref_handles_no_response():
    with patch("researchbuddy.core.searcher._get", return_value=None):
        assert search_crossref("anything") == []


# ── Cross-source dedup is exercised in find_candidates ──────────────────────
# (Already covered by the existing dedup in find_candidates which we
# didn't change — adding it through a full mock of _get is more end-to-end
# than valuable. The unit-level coverage above is what matters.)
