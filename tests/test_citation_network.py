"""Tests for citation_network.py — bibliographic coupling, DOI extraction."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core.citation_network import (
    _fix_ligatures,
    _looks_like_journal_header,
    _normalise_doi,
    _smart_clean_query,
    bibliographic_coupling_matrix,
    compute_edge_confidence,
    extract_doi_from_text,
)


class TestExtractDoiFromText:
    def test_standard(self):
        assert extract_doi_from_text("See 10.1234/abc.123") == "10.1234/abc.123"

    def test_no_doi(self):
        assert extract_doi_from_text("nothing here") == ""

    def test_too_short(self):
        assert extract_doi_from_text("10.1/x") == ""


class TestNormaliseDoi:
    def test_strips_url_prefix(self):
        assert _normalise_doi("https://doi.org/10.1234/Test") == "10.1234/test"

    def test_lowercases(self):
        assert _normalise_doi("10.1234/ABC") == "10.1234/abc"

    def test_empty(self):
        assert _normalise_doi("") == ""


class TestLooksLikeJournalHeader:
    def test_journal_volume(self):
        assert _looks_like_journal_header("Journal of Neuroscience91(2019)1437")

    def test_issn(self):
        assert _looks_like_journal_header("0270-6474/82")

    def test_normal_title(self):
        assert not _looks_like_journal_header("A Theory of Decision Making")

    def test_empty(self):
        assert _looks_like_journal_header("")


class TestSmartCleanQuery:
    def test_strips_journal_noise(self):
        text = "Contents lists available at ScienceDirect Some Real Content"
        result = _smart_clean_query(text)
        assert "ScienceDirect" not in result
        assert "Real Content" in result

    def test_strips_doi(self):
        text = "doi: 10.1234/test Some title here"
        result = _smart_clean_query(text)
        assert "10.1234" not in result


class TestBibliographicCouplingMatrix:
    def test_no_overlap(self):
        refs = {"p1": {"a", "b"}, "p2": {"c", "d"}}
        W = bibliographic_coupling_matrix(["p1", "p2"], refs)
        assert W[0, 1] == 0.0

    def test_full_overlap(self):
        refs = {"p1": {"a", "b"}, "p2": {"a", "b"}}
        W = bibliographic_coupling_matrix(["p1", "p2"], refs)
        assert W[0, 1] == pytest.approx(1.0)

    def test_partial_overlap(self):
        refs = {"p1": {"a", "b", "c"}, "p2": {"a", "b", "d"}}
        W = bibliographic_coupling_matrix(["p1", "p2"], refs)
        assert 0.0 < W[0, 1] < 1.0

    def test_symmetric(self):
        refs = {"p1": {"a", "b"}, "p2": {"a", "c"}}
        W = bibliographic_coupling_matrix(["p1", "p2"], refs)
        assert W[0, 1] == W[1, 0]

    def test_empty_refs(self):
        W = bibliographic_coupling_matrix(["p1", "p2"], {})
        assert W[0, 1] == 0.0


class TestComputeEdgeConfidence:
    def test_single_source(self):
        conf = compute_edge_confidence(1, 1, 0.5)
        assert conf == 0.5

    def test_two_sources(self):
        conf = compute_edge_confidence(2, 2, 0.5)
        assert conf == pytest.approx(0.65)

    def test_three_sources(self):
        conf = compute_edge_confidence(3, 3, 0.5)
        assert conf == pytest.approx(0.75)

    def test_capped_at_one(self):
        conf = compute_edge_confidence(3, 3, 0.9)
        assert conf <= 1.0
