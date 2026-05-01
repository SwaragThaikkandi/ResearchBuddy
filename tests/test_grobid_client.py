"""
Tests for grobid_client.py — TEI-XML parsing and the HTTP availability probe.

The GROBID HTTP service itself is mocked; these tests do not require a
running GROBID instance.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock

import pytest

from researchbuddy.core import grobid_client
from researchbuddy.core.grobid_client import (
    is_available, extract,
    _parse_header, _parse_sections, _parse_figures,
    _parse_equations, _parse_references,
    _full_text_from_sections,
)


# Minimal TEI-XML fixture exercising every parser branch.
SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Causal Inference in High-Dimensional Settings</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <idno type="DOI">10.1234/example.2024.001</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We propose a novel method for causal inference in high-dimensional data.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1. Introduction</head>
        <p>Recent work has highlighted the challenges of confounding.</p>
        <p>Here we develop a regularised estimator.</p>
      </div>
      <div>
        <head>2. Methods</head>
        <p>Let X be the covariate vector.</p>
        <formula>E[Y|do(X)] = sum_z p(z) E[Y|X,Z=z]</formula>
      </div>
      <figure>
        <label>1</label>
        <head>Figure 1</head>
        <figDesc>Convergence rate against sample size.</figDesc>
      </figure>
      <figure type="table">
        <label>1</label>
        <head>Table 1</head>
        <figDesc>Comparison of methods.</figDesc>
        <table>
          <row><cell>Method</cell><cell>RMSE</cell></row>
          <row><cell>Ours</cell><cell>0.12</cell></row>
        </table>
      </figure>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <title type="main">A Pearl-style estimator</title>
              <author>
                <persName><forename>Judea</forename><surname>Pearl</surname></persName>
              </author>
              <idno type="DOI">10.5678/pearl.2023</idno>
            </analytic>
            <monogr>
              <imprint>
                <date when="2023">2023</date>
              </imprint>
            </monogr>
          </biblStruct>
          <biblStruct xml:id="b1">
            <analytic>
              <title type="main">Double machine learning</title>
              <author>
                <persName><forename>Victor</forename><surname>Chernozhukov</surname></persName>
              </author>
            </analytic>
            <monogr>
              <imprint>
                <date when="2018">2018</date>
              </imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


@pytest.fixture
def root() -> ET.Element:
    return ET.fromstring(SAMPLE_TEI)


# ── Header parsing ────────────────────────────────────────────────────────────

def test_parse_header_extracts_title_abstract_doi(root):
    title, abstract, doi = _parse_header(root)
    assert "Causal Inference" in title
    assert "novel method" in abstract
    assert doi == "10.1234/example.2024.001"


def test_parse_header_handles_missing_header():
    empty = ET.fromstring(
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text/></TEI>'
    )
    title, abstract, doi = _parse_header(empty)
    assert (title, abstract, doi) == ("", "", "")


# ── Sections ──────────────────────────────────────────────────────────────────

def test_parse_sections_returns_headed_paragraphs(root):
    secs = _parse_sections(root)
    assert len(secs) == 2
    assert secs[0].heading.startswith("1. Introduction")
    assert "regularised estimator" in secs[0].text
    assert secs[1].heading.startswith("2. Methods")
    assert "covariate" in secs[1].text


def test_full_text_collation(root):
    secs = _parse_sections(root)
    full = _full_text_from_sections(secs)
    assert "Introduction" in full
    assert "Methods" in full
    assert "covariate" in full


# ── Figures and tables ────────────────────────────────────────────────────────

def test_parse_figures_separates_figures_and_tables(root):
    figs, tabs = _parse_figures(root)
    assert len(figs) == 1
    assert len(tabs) == 1
    assert "Convergence rate" in figs[0].caption
    assert "Comparison" in tabs[0].caption
    # Table cells should be flattened
    assert "Method" in tabs[0].text and "0.12" in tabs[0].text


# ── Equations ─────────────────────────────────────────────────────────────────

def test_parse_equations(root):
    eqs = _parse_equations(root)
    assert len(eqs) == 1
    assert "do(X)" in eqs[0]


# ── References ────────────────────────────────────────────────────────────────

def test_parse_references_extracts_doi_and_authors(root):
    refs = _parse_references(root)
    assert len(refs) == 2
    assert refs[0].title == "A Pearl-style estimator"
    assert refs[0].doi == "10.5678/pearl.2023"
    assert refs[0].year == "2023"
    assert "Judea Pearl" in refs[0].authors

    # Reference without DOI still gets parsed
    assert refs[1].doi == ""
    assert refs[1].title == "Double machine learning"
    assert refs[1].year == "2018"


# ── Availability probe ────────────────────────────────────────────────────────

def test_is_available_returns_true_on_alive_response():
    with patch("researchbuddy.core.grobid_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text="true")
        assert is_available("http://localhost:8070") is True


def test_is_available_returns_false_on_404():
    with patch("researchbuddy.core.grobid_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, text="not found")
        assert is_available("http://localhost:8070") is False


def test_is_available_returns_false_on_connection_error():
    import requests
    with patch("researchbuddy.core.grobid_client.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("refused")
        assert is_available("http://localhost:8070") is False


# ── End-to-end extract() with mocked HTTP ─────────────────────────────────────

def test_extract_end_to_end_with_mocked_grobid(tmp_path: pathlib.Path):
    # Make a tiny dummy PDF (content doesn't matter; HTTP is mocked)
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("researchbuddy.core.grobid_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            content=SAMPLE_TEI.encode("utf-8"),
            text=SAMPLE_TEI,
        )
        ep = extract(pdf, base_url="http://localhost:8070")

    assert ep is not None
    assert ep.parser == "grobid"
    assert "Causal Inference" in ep.title
    assert ep.doi == "10.1234/example.2024.001"
    assert len(ep.sections) == 2
    assert len(ep.figures) == 1
    assert len(ep.tables) == 1
    assert len(ep.equations) == 1
    assert len(ep.references) == 2
    # First chunk should be the abstract; subsequent chunks the section text
    assert ep.chunks[0].startswith("We propose")
    # Reference DOIs survive the extraction
    assert any(r.doi == "10.5678/pearl.2023" for r in ep.references)


def test_extract_raises_GrobidTimeout_on_read_timeout(tmp_path: pathlib.Path):
    """Read timeouts should be raised so callers can retry with a larger budget."""
    import requests
    from researchbuddy.core.grobid_client import GrobidTimeout

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("researchbuddy.core.grobid_client.requests.post") as mock_post:
        mock_post.side_effect = requests.ReadTimeout("read timed out")
        with pytest.raises(GrobidTimeout):
            extract(pdf, base_url="http://localhost:8070", timeout=5.0)


def test_extract_returns_none_on_grobid_error(tmp_path: pathlib.Path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("researchbuddy.core.grobid_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        assert extract(pdf, base_url="http://localhost:8070") is None


def test_extract_returns_none_on_missing_file(tmp_path: pathlib.Path):
    missing = tmp_path / "nope.pdf"
    assert extract(missing, base_url="http://localhost:8070") is None


# ── Local-refs population on graph_model ──────────────────────────────────────

def test_local_refs_populates_citation_graph_without_apis():
    """
    Verify the 'holy grail' optimisation: GROBID-parsed refs populate
    self._refs directly, so fetch_citations does not call external APIs
    for those papers.
    """
    from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta

    g = HierarchicalResearchGraph()
    p1 = PaperMeta(paper_id="p1", title="Paper One", abstract="abstract1")
    p1.local_refs = [
        {"title": "Cited A", "doi": "10.1/a", "year": "2020", "authors": [], "raw": ""},
        {"title": "Cited B", "doi": "10.2/b", "year": "2021", "authors": [], "raw": ""},
    ]
    g.add_paper(p1)

    n = g._populate_refs_from_local(verbose=False)
    assert n == 1
    assert g._refs["p1"] == {"10.1/a", "10.2/b"}


def test_local_refs_falls_back_to_title_match_within_corpus():
    """When a parsed ref lacks a DOI, we fuzzy-match the title against
    other papers already in the graph and link them by paper_id."""
    from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta

    g = HierarchicalResearchGraph()
    cited = PaperMeta(paper_id="cited", title="Foundations of Causal Reasoning", abstract="")
    citing = PaperMeta(paper_id="citing", title="Applications of Causal Reasoning", abstract="")
    citing.local_refs = [
        # No DOI — only a title
        {"title": "Foundations of Causal Reasoning!", "doi": "", "year": "2020",
         "authors": [], "raw": ""},
    ]
    g.add_paper(cited)
    g.add_paper(citing)

    g._populate_refs_from_local(verbose=False)
    assert "cited" in g._refs["citing"]
