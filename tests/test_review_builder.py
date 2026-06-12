"""Tests for the Review Forge (core/review_builder.py)."""

from __future__ import annotations

import csv

import pytest

import researchbuddy.core.graph_model as gm
from researchbuddy.core import review_builder as rb
from researchbuddy.core.graph_model import PaperMeta


@pytest.fixture(autouse=True)
def _no_keybert(monkeypatch):
    """Force the word-count fallback for theme labels (no model loads)."""
    monkeypatch.setattr(gm, "_get_cached_keybert", lambda: None)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setattr(rb, "LLM_ENABLED", False)


@pytest.fixture(autouse=True)
def _tmp_audit_log(monkeypatch, tmp_path):
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "prisma.jsonl")


# ── Citation keys ──────────────────────────────────────────────────────────────

def test_bibtex_key_shape():
    m = PaperMeta(paper_id="x", title="The Drift Diffusion Model of Choice",
                  abstract="", authors=["Jane van der Smith"], year=2021)
    assert rb.bibtex_key(m) == "smith2021drift"


def test_bibtex_key_handles_missing_fields():
    m = PaperMeta(paper_id="x", title="", abstract="")
    assert rb.bibtex_key(m) == "anonnd"


def test_assign_keys_dedups_collisions():
    a = PaperMeta(paper_id="a", title="Drift Models", abstract="",
                  authors=["Kim Lee"], year=2020)
    b = PaperMeta(paper_id="b", title="Drift Analysis", abstract="",
                  authors=["Bo Lee"], year=2020)
    keys = rb.assign_keys([a, b])
    assert len(set(keys.values())) == 2
    assert sorted(k[:12] for k in keys.values()) == ["lee2020drift",
                                                     "lee2020drift"]


# ── Paper selection ────────────────────────────────────────────────────────────

def test_review_papers_filters(graph_with_papers):
    g = graph_with_papers
    papers = g.all_papers()
    papers[0].user_rating = 9.0          # included by rating
    papers[1].user_rating = 2.0          # excluded by rating
    # thought nodes never exported
    thought = PaperMeta(paper_id="t1", title="My idea", abstract="x" * 50,
                        kind="essay")
    g._papers[thought.paper_id] = thought

    selected = rb.review_papers(g)
    ids = {m.paper_id for m in selected}
    assert papers[0].paper_id in ids
    assert papers[1].paper_id not in ids
    assert "t1" not in ids
    # unrated seeds stay in
    assert papers[2].paper_id in ids


# ── Exports ────────────────────────────────────────────────────────────────────

def test_export_bibtex(graph_with_papers, tmp_path):
    g = graph_with_papers
    g.all_papers()[0].doi = "10.1/x"
    g.all_papers()[0].venue = "Journal of Testing"
    out = tmp_path / "review.bib"
    n = rb.export_bibtex(g, out)
    text = out.read_text(encoding="utf-8")
    assert n == 5
    assert text.count("@") >= 5
    assert "@article" in text            # venue present -> article
    assert "doi" in text
    assert "Journal of Testing" in text


def test_export_ris(graph_with_papers, tmp_path):
    out = tmp_path / "review.ris"
    n = rb.export_ris(graph_with_papers, out)
    text = out.read_text(encoding="utf-8")
    assert n == 5
    assert text.count("TY  - ") == 5
    assert text.count("ER  - ") == 5


def test_export_synthesis_matrix(graph_with_papers, tmp_path):
    g = graph_with_papers
    g.all_papers()[0].user_rating = 8.0
    out = tmp_path / "matrix.csv"
    n = rb.export_synthesis_matrix(g, out)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == n + 1            # header + papers
    header = rows[0]
    assert "cite_key" in header and "decision" in header
    rated_row = next(r for r in rows[1:] if r[header.index("rating")] == "8")
    assert rated_row[header.index("decision")] == "included"


def test_export_scaffold_contains_themes_and_no_abstracts(
        graph_with_papers, tmp_path):
    g = graph_with_papers
    g.all_papers()[0].user_rating = 9.0
    out = tmp_path / "scaffold.md"
    rb.export_review_scaffold(g, out, use_llm=False)
    text = out.read_text(encoding="utf-8")
    assert "# Literature Review Scaffold" in text
    assert "## Theme 1:" in text
    assert "| Cite | Year | Rating | Title | DOI |" in text
    # No third-party abstract text leaks into the export
    for m in g.all_papers():
        if m.abstract:
            assert m.abstract not in text


def test_export_prisma_markdown(tmp_path, monkeypatch):
    from researchbuddy.core import audit
    log = tmp_path / "prisma.jsonl"
    monkeypatch.setattr(audit, "PRISMA_LOG", log)
    audit.log_event("search", n_results=20, sources={"openalex": 20})
    audit.log_event("screen", paper_id="a", rating=8, decision="included")

    out = tmp_path / "prisma.md"
    rb.export_prisma_markdown(out)
    text = out.read_text(encoding="utf-8")
    assert "PRISMA 2020" in text
    assert "**20**" in text
    assert "openalex: 20" in text


def test_export_review_pack(graph_with_papers, tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "REVIEW_EXPORT_DIR", tmp_path)
    pack = rb.export_review_pack(graph_with_papers, use_llm=False)
    names = {p.name for p in pack.iterdir()}
    assert names == {"review.bib", "review.ris", "synthesis_matrix.csv",
                     "review_scaffold.md", "prisma_flow.md"}
