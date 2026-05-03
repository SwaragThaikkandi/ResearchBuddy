"""
Tests for the post-rating PDF-ingest pipeline.

When the user rates a suggested paper highly and provides a PDF, the
paper is upgraded from an abstract-only embedding to a full GROBID node
with section embeddings, parsed references, etc. — same paper_id, so
the rating is preserved.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from researchbuddy.cli import _ingest_pdf_for_rated_paper
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.pdf_processor import (
    ExtractedPaper, Section, Reference, Figure, Table, CitationContext,
)


def _fake_extracted_paper(filepath: Path) -> ExtractedPaper:
    """Build a realistic ExtractedPaper as if GROBID had parsed it."""
    sections = [
        Section(heading="1. Introduction", text=("intro text " * 40),
                section_type="introduction", number="1"),
        Section(heading="2. Methods",      text=("methods text " * 40),
                section_type="methods",      number="2"),
        Section(heading="3. Results",      text=("results text " * 40),
                section_type="results",      number="3"),
    ]
    refs = [
        Reference(raw="Pearl 2009. Causality.",
                  title="Causality", doi="10.1/pearl",
                  year="2009", authors=["Judea Pearl"],
                  contexts=[CitationContext(
                      ref_index="b0", section_type="methods",
                      section_heading="2. Methods",
                      snippet="we follow [Pearl 2009] in our identification …",
                  )]),
    ]
    return ExtractedPaper(
        filepath=str(filepath),
        paper_id="test-pid",
        title="An Improved Title from GROBID",
        abstract="A longer, GROBID-extracted abstract with more detail.",
        full_text="full body text",
        chunks=["abstract chunk", "methods chunk", "results chunk"],
        doi="10.5678/grobid",
        sections=sections,
        figures=[Figure(label="Figure 1", caption="A nice figure.")],
        tables=[Table(label="Table 1", caption="Some results.",
                      rows=[["a", "b"], ["1", "2"]])],
        equations=["E = mc^2"],
        references=refs,
        parser="grobid",
    )


def test_ingest_rejects_missing_file(tmp_path):
    g = HierarchicalResearchGraph()
    meta = PaperMeta(paper_id="x", title="X", abstract="")
    g.add_paper(meta)

    ok = _ingest_pdf_for_rated_paper(g, meta, str(tmp_path / "nope.pdf"))
    assert ok is False


def test_ingest_rejects_non_pdf(tmp_path):
    g = HierarchicalResearchGraph()
    meta = PaperMeta(paper_id="x", title="X", abstract="")
    g.add_paper(meta)

    txt = tmp_path / "file.txt"
    txt.write_text("not a pdf")

    ok = _ingest_pdf_for_rated_paper(g, meta, str(txt))
    assert ok is False


def test_ingest_upgrades_paper_with_grobid_data(tmp_path):
    """The crux: rating + PDF turns an abstract-only meta into a full node."""
    g = HierarchicalResearchGraph()
    meta = PaperMeta(
        paper_id="suggested-1",
        title="Short title",
        abstract="short abstract",
        doi="",
    )
    # Pretend the user rated it 9 first (current flow installs an
    # abstract-only embedding before ingest)
    meta.embedding = np.zeros(8)
    g.add_paper(meta)
    g.rate_paper(meta.paper_id, 9.0)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_ep = _fake_extracted_paper(pdf)
    with patch("researchbuddy.cli.extract_from_pdf",
               return_value=fake_ep, create=True) as _:
        # The helper imports inside the function — patch the source module
        with patch("researchbuddy.core.pdf_processor.extract_from_pdf",
                   return_value=fake_ep):
            with patch("researchbuddy.core.graph_model.embed",
                       return_value=np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * 3)):
                ok = _ingest_pdf_for_rated_paper(g, meta, str(pdf))

    assert ok is True
    # Title and DOI got upgraded from GROBID
    assert meta.title == "An Improved Title from GROBID"
    assert meta.doi == "10.5678/grobid"
    # Section embeddings now populated
    assert {"introduction", "methods", "results"}.issubset(meta.section_embeddings.keys())
    # GROBID-parsed refs carried over
    assert len(meta.local_refs) == 1
    assert meta.local_refs[0]["doi"] == "10.1/pearl"
    # Citation contexts survived
    assert meta.local_refs[0]["contexts"][0]["section_type"] == "methods"
    # Section index built
    assert any(s["type"] == "methods" for s in meta.section_index)
    # Rating SURVIVED the upgrade — same paper_id, still in graph
    assert g.get_paper("suggested-1") is not None
    assert g.get_paper("suggested-1").user_rating == 9.0
    # Filepath recorded
    assert meta.filepath.endswith("paper.pdf")


def test_ingest_returns_false_when_extractor_returns_none(tmp_path):
    g = HierarchicalResearchGraph()
    meta = PaperMeta(paper_id="x", title="X", abstract="")
    g.add_paper(meta)

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("researchbuddy.core.pdf_processor.extract_from_pdf",
               return_value=None):
        ok = _ingest_pdf_for_rated_paper(g, meta, str(pdf))
    assert ok is False


def test_ingest_does_not_overwrite_better_existing_title(tmp_path):
    """If the existing title is already longer than GROBID's, keep it."""
    g = HierarchicalResearchGraph()
    long_title = "A " + ("very " * 20) + "long original title"
    meta = PaperMeta(paper_id="x", title=long_title, abstract="")
    meta.embedding = np.zeros(8)
    g.add_paper(meta)

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    fake = _fake_extracted_paper(pdf)
    fake.title = "Short"   # GROBID returned a shorter title

    with patch("researchbuddy.core.pdf_processor.extract_from_pdf",
               return_value=fake):
        with patch("researchbuddy.core.graph_model.embed",
                   return_value=np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * 3)):
            _ingest_pdf_for_rated_paper(g, meta, str(pdf))

    assert meta.title == long_title  # untouched
