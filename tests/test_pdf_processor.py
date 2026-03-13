"""Tests for pdf_processor.py — text cleaning, DOI extraction, title parsing."""

from __future__ import annotations

import pytest

from researchbuddy.core.pdf_processor import (
    _clean,
    _extract_abstract,
    _extract_doi,
    _fix_ligatures,
    _guess_title,
    _is_journal_line,
    _stable_id,
    _to_chunks,
)


class TestFixLigatures:
    def test_fi_ligature(self):
        assert _fix_ligatures("uni\ufb01ed") == "unified"

    def test_fl_ligature(self):
        assert _fix_ligatures("re\ufb02ection") == "reflection"

    def test_ff_ligature(self):
        assert _fix_ligatures("e\ufb00ect") == "effect"

    def test_em_dash(self):
        assert _fix_ligatures("a\u2014b") == "a-b"

    def test_smart_quotes(self):
        assert _fix_ligatures("\u201chello\u201d") == '"hello"'

    def test_no_ligatures(self):
        assert _fix_ligatures("normal text") == "normal text"


class TestClean:
    def test_collapses_whitespace(self):
        assert _clean("hello   world") == "hello world"

    def test_strips_non_ascii(self):
        result = _clean("hello\x00world")
        assert "\x00" not in result

    def test_ligatures_before_strip(self):
        # fi ligature should be preserved as "fi" not stripped
        assert "fi" in _clean("uni\ufb01ed")


class TestExtractDoi:
    def test_standard_doi(self):
        assert _extract_doi("doi: 10.1234/abc.def") == "10.1234/abc.def"

    def test_doi_in_url(self):
        result = _extract_doi("https://doi.org/10.1038/nature12373")
        assert "10.1038/nature12373" in result

    def test_no_doi(self):
        assert _extract_doi("no doi here") == ""

    def test_strips_trailing_punctuation(self):
        result = _extract_doi("(10.1234/test).")
        assert not result.endswith(")")
        assert not result.endswith(".")


class TestIsJournalLine:
    def test_url(self):
        assert _is_journal_line("Available online at www.sciencedirect.com")

    def test_doi_line(self):
        assert _is_journal_line("doi: 10.1234/test")

    def test_volume(self):
        assert _is_journal_line("Vol. 81, No. 5")

    def test_normal_title(self):
        assert not _is_journal_line(
            "A Theory of Human Decision Making Under Uncertainty"
        )

    def test_empty(self):
        assert _is_journal_line("")


class TestGuessTitle:
    def test_skips_short_lines(self):
        lines = ["short", "A Theory of Human Decision Making Under Uncertainty"]
        assert "Decision" in _guess_title(lines)

    def test_skips_journal_lines(self):
        lines = [
            "Available online at www.sciencedirect.com",
            "A Theory of Human Decision Making Under Uncertainty",
        ]
        assert "Decision" in _guess_title(lines)


class TestExtractAbstract:
    def test_finds_abstract_section(self):
        text = "Some preamble\nAbstract: This paper examines X.\nIntroduction\nBody text"
        result = _extract_abstract(text)
        assert "examines" in result

    def test_fallback_to_first_chars(self):
        text = "No abstract section here, just body text about research."
        result = _extract_abstract(text)
        assert len(result) > 0


class TestToChunks:
    def test_single_chunk(self):
        text = " ".join(["word"] * 100)
        chunks = _to_chunks(text, chunk_size=400)
        assert len(chunks) >= 1

    def test_overlap(self):
        text = " ".join(["word"] * 500)
        chunks = _to_chunks(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2


class TestStableId:
    def test_deterministic(self):
        assert _stable_id("/path/to/file.pdf") == _stable_id("/path/to/file.pdf")

    def test_different_paths(self):
        assert _stable_id("/path/a.pdf") != _stable_id("/path/b.pdf")

    def test_length(self):
        assert len(_stable_id("test")) == 12
