"""
Tests for the user-content ("thought") ingest path.

Lets the user feed their own writing into the graph as strongly-weighted
nodes that anchor the recommender's user-context vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from researchbuddy.core.graph_model import HierarchicalResearchGraph


# ── Markdown section parser ──────────────────────────────────────────────────

def test_markdown_parser_classifies_atx_headings():
    md = (
        "# My Thoughts\n"
        "Some intro paragraph.\n\n"
        "## Methods\n"
        "We propose a new algorithm.\n\n"
        "## Results\n"
        "It works well in practice.\n\n"
        "### 3.2 Sub-results\n"
        "More detail on results.\n"
    )
    secs = HierarchicalResearchGraph._parse_markdown_sections(md)
    # "My Thoughts" -> other (not in taxonomy); methods + results classified
    assert "methods" in secs
    assert "results" in secs
    assert "We propose" in secs["methods"]
    # Multiple sections of same type are concatenated under one key
    assert "More detail" in secs["results"]


def test_markdown_parser_handles_setext_headings():
    md = (
        "Methods\n"
        "=======\n"
        "First sentence.\n\n"
        "Results\n"
        "-------\n"
        "Findings here.\n"
    )
    secs = HierarchicalResearchGraph._parse_markdown_sections(md)
    # "Methods" as h1 heading reclassifies to methods
    assert "methods" in secs


def test_markdown_parser_returns_empty_for_no_headings():
    secs = HierarchicalResearchGraph._parse_markdown_sections(
        "Just a free-flowing paragraph with no headings."
    )
    # Whatever lands under default key "other" is filtered out
    assert "methods" not in secs
    assert "results" not in secs


# ── add_thought_from_text — core behaviour ───────────────────────────────────

def test_thought_too_short_returns_none():
    g = HierarchicalResearchGraph()
    assert g.add_thought_from_text("too short", title="x") is None


def test_thought_added_with_strong_weight_and_thought_source(monkeypatch):
    """The thought becomes a graph node with source='thought', kind set,
    and a synthetic high user_rating that anchors the context vector."""
    g = HierarchicalResearchGraph()
    # Patch embed so we don't hit the real model
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    text = (
        "I'm interested in causal identification under unobserved "
        "confounding. My current thinking is to use proxy variables "
        "combined with a regularised estimator. " * 3
    )
    meta = g.add_thought_from_text(
        text, title="My causal identification thoughts", kind="essay",
        weight=9.0,
    )
    assert meta is not None
    assert meta.source == "thought"
    assert meta.kind == "essay"
    assert meta.user_rating == 9.0
    # Anchored in the graph
    assert g.get_paper(meta.paper_id) is not None
    # paper_id is a deterministic hash so re-adding same text returns existing
    again = g.add_thought_from_text(
        text, title="My causal identification thoughts", kind="essay",
    )
    # Same node — graph still has just 1 paper
    assert again.paper_id == meta.paper_id
    assert len(g.all_papers()) == 1


def test_thought_with_explicit_section_text_map_builds_section_embeddings(monkeypatch):
    g = HierarchicalResearchGraph()
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    long_filler = "ipsum lorem dolor sit amet consectetur. " * 30
    text = (
        "# Draft outline\n"
        "## Methods\n" + long_filler + "\n"
        "## Results\n" + long_filler + "\n"
    )
    meta = g.add_thought_from_text(
        text, title="Draft", kind="draft",
        section_text_map={
            "methods": "Custom methods text. " * 30,
            "results": "Custom results text. " * 30,
        },
    )
    assert meta is not None
    # Explicit map wins over markdown parsing
    assert "methods" in meta.section_embeddings
    assert "results" in meta.section_embeddings


def test_thought_markdown_auto_section_extraction(monkeypatch):
    """If no explicit section_text_map but Markdown headings present,
    sections are auto-extracted via _parse_markdown_sections."""
    g = HierarchicalResearchGraph()
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    body = "ipsum lorem dolor sit amet. " * 30
    text = (
        "# Draft\n"
        f"## Methods\n{body}\n\n"
        f"## Results\n{body}\n"
    )
    meta = g.add_thought_from_text(text, title="Draft")
    assert meta is not None
    assert "methods" in meta.section_embeddings
    assert "results" in meta.section_embeddings


def test_thoughts_helper_returns_only_user_authored():
    g = HierarchicalResearchGraph()
    from researchbuddy.core.graph_model import PaperMeta
    real = PaperMeta(paper_id="p1", title="Real Paper", abstract="x" * 200)
    real.embedding = np.zeros(8)
    g.add_paper(real)

    fake = PaperMeta(paper_id="t1", title="My Note", abstract="y" * 200,
                     source="thought", kind="note")
    fake.embedding = np.zeros(8)
    g.add_paper(fake)

    thoughts = g.thoughts()
    assert len(thoughts) == 1
    assert thoughts[0].paper_id == "t1"


def test_thought_invalidates_section_context(monkeypatch):
    """Adding a thought must dirty the per-section context cache so
    section signals get recomputed on next score."""
    g = HierarchicalResearchGraph()
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    g._section_context_dirty = False  # pretend it's clean
    text = "ipsum lorem dolor sit amet. " * 50
    g.add_thought_from_text(text, title="My take")
    assert g._section_context_dirty is True


# ── add_thought_from_file ────────────────────────────────────────────────────

def test_thought_from_file_reads_markdown(tmp_path: Path, monkeypatch):
    g = HierarchicalResearchGraph()
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    body = "ipsum lorem dolor sit amet. " * 30
    p = tmp_path / "draft.md"
    p.write_text(
        "# My Title from Heading\n"
        f"## Methods\n{body}\n\n"
        f"## Results\n{body}\n",
        encoding="utf-8",
    )
    meta = g.add_thought_from_file(p, kind="draft")
    assert meta is not None
    # Title taken from first '#' heading
    assert meta.title == "My Title from Heading"
    assert meta.kind == "draft"
    assert "methods" in meta.section_embeddings


def test_thought_from_file_missing_returns_none(tmp_path: Path):
    g = HierarchicalResearchGraph()
    assert g.add_thought_from_file(tmp_path / "nope.md") is None


def test_thought_from_file_uses_filename_when_no_heading(tmp_path: Path, monkeypatch):
    g = HierarchicalResearchGraph()
    monkeypatch.setattr(
        "researchbuddy.core.graph_model.embed",
        lambda chunks: np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]] * len(chunks)),
    )
    p = tmp_path / "my_research_question.txt"
    p.write_text("ipsum lorem dolor sit amet. " * 30, encoding="utf-8")
    meta = g.add_thought_from_file(p, kind="question")
    assert meta is not None
    assert meta.title == "my research question"
