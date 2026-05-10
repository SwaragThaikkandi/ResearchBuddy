"""
Regression tests for the ranking-quality fixes after the user reported
'It's time for my story: soap opera sources' being recommended for a
neuroscience query.

Three independent floors / filters now keep noise out:
  1. n=1 disables the explore bucket (focused result wins the only slot)
  2. relevance floor on the focused bucket (REL_FLOOR = 0.30)
  3. relevance floor on the explore bucket (EXPLORE_REL_FLOOR = 0.25)
  4. CrossRef parser filters known noise DOIs (ALA Choice etc.)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.searcher import _crossref_to_meta


# ── CrossRef noise filter ────────────────────────────────────────────────────

def test_crossref_drops_ala_choice_doi():
    """The exact noise that surfaced in production: 10.5860/choice.* DOIs
    are ALA's Choice library book reviews, not research."""
    item = {
        "DOI": "10.5860/choice.30-4228",
        "title": ["It's time for my story: soap opera sources, structure, "
                  "and response"],
        "type": "journal-article",
        "container-title": ["Choice Reviews Online"],
    }
    assert _crossref_to_meta(item) is None


def test_crossref_drops_choice_reviews_venue():
    item = {
        "DOI": "10.9999/legitimate.42",
        "title": ["A book review"],
        "type": "journal-article",
        "container-title": ["Choice Reviews"],
    }
    assert _crossref_to_meta(item) is None


def test_crossref_drops_unaccepted_types():
    """Letters, errata, editorials etc. shouldn't surface as candidates."""
    for bad_type in ("editorial", "erratum", "letter", "report", "review"):
        item = {
            "DOI": "10.1/x", "title": ["t"], "type": bad_type,
            "container-title": ["Some Journal"],
        }
        assert _crossref_to_meta(item) is None, \
            f"type={bad_type!r} should be filtered"


def test_crossref_keeps_normal_research_paper():
    item = {
        "DOI": "10.1234/cool.001",
        "title": ["A perfectly ordinary research paper"],
        "abstract": "<jats:p>We present results.</jats:p>",
        "type": "journal-article",
        "container-title": ["Annals of Statistics"],
        "issued": {"date-parts": [[2022]]},
        "author": [{"given": "Jane", "family": "Doe"}],
    }
    m = _crossref_to_meta(item)
    assert m is not None
    assert m.is_peer_reviewed is True


# ── Ranking floors and N=1 explore exclusion ────────────────────────────────

def _make_graph_with_corpus(n_seed: int = 12):
    """Tiny graph: a few seed papers around a fake topic embedded near 'topic'."""
    g = HierarchicalResearchGraph()
    # 'topic' direction so context_vector points along axis 0
    topic = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(n_seed):
        m = PaperMeta(
            paper_id=f"seed{i}", title=f"Seed {i} on the topic",
            abstract="abstract " * 20, source="seed",
        )
        # Each seed slightly perturbed near the topic axis
        v = topic + 0.05 * np.random.default_rng(i).standard_normal(8)
        v = v / np.linalg.norm(v)
        m.embedding = v
        m.user_rating = 8.0
        g.add_paper(m)
    return g, topic


def test_explore_bucket_disabled_when_n_is_one(monkeypatch):
    """The exact bug: with n=1, the explore bucket consumed the slot and
    surfaced an off-topic paper. Now n=1 returns the focused-relevance pick,
    or nothing if no candidate clears the floor."""
    g, topic = _make_graph_with_corpus()
    # Stub score_candidate so we can control rel exactly
    far_away = np.array([0, 0, 1.0, 0, 0, 0, 0, 0])  # orthogonal to topic
    cands = [
        # On-topic candidate, decent score
        PaperMeta(paper_id="hit", title="Highly relevant new paper",
                  abstract="x", source="discovered"),
        # Off-topic candidate, novel (huge novelty), trash score
        PaperMeta(paper_id="trash",
                  title="Soap opera sources, structure, and response",
                  abstract="y", source="discovered"),
    ]
    cands[0].embedding = topic.copy()
    cands[1].embedding = far_away.copy()

    score_map = {"hit": 0.7, "trash": 0.32}
    novelty_map = {"hit": 0.05, "trash": 0.95}
    monkeypatch.setattr(g, "score_candidate",
                        lambda meta: score_map[meta.paper_id])
    monkeypatch.setattr(g, "novelty_score",
                        lambda meta: novelty_map[meta.paper_id])
    monkeypatch.setattr(g, "_find_coverage_gaps", lambda: [])

    out = g.rank_candidates(cands, n=1, exploration_ratio=0.4)
    assert len(out) == 1
    meta, score, label = out[0]
    assert meta.paper_id == "hit", \
        "with n=1 the focused-relevance pick must win"
    assert label == "relevant"


def test_relevance_floor_drops_low_match_papers(monkeypatch):
    """A 0.32 match shouldn't surface in the focused bucket — REL_FLOOR=0.30
    keeps everything below 0.30 out, but 0.32 was just above so it used to
    leak through. We now use 0.30 as the floor."""
    g, topic = _make_graph_with_corpus()
    cands = [
        PaperMeta(paper_id="ok",   title="ok",   abstract="x", source="discovered"),
        PaperMeta(paper_id="weak", title="weak", abstract="y", source="discovered"),
        PaperMeta(paper_id="trash",title="t",    abstract="z", source="discovered"),
    ]
    for c in cands:
        c.embedding = topic.copy()

    monkeypatch.setattr(g, "score_candidate",
                        lambda m: {"ok": 0.7, "weak": 0.22, "trash": 0.05}[m.paper_id])
    monkeypatch.setattr(g, "novelty_score", lambda m: 0.1)
    monkeypatch.setattr(g, "_find_coverage_gaps", lambda: [])

    out = g.rank_candidates(cands, n=3, exploration_ratio=0.0)
    pids = [m.paper_id for m, _, _ in out]
    assert "ok"    in pids
    assert "weak"  not in pids   # below 0.30 floor
    assert "trash" not in pids


def test_explore_bucket_skips_low_relevance(monkeypatch):
    """Even with high novelty, an explore pick must clear EXPLORE_REL_FLOOR=0.25.
    Pure-novelty selection caused the 'soap opera' regression."""
    g, topic = _make_graph_with_corpus()
    cands = [
        PaperMeta(paper_id="goodexp", title="ok-novel", abstract="x", source="discovered"),
        PaperMeta(paper_id="garbage", title="garbage-novel", abstract="y", source="discovered"),
    ]
    for c in cands:
        c.embedding = topic.copy()

    monkeypatch.setattr(g, "score_candidate",
                        lambda m: {"goodexp": 0.4, "garbage": 0.10}[m.paper_id])
    monkeypatch.setattr(g, "novelty_score",
                        lambda m: {"goodexp": 0.6, "garbage": 0.98}[m.paper_id])
    monkeypatch.setattr(g, "_find_coverage_gaps", lambda: [])

    # n=4 so there's at least one explore slot (after relevant bucket)
    out = g.rank_candidates(cands, n=4, exploration_ratio=0.5)
    pids = [m.paper_id for m, _, _ in out]
    # garbage should be filtered out by the EXPLORE_REL_FLOOR
    assert "garbage" not in pids


# ── KeyBERT singleton (no reload spam) ───────────────────────────────────────

def test_keybert_helper_is_cached():
    """Calling _get_cached_keybert twice returns the same object — used to
    re-instantiate KeyBERT on every search and re-download all-MiniLM-L6-v2."""
    from researchbuddy.core.graph_model import _get_cached_keybert
    # Patch keybert.KeyBERT so we don't actually load a model
    with patch("keybert.KeyBERT") as mock_kb:
        sentinel = object()
        mock_kb.return_value = sentinel
        # Reset module-level cache so the test is hermetic
        import researchbuddy.core.graph_model as gm
        gm._keybert_singleton = None
        gm._keybert_load_failed = False
        a = _get_cached_keybert()
        b = _get_cached_keybert()
    assert a is b
    assert mock_kb.call_count == 1


def test_keybert_helper_handles_load_failure():
    """If KeyBERT can't be imported / loaded, return None and remember."""
    from researchbuddy.core.graph_model import _get_cached_keybert
    import researchbuddy.core.graph_model as gm
    gm._keybert_singleton = None
    gm._keybert_load_failed = False
    with patch("keybert.KeyBERT", side_effect=RuntimeError("boom")):
        a = _get_cached_keybert()
        b = _get_cached_keybert()
    assert a is None and b is None
    # And we remember so we don't re-attempt
    assert gm._keybert_load_failed is True
