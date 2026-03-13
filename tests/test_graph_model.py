"""Tests for graph_model.py — the central HierarchicalResearchGraph."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


class TestPaperMeta:
    def test_effective_weight_seed(self):
        meta = PaperMeta(paper_id="p1", title="Test", abstract="abs", source="seed")
        assert meta.effective_weight == 5.0

    def test_effective_weight_rated(self):
        meta = PaperMeta(paper_id="p1", title="Test", abstract="abs", source="seed")
        meta.user_rating = 8.0
        assert meta.effective_weight == 8.0

    def test_effective_weight_discovered(self):
        meta = PaperMeta(paper_id="p1", title="Test", abstract="abs", source="discovered")
        assert meta.effective_weight == 0.0


class TestHierarchicalResearchGraph:
    def test_add_paper(self, sample_papers):
        graph = HierarchicalResearchGraph()
        meta = sample_papers[0]
        assert graph.add_paper(meta, meta.embedding) is True

    def test_add_duplicate_paper(self, sample_papers):
        graph = HierarchicalResearchGraph()
        meta = sample_papers[0]
        graph.add_paper(meta, meta.embedding)
        assert graph.add_paper(meta, meta.embedding) is False

    def test_add_duplicate_title(self, sample_papers):
        graph = HierarchicalResearchGraph()
        meta = sample_papers[0]
        graph.add_paper(meta, meta.embedding)
        # Same title, different ID
        meta2 = PaperMeta(
            paper_id="different_id",
            title=meta.title,
            abstract="different",
            embedding=meta.embedding,
        )
        assert graph.add_paper(meta2) is False

    def test_all_papers(self, graph_with_papers, sample_papers):
        assert len(graph_with_papers.all_papers()) == len(sample_papers)

    def test_get_paper(self, graph_with_papers, sample_papers):
        p = graph_with_papers.get_paper(sample_papers[0].paper_id)
        assert p is not None
        assert p.title == sample_papers[0].title

    def test_get_paper_missing(self, graph_with_papers):
        assert graph_with_papers.get_paper("nonexistent") is None

    def test_rate_paper(self, graph_with_papers, sample_papers):
        pid = sample_papers[0].paper_id
        graph_with_papers.rate_paper(pid, 8.0)
        assert graph_with_papers.get_paper(pid).user_rating == 8.0

    def test_rate_paper_invalid(self, graph_with_papers):
        with pytest.raises(KeyError):
            graph_with_papers.rate_paper("nonexistent", 5.0)

    def test_context_vector(self, graph_with_papers):
        ctx = graph_with_papers.context_vector()
        assert ctx is not None
        assert ctx.shape == (384,)
        # Should be unit-normalised
        assert abs(np.linalg.norm(ctx) - 1.0) < 1e-5

    def test_score_candidate(self, graph_with_papers, sample_papers):
        # Create a new candidate
        meta = PaperMeta(
            paper_id="candidate_1",
            title="Decision Making Under Risk",
            abstract="This study investigates risk preferences.",
            embedding=sample_papers[0].embedding,  # reuse embedding
        )
        score = graph_with_papers.score_candidate(meta)
        assert 0.0 <= score <= 1.0

    def test_score_candidate_no_embedding(self, graph_with_papers):
        meta = PaperMeta(paper_id="x", title="No Emb", abstract="")
        assert graph_with_papers.score_candidate(meta) == 0.0

    def test_novelty_score(self, graph_with_papers, sample_papers):
        # Known paper should have low novelty
        score = graph_with_papers.novelty_score(sample_papers[0])
        assert 0.0 <= score <= 1.0

    def test_make_id_s2(self):
        pid = HierarchicalResearchGraph.make_id("Title", s2_id="abc123")
        assert pid == "s2_abc123"

    def test_make_id_doi(self):
        pid = HierarchicalResearchGraph.make_id("Title", doi="10.1234/test")
        assert pid.startswith("doi_")

    def test_make_id_arxiv(self):
        pid = HierarchicalResearchGraph.make_id("Title", arxiv_id="2301.00001")
        assert pid == "arxiv_2301.00001"

    def test_make_id_title(self):
        pid = HierarchicalResearchGraph.make_id("Some Title")
        assert pid.startswith("title_")

    def test_stats(self, graph_with_papers):
        stats = graph_with_papers.stats()
        assert stats["total_papers"] == 5
        assert stats["rated_papers"] == 0
        assert "hierarchy_levels" in stats
        assert "reliability_health" in stats

    def test_seed_papers(self, graph_with_papers):
        seeds = graph_with_papers.seed_papers()
        assert len(seeds) == 5

    def test_rated_papers_empty(self, graph_with_papers):
        assert len(graph_with_papers.rated_papers()) == 0

    def test_pickle_setstate(self):
        """Verify __setstate__ handles missing attributes gracefully."""
        graph = HierarchicalResearchGraph()
        state = graph.__dict__.copy()
        # Remove some newer attributes
        state.pop("_query_interactions", None)
        state.pop("_argument_interactions", None)
        state.pop("_style_profile", None)
        state.pop("G_causal", None)
        state.pop("_ref_sources", None)
        state.pop("_edge_anomalies", None)
        state.pop("_reliability_history", None)

        graph2 = HierarchicalResearchGraph.__new__(HierarchicalResearchGraph)
        graph2.__setstate__(state)
        assert hasattr(graph2, "_query_interactions")
        assert hasattr(graph2, "G_causal")
        assert hasattr(graph2, "_ref_sources")

    def test_peer_review_inference_arxiv(self):
        meta = PaperMeta(
            paper_id="arx1", title="Test", abstract="",
            arxiv_id="2301.00001", doi="",
        )
        HierarchicalResearchGraph._infer_peer_review_status(meta)
        assert meta.is_peer_reviewed is False

    def test_peer_review_inference_doi(self):
        meta = PaperMeta(
            paper_id="doi1", title="Test", abstract="",
            doi="10.1234/test", arxiv_id="",
        )
        HierarchicalResearchGraph._infer_peer_review_status(meta)
        assert meta.is_peer_reviewed is True
