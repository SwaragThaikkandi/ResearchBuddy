"""Tests for reasoner.py — query scoring, MMR, keyword matching."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core.graph_model import PaperMeta
from researchbuddy.core.reasoner import Reasoner


class TestKeywordScore:
    def test_exact_match(self):
        meta = PaperMeta(
            paper_id="p1",
            title="Drift Diffusion Models",
            abstract="drift diffusion model for perceptual choice",
        )
        score = Reasoner._keyword_score("drift diffusion", meta)
        assert score > 0.5

    def test_no_match(self):
        meta = PaperMeta(
            paper_id="p1",
            title="Quantum Computing",
            abstract="quantum gates and qubits",
        )
        score = Reasoner._keyword_score("drift diffusion", meta)
        assert score == 0.0

    def test_title_bonus(self):
        meta_title = PaperMeta(
            paper_id="p1",
            title="Neural Networks in Vision",
            abstract="This paper studies classification.",
        )
        meta_abstract = PaperMeta(
            paper_id="p2",
            title="A Computational Study",
            abstract="Neural networks in vision systems.",
        )
        score_title = Reasoner._keyword_score("neural networks vision", meta_title)
        score_abstract = Reasoner._keyword_score("neural networks vision", meta_abstract)
        assert score_title > score_abstract

    def test_stopwords_filtered(self):
        meta = PaperMeta(
            paper_id="p1",
            title="The Theory of Everything",
            abstract="",
        )
        # Query is all stopwords except "theory"
        score = Reasoner._keyword_score("the theory of", meta)
        assert score > 0.0


class TestMMRRerank:
    def test_returns_k_items(self, sample_papers):
        scored = [(m, 0.9 - i * 0.1, {}) for i, m in enumerate(sample_papers)]
        result = Reasoner._mmr_rerank(scored, k=3)
        assert len(result) == 3

    def test_preserves_top_item(self, sample_papers):
        scored = [(m, 0.9 - i * 0.1, {}) for i, m in enumerate(sample_papers)]
        result = Reasoner._mmr_rerank(scored, k=3)
        assert result[0][0].paper_id == sample_papers[0].paper_id

    def test_small_list_unchanged(self, sample_papers):
        scored = [(sample_papers[0], 0.9, {})]
        result = Reasoner._mmr_rerank(scored, k=5)
        assert len(result) == 1


class TestReasonerPageRankCache:
    def test_cache_reused(self, graph_with_papers):
        reasoner = Reasoner(top_k=5)
        # First query populates cache
        result1 = reasoner.reason("decision making", graph_with_papers)
        edge_count = reasoner._pr_edge_count

        # Second query should reuse cache (same graph)
        result2 = reasoner.reason("visual attention", graph_with_papers)
        assert reasoner._pr_edge_count == edge_count

    def test_returns_query_result(self, graph_with_papers):
        reasoner = Reasoner(top_k=5)
        result = reasoner.reason("decision making", graph_with_papers)
        assert result.query == "decision making"
        assert result.query_embedding is not None
