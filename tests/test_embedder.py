"""Tests for embedder.py — torchvision guard, cosine similarity, mean pool."""

from __future__ import annotations

import numpy as np
import pytest

from researchbuddy.core.embedder import cosine_similarity, mean_pool


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.random.randn(384).astype(float)
        v /= np.linalg.norm(v)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.zeros(384)
        a[0] = 1.0
        b = np.zeros(384)
        b[1] = 1.0
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = np.random.randn(384).astype(float)
        v /= np.linalg.norm(v)
        assert abs(cosine_similarity(v, -v) + 1.0) < 1e-6


class TestMeanPool:
    def test_single_vector(self):
        v = np.random.randn(384).astype(float)
        v /= np.linalg.norm(v)
        result = mean_pool([v])
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_uniform_weights(self):
        vecs = [np.random.randn(384).astype(float) for _ in range(3)]
        result = mean_pool(vecs)
        assert result.shape == (384,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_weighted_pool(self):
        vecs = [np.random.randn(384).astype(float) for _ in range(3)]
        weights = [1.0, 2.0, 3.0]
        result = mean_pool(vecs, weights)
        assert result.shape == (384,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_pool([])


class TestTorchvisionGuard:
    def test_guard_does_not_crash(self):
        """Importing the guard function should not raise."""
        from researchbuddy.core.embedder import _guard_torchvision
        # Just verify it's callable — don't actually run it in CI
        # as it manipulates sys.modules
        assert callable(_guard_torchvision)
