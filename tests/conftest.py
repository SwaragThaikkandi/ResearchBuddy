"""
Shared fixtures for ResearchBuddy tests.

All tests run on CPU with mocked embeddings — no GPU or model download
required.  This makes the suite safe for CI environments.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Mock the embedding model before any researchbuddy import ──────────────────

_DIM = 384  # same dim as all-MiniLM-L6-v2


def _fake_embed(texts):
    """Deterministic fake embedder: hash-based, normalised to unit length."""
    import hashlib

    single = isinstance(texts, str)
    if single:
        texts = [texts]

    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        raw = np.frombuffer(h * (_DIM // len(h) + 1), dtype=np.uint8)[:_DIM].astype(float)
        raw = raw / (np.linalg.norm(raw) or 1.0)
        vecs.append(raw)

    arr = np.stack(vecs)
    return arr[0] if single else arr


@pytest.fixture(autouse=True)
def _mock_embedder(monkeypatch):
    """Replace the real embedding model with a deterministic fake."""
    import researchbuddy.core.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "embed", _fake_embed)
    # Also patch _get_model so it never tries to load sentence-transformers
    monkeypatch.setattr(emb_mod, "_model", True)
    monkeypatch.setattr(emb_mod, "_device", "cpu")

    # Patch embed in modules that import it directly (from ... import embed)
    for mod_path in [
        "researchbuddy.core.reasoner",
        "researchbuddy.core.graph_model",
        "researchbuddy.core.searcher",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["embed"])
            if hasattr(mod, "embed"):
                monkeypatch.setattr(mod, "embed", _fake_embed)
        except ImportError:
            pass


@pytest.fixture()
def sample_papers():
    """Return a list of PaperMeta objects for testing."""
    from researchbuddy.core.graph_model import PaperMeta

    papers = []
    titles = [
        "A Theory of Human Decision Making Under Uncertainty",
        "Neural Correlates of Visual Attention in Primates",
        "Bayesian Models of Cognitive Development",
        "Drift Diffusion Models in Perceptual Choice",
        "Reinforcement Learning in the Human Brain",
    ]
    for i, title in enumerate(titles):
        meta = PaperMeta(
            paper_id=f"test_{i:03d}",
            title=title,
            abstract=f"This paper examines {title.lower()}. We present results.",
            authors=[f"Author_{i}"],
            year=2015 + i,
            source="seed",
            embedding=_fake_embed(title),
        )
        papers.append(meta)
    return papers


@pytest.fixture()
def graph_with_papers(sample_papers):
    """Return a HierarchicalResearchGraph populated with sample papers."""
    from researchbuddy.core.graph_model import HierarchicalResearchGraph

    graph = HierarchicalResearchGraph(alpha=0.6)
    for meta in sample_papers:
        graph.add_paper(meta, meta.embedding)
    return graph
