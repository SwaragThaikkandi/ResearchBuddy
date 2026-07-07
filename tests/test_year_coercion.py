"""
Regression for the UI crash:
    TypeError: unsupported operand type(s) for -: 'int' and 'str'
    at _extract_signals: now().year - meta.year

GROBID-parsed references carry years as strings ("2018", "n.d."); the
snowball bare-ref path fed them straight into PaperMeta and ranking died.
Coercion now lives at the PaperMeta boundary (+ pickle migration).
"""

from __future__ import annotations

import pickle

import numpy as np

from researchbuddy.core.graph_model import (
    HierarchicalResearchGraph, PaperMeta, _coerce_year,
)


def test_coerce_year_values():
    assert _coerce_year("2018") == 2018
    assert _coerce_year(2018) == 2018
    assert _coerce_year("2018-05-01") == 2018   # leading-4-digit tolerance
    assert _coerce_year("n.d.") is None
    assert _coerce_year("") is None
    assert _coerce_year(None) is None
    assert _coerce_year(99999) is None          # OCR garbage out of range
    assert _coerce_year(150) is None


def test_papermeta_construction_coerces_string_year():
    m = PaperMeta(paper_id="x", title="T", abstract="a", year="2018")
    assert m.year == 2018
    m2 = PaperMeta(paper_id="y", title="T", abstract="a", year="n.d.")
    assert m2.year is None


def test_rank_candidates_survives_string_year(graph_with_papers):
    """Exact repro of the field trace: snowball-shaped candidate with a
    string year must rank, not TypeError."""
    g = graph_with_papers
    base = g.all_papers()[0].embedding
    cand = PaperMeta(paper_id="ref_1", title="A Reference With String Year",
                     abstract="", doi="10.1/ref", source="snowball",
                     year="2018", embedding=base)
    results = g.rank_candidates([cand], n=5, exploration_ratio=0.0)
    assert isinstance(results, list)            # no crash is the assertion


def test_unpickle_migrates_poisoned_years(graph_with_papers):
    """Old pickles hold PaperMeta objects whose year is already a string
    (set before the fix; __post_init__ doesn't run on unpickle)."""
    g = graph_with_papers
    victim = g.all_papers()[0]
    victim.__dict__["year"] = "2019"            # bypass __post_init__
    g2 = pickle.loads(pickle.dumps(g))
    years = {m.paper_id: m.year for m in g2.all_papers()}
    assert years[victim.paper_id] == 2019
    assert all(y is None or isinstance(y, int) for y in years.values())
