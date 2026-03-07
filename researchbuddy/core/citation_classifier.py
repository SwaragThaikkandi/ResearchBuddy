"""
citation_classifier.py — Context-aware citation relationship classifier.

Classifies how paper A relates to paper B:
  "supports"    — A uses B as supporting evidence or builds upon its findings
  "contradicts" — A challenges, revises, or disagrees with B
  "mentions"    — A references B as background/method without a strong stance

Two classification methods (tried in order):
  1. Keyword pattern matching on surrounding text (fast, high precision)
  2. Sentence embedding similarity to prototype sentences (robust to paraphrase)

Used by:
  - rebuild_hierarchy() to annotate G_citation edges in-place
  - Arguer to understand WHY papers are structurally connected
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import PaperMeta

# ── Type labels ────────────────────────────────────────────────────────────────

CIT_SUPPORT    = "supports"
CIT_CONTRADICT = "contradicts"
CIT_MENTION    = "mentions"

# ── Keyword banks ──────────────────────────────────────────────────────────────

_SUPPORT_KWS = [
    "consistent with", "in agreement with", "confirms", "confirm",
    "supports", "support", "demonstrates", "validate", "corroborate",
    "replicate", "align with", "in line with", "extends", "extend",
    "builds on", "build on", "builds upon", "following", "based on",
    "as shown by", "as demonstrated by", "provides evidence",
    "further supports", "echoes", "reinforces", "converges",
    "is consistent", "corroborating", "substantiates", "replicating",
    "corroborated by", "well established", "widely accepted",
]

_CONTRADICT_KWS = [
    "contradicts", "contrary to", "in contrast", "unlike",
    "challenges", "challenge", "disputes", "dispute",
    "inconsistent with", "questions", "whereas",
    "on the other hand", "however,", "but found",
    "but suggest", "fails to", "does not support", "do not support",
    "cannot be explained", "alternative", "differ from",
    "revises", "cast doubt", "at odds with", "incompatible",
    "not fully", "problematic for", "instead of", "rather than",
    "contends", "refutes", "rebuts", "incompatible with",
    "calls into question", "has been criticized",
]

_MENTION_KWS = [
    "see also", "e.g.,", "i.e.,", "for a review", "reviewed by",
    "proposed by", "introduced by", "developed by", "defined by",
    "according to", "as described", "as noted", "as discussed",
    "the method of", "the approach of", "the framework of",
    "first described", "originally proposed", "classically",
    "summarized by", "surveyed by", "for example",
]

# ── Prototype sentences for embedding fallback ─────────────────────────────────

_PROTO_SENTENCES = {
    CIT_SUPPORT: (
        "This finding is consistent with and supports the prior evidence, "
        "confirming and extending earlier work in agreement with predictions."
    ),
    CIT_CONTRADICT: (
        "However, this challenges and contradicts previous work. "
        "Contrary to earlier findings, this result is inconsistent with prior claims."
    ),
    CIT_MENTION: (
        "As originally proposed and introduced in prior work, "
        "this method was reviewed and defined according to established procedures."
    ),
}

_cached_protos: Optional[dict[str, np.ndarray]] = None


def _get_proto_vecs() -> dict[str, np.ndarray]:
    global _cached_protos
    if _cached_protos is None:
        try:
            from researchbuddy.core.embedder import embed
            _cached_protos = {k: embed(v) for k, v in _PROTO_SENTENCES.items()}
        except Exception:
            _cached_protos = {}
    return _cached_protos


# ── Core classifier ────────────────────────────────────────────────────────────

def classify_citation_context(context_text: str) -> tuple[str, float]:
    """
    Classify the citation relationship type from surrounding context text.

    Returns (cit_type, confidence) where:
      cit_type   ∈ {"supports", "contradicts", "mentions"}
      confidence ∈ [0.0, 1.0]
    """
    if not context_text or len(context_text.strip()) < 10:
        return CIT_MENTION, 0.30

    tl = context_text.lower()

    n_sup = sum(1 for kw in _SUPPORT_KWS    if kw in tl)
    n_con = sum(1 for kw in _CONTRADICT_KWS if kw in tl)
    n_men = sum(1 for kw in _MENTION_KWS    if kw in tl)
    total = n_sup + n_con + n_men

    if total > 0:
        best_count = max(n_sup, n_con, n_men)
        if n_sup == best_count:
            best_type = CIT_SUPPORT
        elif n_con == best_count:
            best_type = CIT_CONTRADICT
        else:
            best_type = CIT_MENTION
        if best_count >= 1:
            conf = min(0.90, 0.45 + best_count * 0.12)
            return best_type, conf

    # Embedding-based fallback
    try:
        from researchbuddy.core.embedder import embed, cosine_similarity
        protos = _get_proto_vecs()
        if protos:
            vec  = embed(context_text[:400])
            sims = {t: float(cosine_similarity(vec, pv)) for t, pv in protos.items()}
            best = max(sims, key=sims.get)
            if sims[best] > 0.35:
                return best, sims[best]
    except Exception:
        pass

    return CIT_MENTION, 0.30


def classify_from_abstracts(
    abstract_a: str,
    title_b: str,
    abstract_b: str,
) -> tuple[str, float]:
    """
    Infer citation type from the citing paper's abstract (A) and the cited
    paper's title/abstract (B).

    Strategy:
      1. Search for title_b key-terms in abstract_a; classify that context.
      2. Use semantic similarity between the two abstracts as a proxy:
           high sim → likely consensus/support
           moderate → parallel development / mention
    """
    if not abstract_a:
        return CIT_MENTION, 0.25

    # 1. Look for title_b fragments in abstract_a
    if title_b and len(title_b) > 10:
        sig_words = [
            w for w in title_b.split()
            if len(w) > 4 and w.lower() not in {
                "study", "analysis", "method", "model", "theory", "approach",
                "using", "toward", "beyond", "neural", "based", "novel",
                "improved", "unified", "general", "framework",
            }
        ]
        if sig_words:
            fragment = " ".join(sig_words[:3]).lower()
            al = abstract_a.lower()
            idx = al.find(fragment)
            if idx >= 0:
                context = abstract_a[max(0, idx - 150): idx + 200]
                ctype, conf = classify_citation_context(context)
                return ctype, conf

    # 2. Semantic similarity as proxy
    try:
        from researchbuddy.core.embedder import embed, cosine_similarity
        if abstract_a and abstract_b:
            vec_a = embed(abstract_a[:300])
            vec_b = embed(abstract_b[:300])
            sim   = float(cosine_similarity(vec_a, vec_b))
            if sim > 0.72:
                return CIT_SUPPORT, 0.50
            elif sim > 0.45:
                return CIT_MENTION, 0.45
            else:
                return CIT_MENTION, 0.35
    except Exception:
        pass

    return CIT_MENTION, 0.28


# ── Graph annotation ───────────────────────────────────────────────────────────

def annotate_citation_types(
    G_citation,
    paper_metas: dict,
) -> dict[tuple[str, str], tuple[str, float]]:
    """
    Annotate every edge in G_citation with (cit_type, confidence).

    Updates edge attributes "cit_type" and "cit_confidence" in-place, and
    returns a mapping (citing_id, cited_id) → (cit_type, confidence).

    Edge type handling:
      etype="citation"     → direct cite: classify from abstracts
      etype="bib_coupling" → co-citation: use semantic similarity as proxy
    """
    edge_types: dict[tuple[str, str], tuple[str, float]] = {}

    for u, v, data in list(G_citation.edges(data=True)):
        meta_u = paper_metas.get(u)
        meta_v = paper_metas.get(v)

        if not meta_u or not meta_v:
            ctype, conf = CIT_MENTION, 0.30
        else:
            etype = data.get("etype", "")
            if etype == "citation":
                ctype, conf = classify_from_abstracts(
                    abstract_a = meta_u.abstract or "",
                    title_b    = meta_v.title    or "",
                    abstract_b = meta_v.abstract or "",
                )
            elif etype == "bib_coupling":
                try:
                    from researchbuddy.core.embedder import cosine_similarity
                    if (meta_u.embedding is not None
                            and meta_v.embedding is not None):
                        sim = float(cosine_similarity(
                            meta_u.embedding, meta_v.embedding))
                        if sim > 0.68:
                            ctype, conf = CIT_SUPPORT, 0.52
                        else:
                            ctype, conf = CIT_MENTION, 0.45
                    else:
                        ctype, conf = CIT_MENTION, 0.35
                except Exception:
                    ctype, conf = CIT_MENTION, 0.35
            else:
                # interest or unknown edge type
                ctype, conf = classify_citation_context(
                    (meta_u.abstract or "")[-300:]
                )

        edge_types[(u, v)] = (ctype, conf)
        G_citation[u][v]["cit_type"]       = ctype
        G_citation[u][v]["cit_confidence"] = round(conf, 3)

    return edge_types
