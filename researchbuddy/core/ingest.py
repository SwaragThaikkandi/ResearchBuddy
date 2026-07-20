"""
Shared PDF -> full-graph-node ingestion.

One place that upgrades a PaperMeta from "abstract embedding only" to a
fully parsed node: GROBID (pdfplumber fallback) extraction, section
embeddings, parsed local references, figure/table captions, equations.

Used by the interactive rating loop (user supplies a PDF path) and the
open-access harvester (PDF arrives from a legal OA location).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when a PDF cannot be turned into a full graph node."""


def ingest_pdf_into_meta(
    graph: HierarchicalResearchGraph,
    meta: PaperMeta,
    pdf_path: Union[str, Path],
) -> dict:
    """
    Parse `pdf_path` and fold the extracted structure into `meta`, replacing
    its abstract-only embedding with full-text + per-section embeddings.

    The paper_id stays the same, so any rating travels with the upgrade.
    Returns a small report dict; raises IngestError on failure.
    """
    from researchbuddy.core.pdf_processor import extract_from_pdf

    p = Path(pdf_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise IngestError(f"File not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise IngestError(f"Not a PDF: {p.name}")

    try:
        ep = extract_from_pdf(p)
    except Exception as e:
        raise IngestError(f"Extraction failed: {e}") from e
    if ep is None:
        raise IngestError("Extractor returned no content.")

    # Prefer GROBID-parsed values when they beat what the search API gave us.
    meta.filepath = str(p)
    if ep.doi and not getattr(meta, "doi", ""):
        meta.doi = ep.doi
    if ep.title and len(ep.title) > len(meta.title or ""):
        meta.title = ep.title
    if ep.abstract and (not meta.abstract or len(ep.abstract) > len(meta.abstract)):
        meta.abstract = ep.abstract

    if getattr(ep, "references", None):
        meta.local_refs = [
            {
                "title": r.title,
                "doi":   (r.doi or "").lower(),
                "year":  r.year,
                "authors": list(r.authors),
                "raw":   r.raw,
                "contexts": [
                    {"section_type": c.section_type,
                     "section_heading": c.section_heading,
                     "snippet": c.snippet}
                    for c in r.contexts
                ],
            }
            for r in ep.references if (r.title or r.doi)
        ]
    if getattr(ep, "sections", None):
        meta.section_index = [
            {"type": s.section_type, "heading": s.heading,
             "number": s.number, "n_words": len(s.text.split())}
            for s in ep.sections
        ]
    if getattr(ep, "figures", None):
        meta.figure_captions = [
            (f"{f.label}: {f.caption}".strip(": ").strip())
            for f in ep.figures if (f.label or f.caption)
        ]
    if getattr(ep, "tables", None):
        meta.table_captions = [
            (f"{t.label}: {t.caption}".strip(": ").strip())
            for t in ep.tables if (t.label or t.caption)
        ]
    if getattr(ep, "equations", None):
        meta.equations = list(ep.equations)

    # Re-embed with the full text + per-section embeddings.
    if ep.chunks:
        graph.embed_paper(meta, ep.chunks)
    if getattr(ep, "sections", None):
        graph.embed_paper_sections(meta, ep.sections)
    # Equation embedding (the 'equation' scoring signal) — full text only.
    graph.embed_equations(meta)

    return {
        "parser":     getattr(ep, "parser", "pdfplumber"),
        "n_sections": len(meta.section_embeddings),
        "n_refs":     len(meta.local_refs),
    }
