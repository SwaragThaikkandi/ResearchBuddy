"""
state_manager.py
Save and load the ResearchGraph to disk (pickle).
Also provides the import-from-PDF-folder workflow.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

from researchbuddy.config import STATE_FILE, DATA_DIR
from researchbuddy.core.graph_model import HierarchicalResearchGraph, ResearchGraph, PaperMeta
from researchbuddy.core.pdf_processor import extract_from_folder, ExtractedPaper


# ── Save / Load ────────────────────────────────────────────────────────────────

def save(graph: ResearchGraph, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[state] Graph saved → {path}")


def load(path: Path = STATE_FILE) -> Optional[HierarchicalResearchGraph]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            graph = pickle.load(f)
        # Migrate old flat ResearchGraph to HierarchicalResearchGraph
        if not isinstance(graph, HierarchicalResearchGraph):
            graph = HierarchicalResearchGraph.from_legacy(graph)
        print(f"[state] Graph loaded <- {path}")
        return graph
    except Exception as e:
        print(f"[state] Could not load saved state ({e}). Starting fresh.")
        return None


# ── Import seed PDFs ───────────────────────────────────────────────────────────

def import_pdf_folder(graph: HierarchicalResearchGraph, folder: str | Path) -> int:
    """
    Extract text + embeddings from every PDF in `folder` and add them to the
    graph as seed nodes. Returns the number of newly added papers.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"[state] Folder not found: {folder}")
        return 0

    print(f"\n[state] Importing PDFs from {folder} ...")
    extracted: list[ExtractedPaper] = extract_from_folder(folder)
    if not extracted:
        return 0

    added = 0
    for ep in extracted:
        meta = PaperMeta(
            paper_id = ep.paper_id,
            title    = ep.title,
            abstract = ep.abstract,
            source   = "seed",
            filepath = ep.filepath,
            doi      = ep.doi,          # DOI extracted from PDF text
        )
        graph.embed_paper(meta, ep.chunks)

        if graph.add_paper(meta):
            added += 1
        else:
            print(f"  ~ Already in graph: {ep.title[:70]}")

    print(f"[state] {added} new seed papers added ({len(extracted)} PDFs processed).")
    if added > 0:
        print("[state] Fetching citation data via OpenAlex (DOI/title lookup) ...")
        graph.fetch_citations(verbose=True)
        print("[state] Rebuilding hierarchy ...")
        graph.rebuild_hierarchy()
    return added


# ── Resolve S2 IDs for seed papers ────────────────────────────────────────────

def resolve_seed_s2_ids(graph: ResearchGraph, verbose: bool = True) -> None:
    """
    For seed papers without a Semantic Scholar ID, try to find one.
    Improves recommendation quality; requires internet access.
    """
    from researchbuddy.core.searcher import resolve_s2_id

    unresolved = [m for m in graph.seed_papers() if not m.s2_id]
    if not unresolved:
        return

    if verbose:
        print(f"\n[state] Resolving S2 IDs for {len(unresolved)} seed papers ...")

    for meta in unresolved:
        s2_id = resolve_s2_id(meta.title)
        if s2_id:
            meta.s2_id = s2_id
            if verbose:
                print(f"  + {meta.title[:60]} -> {s2_id}")
        else:
            if verbose:
                print(f"  x {meta.title[:60]} (not found on S2)")
        time.sleep(1.0)
