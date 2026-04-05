"""
state_manager.py
Save and load the ResearchGraph to disk (pickle).
Also provides the import-from-PDF-folder workflow.
"""

from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from researchbuddy.config import STATE_FILE, DATA_DIR, HISTORY_DIR, STATE_HISTORY_KEEP
from researchbuddy.core.graph_model import HierarchicalResearchGraph, ResearchGraph, PaperMeta
from researchbuddy.core.graph_backend import create_backend, NetworkXBackend
from researchbuddy.core.pdf_processor import extract_from_folder, ExtractedPaper


# Save / Load -----------------------------------------------------------------

def save(graph: ResearchGraph, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Flush any pending writes to the backend before persisting
    if hasattr(graph, "_backend"):
        graph._backend.sync()
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    _save_history_snapshot(graph)
    logger.info("Graph saved -> %s", path)


def _save_history_snapshot(graph: ResearchGraph) -> None:
    """
    Save a timestamped snapshot for topology-evolution analysis.
    Keeps only the newest STATE_HISTORY_KEEP snapshots.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snap_path = HISTORY_DIR / f"graph_{ts}.pkl"

    try:
        with open(snap_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning("Snapshot save skipped (%s)", e)
        return

    snapshots = sorted(HISTORY_DIR.glob("graph_*.pkl"))
    if len(snapshots) <= STATE_HISTORY_KEEP:
        return

    # Prune oldest snapshots first.
    for old in snapshots[: len(snapshots) - STATE_HISTORY_KEEP]:
        try:
            old.unlink()
        except Exception:
            pass


def load(path: Path = STATE_FILE) -> Optional[HierarchicalResearchGraph]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            graph = pickle.load(f)
        # Migrate old flat ResearchGraph to HierarchicalResearchGraph
        if not isinstance(graph, HierarchicalResearchGraph):
            graph = HierarchicalResearchGraph.from_legacy(graph)
        logger.info("Graph loaded <- %s", path)
        _migrate_embeddings_if_needed(graph)

        # If Neo4j is configured, migrate the in-memory graph to Neo4j backend
        backend = create_backend()
        if not isinstance(backend, NetworkXBackend):
            _migrate_to_backend(graph, backend)

        return graph
    except Exception as e:
        logger.warning("Could not load saved state (%s). Starting fresh.", e)
        return None


def _migrate_to_backend(graph: HierarchicalResearchGraph, backend) -> None:
    """Migrate graph data from NetworkX backend to a new backend (e.g. Neo4j)."""
    from researchbuddy.core.graph_backend import (
        LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL,
    )

    # Check if Neo4j already has data
    if backend.node_count(LAYER_SEMANTIC) > 0:
        logger.info("Neo4j backend already has data, using existing graph.")
        graph._backend = backend
        return

    logger.info("Migrating graph to Neo4j backend ...")
    old_backend = graph._backend

    # Transfer each layer
    for layer, attr_name in [
        (LAYER_SEMANTIC, "semantic"),
        (LAYER_CITATION, "citation"),
        (LAYER_COMBINED, "combined"),
        (LAYER_CAUSAL, "causal"),
    ]:
        G = old_backend.to_networkx(layer)
        if G.number_of_nodes() > 0:
            backend.set_from_networkx(layer, G)

    graph._backend = backend
    backend.sync()
    logger.info("Migration to Neo4j complete.")


def _migrate_embeddings_if_needed(graph: HierarchicalResearchGraph) -> None:
    """
    Detect an embedding dimension mismatch (model changed in config) and
    re-embed all papers with the current model when one is found.
    """
    from researchbuddy.config import EMBEDDING_DIM
    papers_with_emb = [
        m for m in graph._papers.values() if m.embedding is not None
    ]
    if not papers_with_emb:
        return
    actual_dim = papers_with_emb[0].embedding.shape[0]
    if actual_dim != EMBEDDING_DIM:
        logger.warning(
            "Embedding dim mismatch: stored=%d, model expects=%d. "
            "Re-embedding all papers — this runs once and may take a minute.",
            actual_dim, EMBEDDING_DIM,
        )
        graph.reembed_all_papers()


# Import seed PDFs ------------------------------------------------------------

def import_pdf_folder(graph: HierarchicalResearchGraph, folder: str | Path) -> int:
    """
    Extract text + embeddings from every PDF in `folder` and add them to the
    graph as seed nodes. Returns the number of newly added papers.
    """
    folder = Path(folder)
    if not folder.exists():
        logger.warning("Folder not found: %s", folder)
        return 0

    logger.info("Importing PDFs from %s ...", folder)
    extracted: list[ExtractedPaper] = extract_from_folder(folder)
    if not extracted:
        return 0

    added = 0
    for ep in extracted:
        meta = PaperMeta(
            paper_id=ep.paper_id,
            title=ep.title,
            abstract=ep.abstract,
            source="seed",
            filepath=ep.filepath,
            doi=ep.doi,  # DOI extracted from PDF text
        )
        graph.embed_paper(meta, ep.chunks)

        if graph.add_paper(meta):
            added += 1
        else:
            logger.info("  ~ Already in graph: %s", ep.title[:70])

    logger.info("%d new seed papers added (%d PDFs processed).", added, len(extracted))
    if added > 0:
        logger.info("Fetching citation data via OpenAlex (DOI/title lookup) ...")
        graph.fetch_citations(verbose=True)
        logger.info("Enriching with full text via CORE ...")
        graph.enrich_with_full_text(verbose=True)
        logger.info("Rebuilding hierarchy ...")
        graph.rebuild_hierarchy()
    return added


# Resolve S2 IDs for seed papers ---------------------------------------------

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
        logger.info("Resolving S2 IDs for %d seed papers ...", len(unresolved))

    for meta in unresolved:
        s2_id = resolve_s2_id(meta.title)
        if s2_id:
            meta.s2_id = s2_id
            if verbose:
                logger.info("  + %s -> %s", meta.title[:60], s2_id)
        else:
            if verbose:
                logger.info("  x %s (not found on S2)", meta.title[:60])
        time.sleep(1.0)
