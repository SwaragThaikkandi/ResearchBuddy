"""
state_manager.py
Save and load the ResearchGraph to disk (pickle).
Also provides the import-from-PDF-folder workflow.
"""

from __future__ import annotations

import json
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
    """
    Persist the graph to disk.

    Includes a defensive guard: if the snapshot we're about to write has
    *fewer* edges than the existing on-disk pickle (and the on-disk pickle
    isn't empty), refuse to overwrite. This prevents silent data loss when
    a buggy backend migration leaves the in-memory graph in a regressed
    state (the exact bug that wiped a user's 13k-edge graph). The fresh
    snapshot is still saved to history/ so nothing is lost — only the
    canonical pickle is protected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(graph, "_backend"):
        graph._backend.sync()

    new_edges = _total_edges(graph)
    old_edges = _on_disk_edge_count(path)

    if old_edges > 0 and new_edges < old_edges:
        logger.warning(
            "Refusing to overwrite %s: in-memory graph has %d edges but "
            "the existing pickle has %d. This usually means a backend "
            "migration regressed the graph state. Saving the new state to "
            "history/ instead — the canonical pickle is unchanged. Use "
            "menu option 13 -> 8 to restore from a healthy snapshot.",
            path.name, new_edges, old_edges,
        )
        _save_history_snapshot(graph)
        return

    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    _save_history_snapshot(graph)
    logger.info("Graph saved -> %s (%d edges)", path, new_edges)


def _total_edges(graph) -> int:
    """Sum of edges across all 4 graph layers, or 0 on failure."""
    try:
        from researchbuddy.core.graph_backend import (
            LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL,
        )
        b = graph._backend
        return sum(b.edge_count(L) for L in
                   (LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL))
    except Exception:
        return 0


def _on_disk_edge_count(path: Path) -> int:
    """Peek at the existing pickle to count its edges. Returns 0 on any error."""
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as f:
            existing = pickle.load(f)
        return _total_edges(existing)
    except Exception:
        return 0


_EVOLUTION_LOG_NAME = "evolution.jsonl"


def _append_evolution_log(graph: ResearchGraph, timestamp_iso: str,
                          snapshot_file: str = "") -> None:
    """
    Append one line of summary metrics to history/evolution.jsonl.
    ~1 KB per save vs ~50-200 MB per pickle, so we can keep this forever.
    """
    try:
        from researchbuddy.evolution import compute_metrics_from_graph
    except Exception as e:
        logger.debug("Evolution metric computation unavailable: %s", e)
        return
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    log_path = HISTORY_DIR / _EVOLUTION_LOG_NAME
    try:
        metrics = compute_metrics_from_graph(graph, timestamp_iso, snapshot_file)
        d = metrics.__dict__.copy()
        # Clean NaN -> None so the JSONL is parseable everywhere
        import math
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
    except Exception as e:
        logger.debug("Evolution log append failed: %s", e)


def _save_history_snapshot(graph: ResearchGraph) -> None:
    """
    Append summary metrics to evolution.jsonl, plus keep the newest
    STATE_HISTORY_KEEP full pickles for emergency recovery.

    The pickle history used to be 200 entries (gigabytes!). It's now 3 by
    default — enough to roll back a bad save, but the long-term evolution
    record lives in the much smaller JSONL log.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ts_iso = datetime.now().isoformat(timespec="seconds")
    snap_path = HISTORY_DIR / f"graph_{ts_filename}.pkl"

    # 1. Always append the lightweight summary log
    _append_evolution_log(graph, ts_iso, snap_path.name)

    # 2. Optionally write a full pickle for recovery (skip if user set
    #    RESEARCHBUDDY_HISTORY_KEEP=0).
    if STATE_HISTORY_KEEP <= 0:
        return
    try:
        with open(snap_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning("Snapshot save skipped (%s)", e)
        return

    # 3. Prune surplus pickles
    snapshots = sorted(HISTORY_DIR.glob("graph_*.pkl"))
    if len(snapshots) <= STATE_HISTORY_KEEP:
        return
    for old in snapshots[: len(snapshots) - STATE_HISTORY_KEEP]:
        try:
            old.unlink()
        except Exception:
            pass


def compact_history() -> dict:
    """
    Walk every existing pickle snapshot, fold its metrics into the JSONL
    log, and delete the pickle. Returns a small report dict.

    Idempotent: snapshots already represented in the log (by `snapshot_file`
    field) are skipped. Safe to run multiple times.
    """
    from researchbuddy.evolution import _compute_metrics, _load_metrics_jsonl
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    log_path = HISTORY_DIR / _EVOLUTION_LOG_NAME

    existing = {r.snapshot_file for r in _load_metrics_jsonl(log_path)}
    pickles = sorted(HISTORY_DIR.glob("graph_*.pkl"))
    bytes_freed = 0
    n_ingested = 0
    n_deleted = 0

    for pkl in pickles:
        if pkl.name not in existing:
            try:
                metrics = _compute_metrics(pkl)
                with open(log_path, "a", encoding="utf-8") as f:
                    d = metrics.__dict__.copy()
                    import math
                    for k, v in list(d.items()):
                        if isinstance(v, float) and math.isnan(v):
                            d[k] = None
                    f.write(json.dumps(d) + "\n")
                n_ingested += 1
            except Exception as e:
                logger.warning("Could not ingest %s: %s", pkl.name, e)
                continue   # don't delete what we couldn't read

    # Now apply the retention policy: keep only the newest STATE_HISTORY_KEEP
    pickles = sorted(HISTORY_DIR.glob("graph_*.pkl"))
    keep_n = max(0, STATE_HISTORY_KEEP)
    delete_set = pickles[:-keep_n] if keep_n > 0 else pickles
    for old in delete_set:
        try:
            sz = old.stat().st_size
            old.unlink()
            bytes_freed += sz
            n_deleted += 1
        except Exception:
            pass

    return {
        "ingested": n_ingested,
        "deleted":  n_deleted,
        "bytes_freed": bytes_freed,
        "log_path": str(log_path),
        "kept_pickles": min(len(pickles) - n_deleted, keep_n),
    }


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
    """
    Migrate graph data from the in-memory NetworkX backend to `backend`
    (typically Neo4j).

    The local pickle is the source of truth for graph structure. Neo4j is
    a derived materialisation — so we re-migrate whenever the pickle has
    more edges than Neo4j currently does. This covers:

      * Fresh Neo4j (empty)              → migrate
      * Previous migration was partial   → re-migrate (e.g. APOC bug left
                                            nodes but no edges)
      * Both sides match                  → skip
      * Both empty                        → skip
    """
    from researchbuddy.core.graph_backend import (
        LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL,
    )

    layers = (LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL)
    old_backend = graph._backend

    # Compare per-layer edge counts: pickle (source) vs new backend (target).
    pickle_edges = {L: old_backend.to_networkx(L).number_of_edges() for L in layers}
    backend_edges = {L: backend.edge_count(L) for L in layers}

    pickle_total = sum(pickle_edges.values())
    backend_total = sum(backend_edges.values())

    if pickle_total == 0 and backend_total == 0:
        # Nothing to do either way — both sides empty
        graph._backend = backend
        return

    if backend_total >= pickle_total and backend_total > 0:
        # Backend already has the data (or more — e.g. external edits).
        logger.info(
            "Neo4j backend already has %d edges (pickle: %d) - using existing graph.",
            backend_total, pickle_total,
        )
        graph._backend = backend
        return

    if backend_total > 0:
        logger.info(
            "Neo4j backend has %d edges but pickle has %d - re-migrating "
            "to bring Neo4j up to date.",
            backend_total, pickle_total,
        )
    else:
        logger.info(
            "Migrating graph to Neo4j backend (%d total edges) ...",
            pickle_total,
        )

    # Transfer each layer (set_from_networkx clears the layer first)
    for layer in layers:
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
        # Carry over GROBID-derived structured fields when present.
        # These let fetch_citations() build the citation network from the
        # PDFs themselves before any external API call, and let the
        # reasoner answer section-targeted queries.
        if getattr(ep, "references", None):
            meta.local_refs = [
                {
                    "title": r.title,
                    "doi":   (r.doi or "").lower(),
                    "year":  r.year,
                    "authors": list(r.authors),
                    "raw":   r.raw,
                    # Each context: {section_type, section_heading, snippet}
                    "contexts": [
                        {
                            "section_type":    c.section_type,
                            "section_heading": c.section_heading,
                            "snippet":         c.snippet,
                        }
                        for c in r.contexts
                    ],
                }
                for r in ep.references if (r.title or r.doi)
            ]
        if getattr(ep, "sections", None):
            # Compact: only the type + heading (full text is already in chunks).
            # Useful for downstream queries like "summarise the methods".
            meta.section_index = [
                {
                    "type":    s.section_type,
                    "heading": s.heading,
                    "number":  s.number,
                    "n_words": len(s.text.split()),
                }
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

        graph.embed_paper(meta, ep.chunks)
        graph.embed_equations(meta)
        # When GROBID parsed the paper into typed sections, also build the
        # per-section embeddings so the section-similarity layer can compare
        # this paper's methods/results/etc. to other papers' equivalents.
        if getattr(ep, "sections", None):
            graph.embed_paper_sections(meta, ep.sections)

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
