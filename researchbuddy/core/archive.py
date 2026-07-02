"""
Open archival format — the anti-lock-in escape hatch.

ResearchBuddy's working state is a Python pickle: fast, but readable only by
this tool, this language, and (practically) this Python version. If the
knowledge topology a researcher builds over years can only be opened by one
program, that program has become the very monopoly this project exists to
prevent. This module removes that risk:

    archive/
      manifest.json     format version, counts, sha256 of every file
      papers.jsonl      one JSON object per paper — every metadata field
      edges.jsonl       one JSON object per edge, all layers, with attributes
      embeddings.npz    paper + per-section embeddings (open NumPy format)
      state.json        scalar state: alpha, learned weights, ...

Everything is plain JSON, CSV-like JSONL, and the documented NPZ container —
readable from any language, greppable, diffable, and hostable anywhere.
`import_archive` rebuilds a full working graph from these files alone, so the
pickle is a cache, not a prison. Integrity is verifiable offline via the
manifest hashes.

Note: an archive is a PERSONAL backup — it contains abstracts and file paths.
For sharing topology publicly, export a privacy-scrubbed capsule instead
(menu 19); capsules drop private data by construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta

logger = logging.getLogger(__name__)

ARCHIVE_FORMAT_VERSION = 1

# PaperMeta fields serialised to JSON (arrays go to embeddings.npz instead).
_SKIP_FIELDS = {"embedding", "section_embeddings"}


class ArchiveError(ValueError):
    """Archive missing, corrupt, tampered, or from an unsupported version."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Export ────────────────────────────────────────────────────────────────────

def export_archive(graph: HierarchicalResearchGraph,
                   out_dir: Path) -> Path:
    """Write the complete graph as an open, self-describing archive."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    papers = graph.all_papers()
    meta_fields = [f.name for f in dataclasses.fields(PaperMeta)
                   if f.name not in _SKIP_FIELDS]

    # papers.jsonl — every metadata field, one line per paper
    with open(out / "papers.jsonl", "w", encoding="utf-8") as f:
        for m in papers:
            rec = {k: getattr(m, k) for k in meta_fields}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # embeddings.npz — paper embeddings + per-section embeddings
    emb_ids, emb_rows = [], []
    sec_ids, sec_types, sec_rows = [], [], []
    for m in papers:
        if m.embedding is not None:
            emb_ids.append(m.paper_id)
            emb_rows.append(np.asarray(m.embedding, dtype=np.float32))
        for stype, vec in (m.section_embeddings or {}).items():
            sec_ids.append(m.paper_id)
            sec_types.append(stype)
            sec_rows.append(np.asarray(vec, dtype=np.float32))
    # dtype=str (numpy 'U') keeps the NPZ pickle-free => readable from any
    # language with an NPZ parser, and loadable with allow_pickle=False.
    np.savez_compressed(
        out / "embeddings.npz",
        emb_ids=np.array(emb_ids, dtype=str),
        embeddings=(np.vstack(emb_rows) if emb_rows
                    else np.zeros((0, 0), dtype=np.float32)),
        sec_ids=np.array(sec_ids, dtype=str),
        sec_types=np.array(sec_types, dtype=str),
        sec_embeddings=(np.vstack(sec_rows) if sec_rows
                        else np.zeros((0, 0), dtype=np.float32)),
    )

    # edges.jsonl — all layers with their attributes
    layers = {
        "semantic":    graph.G_semantic,
        "citation":    graph.G_citation,
        "causal":      graph.G_causal,
        "section_sim": graph.G_section_sim,
    }
    n_edges = 0
    with open(out / "edges.jsonl", "w", encoding="utf-8") as f:
        for layer, G in layers.items():
            try:
                for u, v, data in G.edges(data=True):
                    rec = {"layer": layer, "u": u, "v": v,
                           "attrs": {k: val for k, val in data.items()}}
                    f.write(json.dumps(rec, ensure_ascii=False,
                                       default=str) + "\n")
                    n_edges += 1
            except Exception as e:
                logger.debug("edge export skipped for %s: %s", layer, e)

    # state.json — scalar/global state worth preserving
    lw = getattr(graph, "_learned_signal_weights", None)
    state = {
        "alpha": graph.alpha,
        "learned_signal_weights": (list(map(float, lw))
                                   if lw is not None else None),
    }
    (out / "state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8")

    # manifest.json — last, hashing everything above
    files = ["papers.jsonl", "edges.jsonl", "embeddings.npz", "state.json"]
    manifest = {
        "format": "researchbuddy-archive",
        "format_version": ARCHIVE_FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_papers": len(papers),
        "n_edges": n_edges,
        "files": {name: _sha256_file(out / name) for name in files},
        "note": ("Open formats: JSONL + NPZ. Hierarchy/clusters are "
                 "recomputable (rebuild_hierarchy). Personal backup — "
                 "contains abstracts/paths; publish capsules, not archives."),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_archive(archive_dir: Path) -> dict:
    """Check manifest + file hashes. Returns the manifest; raises ArchiveError."""
    d = Path(archive_dir)
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise ArchiveError(f"no manifest.json in {d}")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ArchiveError(f"manifest unreadable: {e}") from e
    if manifest.get("format") != "researchbuddy-archive":
        raise ArchiveError("not a researchbuddy archive")
    if manifest.get("format_version", 0) > ARCHIVE_FORMAT_VERSION:
        raise ArchiveError("archive from a newer format — upgrade ResearchBuddy")
    for name, expect in (manifest.get("files") or {}).items():
        p = d / name
        if not p.exists():
            raise ArchiveError(f"missing file: {name}")
        got = _sha256_file(p)
        if got != expect:
            raise ArchiveError(
                f"hash mismatch for {name} (corrupted or tampered)")
    return manifest


# ── Import ────────────────────────────────────────────────────────────────────

def import_archive(archive_dir: Path,
                   rebuild: bool = True) -> HierarchicalResearchGraph:
    """Rebuild a full working graph from an open archive (verifies first)."""
    d = Path(archive_dir)
    verify_archive(d)

    valid_fields = {f.name for f in dataclasses.fields(PaperMeta)}

    # embeddings (pickle-free NPZ — see export)
    arr = np.load(d / "embeddings.npz")
    emb_by_id = {str(pid): arr["embeddings"][i]
                 for i, pid in enumerate(arr["emb_ids"])}
    sec_by_id: dict[str, dict[str, np.ndarray]] = {}
    for i, pid in enumerate(arr["sec_ids"]):
        sec_by_id.setdefault(str(pid), {})[str(arr["sec_types"][i])] = \
            arr["sec_embeddings"][i]

    graph = HierarchicalResearchGraph()

    # papers (unknown fields from future versions are ignored, not fatal)
    with open(d / "papers.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kwargs = {k: v for k, v in rec.items() if k in valid_fields}
            meta = PaperMeta(**kwargs)
            meta.section_embeddings = sec_by_id.get(meta.paper_id, {})
            graph.add_paper(meta, emb_by_id.get(meta.paper_id))

    # edges — restore each layer through the backend write-through setters
    import networkx as nx
    layer_graphs = {name: nx.DiGraph() for name in
                    ("semantic", "citation", "causal", "section_sim")}
    known = set(graph._papers.keys())
    with open(d / "edges.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            G = layer_graphs.get(rec.get("layer"))
            u, v = rec.get("u"), rec.get("v")
            if G is None or u not in known or v not in known:
                continue    # cluster nodes etc. are recomputed, not restored
            G.add_edge(u, v, **(rec.get("attrs") or {}))
    graph.G_semantic = layer_graphs["semantic"]
    graph.G_citation = layer_graphs["citation"]
    graph.G_causal = layer_graphs["causal"]
    graph.G_section_sim = layer_graphs["section_sim"]

    # scalar state
    try:
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        graph.alpha = float(state.get("alpha", graph.alpha))
        lw = state.get("learned_signal_weights")
        if lw:
            graph._learned_signal_weights = np.asarray(lw, dtype=float)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.debug("state.json not restored: %s", e)

    if rebuild:
        try:
            graph.rebuild_hierarchy()
        except Exception as e:
            logger.debug("rebuild after import skipped: %s", e)
    return graph
