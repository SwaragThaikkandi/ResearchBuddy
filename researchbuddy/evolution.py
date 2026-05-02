"""
Topology evolution analytics for ResearchBuddy.

Usage:
    python -m researchbuddy.evolution
    researchbuddy-evolution
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from researchbuddy.config import DATA_DIR, HISTORY_DIR, STATE_FILE
from researchbuddy.core.graph_model import HierarchicalResearchGraph


@dataclass
class SnapshotMetrics:
    timestamp_iso: str
    snapshot_file: str
    total_papers: int
    rated_papers: int
    avg_user_rating: float
    hierarchy_levels: int
    niche_clusters: int
    area_clusters: int
    semantic_edges: int
    citation_edges: int
    combined_edges: int
    causal_edges: int
    sem_density: float
    combined_density: float
    avg_degree_combined: float
    clustering_combined: float
    transitivity_combined: float
    modularity_combined: float
    centralization_combined: float
    largest_component_frac: float
    avg_shortest_path_lcc: float
    diameter_lcc: float
    query_feedback_events: int
    argument_feedback_events: int


def _safe_float(v: float | int | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _degree_centralization(gu: nx.Graph) -> float:
    n = gu.number_of_nodes()
    if n <= 2:
        return 0.0
    degrees = [d for _, d in gu.degree()]
    max_deg = max(degrees) if degrees else 0
    numer = sum(max_deg - d for d in degrees)
    denom = (n - 1) * (n - 2)
    return float(numer / denom) if denom > 0 else 0.0


def _modularity(gu: nx.Graph) -> float:
    if gu.number_of_nodes() < 3 or gu.number_of_edges() == 0:
        return 0.0
    try:
        communities = list(nx.community.greedy_modularity_communities(gu))
        if len(communities) <= 1:
            return 0.0
        return float(nx.community.modularity(gu, communities))
    except Exception:
        return float("nan")


def _lcc_stats(gu: nx.Graph) -> tuple[float, float, float]:
    """
    Returns:
        largest_component_fraction, avg_shortest_path_in_lcc, diameter_in_lcc
    """
    n = gu.number_of_nodes()
    if n == 0:
        return 0.0, float("nan"), float("nan")
    if n == 1:
        return 1.0, 0.0, 0.0

    components = list(nx.connected_components(gu))
    if not components:
        return 0.0, float("nan"), float("nan")
    largest = max(components, key=len)
    frac = len(largest) / n
    if len(largest) <= 1:
        return frac, 0.0, 0.0

    lcc = gu.subgraph(largest).copy()

    # Keep this tractable for larger graphs.
    if lcc.number_of_nodes() > 2000:
        return frac, float("nan"), float("nan")

    try:
        avg_path = float(nx.average_shortest_path_length(lcc))
    except Exception:
        avg_path = float("nan")
    try:
        diam = float(nx.diameter(lcc))
    except Exception:
        diam = float("nan")
    return frac, avg_path, diam


def _load_graph(path: Path) -> HierarchicalResearchGraph:
    with open(path, "rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, HierarchicalResearchGraph):
        graph = HierarchicalResearchGraph.from_legacy(graph)
    return graph


def compute_metrics_from_graph(
    graph: HierarchicalResearchGraph,
    timestamp_iso: str,
    snapshot_file: str = "",
) -> SnapshotMetrics:
    """
    Compute all evolution metrics from an in-memory graph. Pure function —
    no I/O. Re-used by state_manager.save() to write the JSONL log without
    needing to round-trip a pickle.
    """
    stats = graph.stats()

    rated = graph.rated_papers()
    avg_rating = (
        float(np.mean([m.user_rating for m in rated if m.user_rating is not None]))
        if rated
        else float("nan")
    )

    g_sem = graph.G_semantic.to_undirected()
    g_combined = graph.G.to_undirected()

    sem_density = float(nx.density(g_sem)) if g_sem.number_of_nodes() > 1 else 0.0
    combined_density = float(nx.density(g_combined)) if g_combined.number_of_nodes() > 1 else 0.0
    avg_degree = (
        float(np.mean([d for _, d in g_combined.degree()]))
        if g_combined.number_of_nodes() > 0
        else 0.0
    )
    clustering = (
        float(nx.average_clustering(g_combined))
        if g_combined.number_of_nodes() > 1 and g_combined.number_of_edges() > 0
        else 0.0
    )
    transitivity = (
        float(nx.transitivity(g_combined))
        if g_combined.number_of_nodes() > 2 and g_combined.number_of_edges() > 0
        else 0.0
    )
    modularity = _modularity(g_combined)
    centralization = _degree_centralization(g_combined)
    lcc_frac, avg_path, diameter = _lcc_stats(g_combined)

    return SnapshotMetrics(
        timestamp_iso=timestamp_iso,
        snapshot_file=snapshot_file,
        total_papers=int(stats.get("total_papers", 0)),
        rated_papers=int(stats.get("rated_papers", 0)),
        avg_user_rating=avg_rating,
        hierarchy_levels=int(stats.get("hierarchy_levels", 0)),
        niche_clusters=int(stats.get("niche_clusters", 0)),
        area_clusters=int(stats.get("area_clusters", 0)),
        semantic_edges=int(stats.get("semantic_edges", 0)),
        citation_edges=int(stats.get("citation_edges", 0)),
        combined_edges=int(stats.get("combined_edges", 0)),
        causal_edges=int(stats.get("causal_edges", 0)),
        sem_density=sem_density,
        combined_density=combined_density,
        avg_degree_combined=avg_degree,
        clustering_combined=clustering,
        transitivity_combined=transitivity,
        modularity_combined=modularity,
        centralization_combined=centralization,
        largest_component_frac=lcc_frac,
        avg_shortest_path_lcc=avg_path,
        diameter_lcc=diameter,
        query_feedback_events=len(getattr(graph, "_query_interactions", [])),
        argument_feedback_events=len(getattr(graph, "_argument_interactions", [])),
    )


def _compute_metrics(path: Path) -> SnapshotMetrics:
    """Legacy: load a pickle and compute metrics. Used to ingest old snapshots."""
    graph = _load_graph(path)
    ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return compute_metrics_from_graph(graph, ts, path.name)


def _load_metrics_jsonl(
    jsonl_path: Path, max_rows: int | None = None,
) -> list[SnapshotMetrics]:
    """
    Read evolution metrics from a JSON-lines log. Each line = one save event,
    serialised as the SnapshotMetrics dataclass.
    """
    import json
    if not jsonl_path.exists():
        return []
    rows: list[SnapshotMetrics] = []
    fields = set(SnapshotMetrics.__dataclass_fields__.keys())
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # Fill any missing fields with sensible defaults so older log
            # entries don't break us when we add new metrics.
            data = {k: d.get(k) for k in fields}
            for k, v in data.items():
                if v is None:
                    if k in {"timestamp_iso", "snapshot_file"}:
                        data[k] = ""
                    elif k.endswith("_papers") or k.endswith("_clusters") \
                         or k.endswith("_edges") or k.endswith("_levels") \
                         or k.endswith("_events"):
                        data[k] = 0
                    else:
                        data[k] = float("nan")
            try:
                rows.append(SnapshotMetrics(**data))
            except Exception:
                continue
    rows.sort(key=lambda r: r.timestamp_iso)
    if max_rows and max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def _collect_snapshot_paths(
    history_dir: Path,
    include_current: bool,
    state_file: Path,
    max_snapshots: int | None,
) -> list[Path]:
    paths = sorted(history_dir.glob("graph_*.pkl"))

    if include_current and state_file.exists():
        paths.append(state_file)

    # De-duplicate resolved paths while preserving order.
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if max_snapshots and max_snapshots > 0 and len(deduped) > max_snapshots:
        deduped = deduped[-max_snapshots:]
    return deduped


def _write_csv(rows: list[SnapshotMetrics], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            d = row.__dict__.copy()
            # Preserve numeric NaN as blank in CSV for easier reading.
            for k, v in list(d.items()):
                if isinstance(v, float) and math.isnan(v):
                    d[k] = ""
            w.writerow(d)


def _plot(rows: list[SnapshotMetrics], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    x = [datetime.fromisoformat(r.timestamp_iso) for r in rows]

    def series(name: str) -> list[float]:
        return [_safe_float(getattr(r, name)) for r in rows]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    ax = axes.ravel()

    # 1) Growth
    ax[0].plot(x, series("total_papers"), marker="o", label="papers")
    ax[0].plot(x, series("rated_papers"), marker="o", label="rated")
    ax[0].set_title("Graph Growth")
    ax[0].set_ylabel("Count")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # 2) Edge dynamics
    ax[1].plot(x, series("semantic_edges"), marker="o", label="semantic")
    ax[1].plot(x, series("citation_edges"), marker="o", label="citation")
    ax[1].plot(x, series("combined_edges"), marker="o", label="combined")
    ax[1].set_title("Edge Count Dynamics")
    ax[1].set_ylabel("Edges")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # 3) Local connectivity
    ax[2].plot(x, series("combined_density"), marker="o", label="density")
    ax[2].plot(x, series("clustering_combined"), marker="o", label="clustering")
    ax[2].set_title("Connectivity")
    ax[2].set_ylabel("Ratio")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    # 4) Global navigability
    ax[3].plot(x, series("avg_shortest_path_lcc"), marker="o", label="avg path (LCC)")
    ax[3].plot(x, series("largest_component_frac"), marker="o", label="LCC fraction")
    ax[3].set_title("Navigability")
    ax[3].set_ylabel("Value")
    ax[3].legend()
    ax[3].grid(alpha=0.3)

    # 5) Macro-structure
    ax[4].plot(x, series("modularity_combined"), marker="o", label="modularity")
    ax[4].plot(x, series("centralization_combined"), marker="o", label="centralization")
    ax[4].set_title("Macro Structure")
    ax[4].set_ylabel("Ratio")
    ax[4].legend()
    ax[4].grid(alpha=0.3)

    # 6) Hierarchy evolution
    ax[5].plot(x, series("hierarchy_levels"), marker="o", label="levels")
    ax[5].plot(x, series("niche_clusters"), marker="o", label="niches")
    ax[5].plot(x, series("area_clusters"), marker="o", label="areas")
    ax[5].set_title("Hierarchy Evolution")
    ax[5].set_ylabel("Count")
    ax[5].legend()
    ax[5].grid(alpha=0.3)

    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary(rows: list[SnapshotMetrics], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_txt.write_text("No snapshots found.\n", encoding="utf-8")
        return

    start = rows[0]
    end = rows[-1]
    growth = end.total_papers - start.total_papers
    edge_growth = end.combined_edges - start.combined_edges

    lines = [
        "ResearchBuddy Topology Evolution Summary",
        "=======================================",
        f"Snapshots analyzed: {len(rows)}",
        f"Time range        : {start.timestamp_iso} -> {end.timestamp_iso}",
        "",
        f"Papers            : {start.total_papers} -> {end.total_papers} (delta {growth:+d})",
        f"Combined edges    : {start.combined_edges} -> {end.combined_edges} (delta {edge_growth:+d})",
        f"Hierarchy levels  : {start.hierarchy_levels} -> {end.hierarchy_levels}",
        f"Niches            : {start.niche_clusters} -> {end.niche_clusters}",
        f"Combined density  : {start.combined_density:.4f} -> {end.combined_density:.4f}",
        f"Clustering coeff  : {start.clustering_combined:.4f} -> {end.clustering_combined:.4f}",
        f"Modularity        : {start.modularity_combined:.4f} -> {end.modularity_combined:.4f}",
        f"LCC fraction      : {start.largest_component_frac:.4f} -> {end.largest_component_frac:.4f}",
        "",
        "Interpretation:",
        "- Rising density + clustering: stronger local thematic consolidation.",
        "- Rising modularity: clearer niche separation.",
        "- Rising LCC fraction: better global connectivity and discoverability.",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Quantify and visualize ResearchBuddy graph topology evolution."
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIR,
        help="Directory containing timestamped graph_*.pkl snapshots.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=STATE_FILE,
        help="Current state pickle (optionally included as the latest point).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "evolution",
        help="Output directory for CSV/PNG/TXT artifacts.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=500,
        help="Analyze only the most recent N snapshots.",
    )
    parser.add_argument(
        "--no-current",
        action="store_true",
        help="Do not append the current state file as the newest data point.",
    )
    args = parser.parse_args()

    rows: list[SnapshotMetrics] = []

    # Preferred path: read the lightweight JSONL log written at every save.
    # Each line is ~1 KB vs ~50-200 MB per pickle.
    jsonl_path = args.history_dir / "evolution.jsonl"
    if jsonl_path.exists():
        rows = _load_metrics_jsonl(jsonl_path, args.max_snapshots)
        print(f"[evolution] Loaded {len(rows)} entries from evolution.jsonl")

    # Fallback / supplement: any leftover pickle snapshots get folded in too.
    paths = _collect_snapshot_paths(
        history_dir=args.history_dir,
        include_current=not args.no_current,
        state_file=args.state_file,
        max_snapshots=args.max_snapshots,
    )
    if paths:
        existing_files = {r.snapshot_file for r in rows}
        for p in paths:
            if p.name in existing_files:
                continue
            try:
                rows.append(_compute_metrics(p))
            except Exception as e:
                print(f"[evolution] Skipping {p.name}: {e}")

    rows.sort(key=lambda r: r.timestamp_iso)
    if not rows:
        print("[evolution] No history data found (no JSONL log and no pickles).")
        return

    out_csv = args.out_dir / "graph_evolution_metrics.csv"
    out_png = args.out_dir / "graph_evolution_timeline.png"
    out_txt = args.out_dir / "graph_evolution_summary.txt"

    _write_csv(rows, out_csv)
    _plot(rows, out_png)
    _write_summary(rows, out_txt)

    print(f"[evolution] Snapshots analyzed: {len(rows)}")
    print(f"[evolution] Metrics CSV: {out_csv}")
    print(f"[evolution] Timeline PNG: {out_png}")
    print(f"[evolution] Summary TXT: {out_txt}")


if __name__ == "__main__":
    main()
