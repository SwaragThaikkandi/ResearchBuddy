"""
visualizer.py
Render the hierarchical research graph as a multi-page PDF.

Page 1  — Overview: full hierarchical graph (papers + cluster nodes)
Page 2  — Paper layer: papers coloured by niche, size ∝ rating
Page 3  — Niche layer: niche nodes coloured by area, edges = inter-niche similarity
Page 4+ — Citation edges (if any)
Last    — Summary statistics table

Relies on matplotlib only (no graphviz required).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")                    # headless — no GUI window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import HierarchicalResearchGraph

# ── Colour palettes ────────────────────────────────────────────────────────────
_LEVEL_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
_NODE_ALPHA     = 0.85
_EDGE_ALPHA     = 0.25
_SHORTCUT_ALPHA = 0.15


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _spring_pos(G: nx.Graph, seed: int = 42) -> dict:
    return nx.spring_layout(G, seed=seed, k=1.5 / max(len(G), 1) ** 0.5,
                            iterations=60)


def _cluster_aware_pos(
    G: nx.DiGraph,
    paper_ids: list[str],
    cluster_map: dict[str, str],  # paper_id → niche_node_id
    seed: int = 42,
) -> dict:
    """
    Position papers within their niche cluster, niches in a ring.
    Falls back to spring layout if cluster info is unavailable.
    """
    if not cluster_map:
        return _spring_pos(G, seed)

    niche_ids = sorted(set(cluster_map.values()))
    n_niches  = max(len(niche_ids), 1)
    niche_centers: dict[str, tuple] = {}
    radius = 3.0
    for i, nid in enumerate(niche_ids):
        angle = 2 * np.pi * i / n_niches
        niche_centers[nid] = (radius * np.cos(angle), radius * np.sin(angle))

    pos: dict = {}
    rng = np.random.default_rng(seed)
    niche_members: dict[str, list] = {nid: [] for nid in niche_ids}
    for pid in paper_ids:
        niche_members[cluster_map.get(pid, niche_ids[0])].append(pid)

    for nid, members in niche_members.items():
        cx, cy = niche_centers[nid]
        for j, pid in enumerate(members):
            angle = 2 * np.pi * j / max(len(members), 1)
            r = 0.5 + 0.3 * rng.random()
            pos[pid] = (cx + r * np.cos(angle), cy + r * np.sin(angle))

    # Place cluster nodes at their centre
    for nid in niche_ids:
        pos[nid] = niche_centers[nid]

    # Higher-level clusters: rough centre of their children
    for node_id in G.nodes():
        if node_id not in pos:
            children_with_pos = [c for c in G.predecessors(node_id) if c in pos]
            if children_with_pos:
                xs = [pos[c][0] for c in children_with_pos]
                ys = [pos[c][1] for c in children_with_pos]
                pos[node_id] = (float(np.mean(xs)), float(np.mean(ys)))
            else:
                pos[node_id] = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    return pos


# ── Individual pages ───────────────────────────────────────────────────────────

def _page_overview(
    pdf: PdfPages,
    G: nx.DiGraph,
    pos: dict,
    paper_ids: list[str],
    cluster_map: dict[str, str],
    weights: dict[str, float],
    title: str,
):
    """Full hierarchical graph overview."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    # Separate nodes by level
    node_levels = nx.get_node_attributes(G, "level")
    niche_ids   = set(cluster_map.values())
    higher_ids  = set(G.nodes()) - set(paper_ids) - niche_ids

    # Draw edges by type
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        etype = data.get("etype", "")
        colour = {"semantic": "#4C72B0", "citation": "#DD8452",
                  "shortcut": "#55A868", "member": "#aaaaaa",
                  "cluster_sim": "#C44E52"}.get(etype, "#cccccc")
        alpha  = 0.5 if etype == "member" else (_SHORTCUT_ALPHA if etype == "shortcut" else _EDGE_ALPHA)
        lw     = 0.4 if etype in ("member", "shortcut") else 0.8
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", color=colour,
                                    alpha=alpha, lw=lw))

    # Paper nodes
    if paper_ids:
        ppos  = np.array([pos[p] for p in paper_ids if p in pos])
        sizes = [max(30, weights.get(p, 5) * 20) for p in paper_ids if p in pos]
        niche_list = sorted(niche_ids)
        niche_idx  = {nid: i for i, nid in enumerate(niche_list)}
        colours    = [_LEVEL_COLOURS[niche_idx.get(cluster_map.get(p, ""), 0)
                                      % len(_LEVEL_COLOURS)]
                      for p in paper_ids if p in pos]
        ax.scatter(ppos[:, 0], ppos[:, 1], s=sizes, c=colours,
                   alpha=_NODE_ALPHA, zorder=3, linewidths=0.3, edgecolors="white")

    # Niche nodes
    if niche_ids:
        npos = np.array([pos[n] for n in niche_ids if n in pos])
        if len(npos):
            ax.scatter(npos[:, 0], npos[:, 1], s=200, marker="D",
                       c="#DD8452", alpha=0.9, zorder=4, linewidths=0.5,
                       edgecolors="white", label="Niche")

    # Higher-level cluster nodes
    if higher_ids:
        hpos = np.array([pos[n] for n in higher_ids if n in pos])
        if len(hpos):
            ax.scatter(hpos[:, 0], hpos[:, 1], s=350, marker="*",
                       c="#C44E52", alpha=0.9, zorder=5, linewidths=0.5,
                       edgecolors="white", label="Area")

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Paper (semantic edge)"),
        mpatches.Patch(color="#DD8452", label="Niche cluster"),
        mpatches.Patch(color="#C44E52", label="Area cluster"),
        mpatches.Patch(color="#55A868", label="Shortcut edge"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_paper_layer(
    pdf: PdfPages,
    paper_ids: list[str],
    pos: dict,
    cluster_map: dict[str, str],
    weights: dict[str, float],
    titles: dict[str, str],
    G_papers: nx.DiGraph,
):
    """Paper-only view: coloured by niche."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title("Paper Layer  (coloured by research niche)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.axis("off")

    niche_list = sorted(set(cluster_map.values()))
    niche_idx  = {nid: i for i, nid in enumerate(niche_list)}

    # Draw paper-paper edges
    for u, v, d in G_papers.edges(data=True):
        if u not in pos or v not in pos or d.get("etype") not in ("semantic", "shortcut", "citation"):
            continue
        alpha = 0.5 if d.get("etype") == "semantic" else 0.2
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#aaaaaa", alpha=alpha, lw=0.5, zorder=1)

    # Draw paper nodes
    for pid in paper_ids:
        if pid not in pos:
            continue
        x, y    = pos[pid]
        niche   = cluster_map.get(pid, "")
        colour  = _LEVEL_COLOURS[niche_idx.get(niche, 0) % len(_LEVEL_COLOURS)]
        size    = max(40, weights.get(pid, 5) * 25)
        ax.scatter(x, y, s=size, c=colour, alpha=_NODE_ALPHA, zorder=3,
                   linewidths=0.4, edgecolors="white")
        # Label high-rated papers
        if weights.get(pid, 0) >= 7:
            label = textwrap.shorten(titles.get(pid, pid), 30)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=5, alpha=0.75)

    # Niche legend
    patches = [mpatches.Patch(color=_LEVEL_COLOURS[i % len(_LEVEL_COLOURS)],
                               label=f"Niche {nid.split('_')[-1]}")
               for i, nid in enumerate(niche_list[:8])]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_niche_layer(
    pdf: PdfPages,
    G: nx.DiGraph,
    clusters,
    pos: dict,
):
    """Niche-level graph: one node per niche, edges = inter-niche similarity."""
    niche_nodes = {nid: nd for nid, nd in clusters.items() if nd.level == 1}
    if not niche_nodes:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title("Niche Layer  (research niches and their relationships)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    niche_pos = {nid: pos[nid] for nid in niche_nodes if nid in pos}
    if len(niche_pos) < 2:
        plt.close(fig)
        return

    # Draw inter-niche edges
    for u, v, d in G.edges(data=True):
        if u not in niche_pos or v not in niche_pos:
            continue
        if d.get("etype") != "cluster_sim":
            continue
        w = d.get("weight", 0.3)
        ax.plot([niche_pos[u][0], niche_pos[v][0]],
                [niche_pos[u][1], niche_pos[v][1]],
                color="#4C72B0", alpha=min(w * 0.8, 0.7), lw=w * 2, zorder=1)

    # Draw niche nodes
    area_map: dict[str, str] = {}
    for nid, nd in clusters.items():
        if nd.level == 2:
            for child in nd.child_ids:
                area_map[child] = nid
    area_list = sorted(set(area_map.values()))
    area_idx  = {aid: i for i, aid in enumerate(area_list)}

    for nid, nd in niche_nodes.items():
        if nid not in niche_pos:
            continue
        x, y   = niche_pos[nid]
        area   = area_map.get(nid, "")
        colour = _LEVEL_COLOURS[area_idx.get(area, 0) % len(_LEVEL_COLOURS)]
        size   = max(150, len(nd.paper_ids) * 40)
        ax.scatter(x, y, s=size, c=colour, marker="D", alpha=0.9, zorder=3,
                   linewidths=0.5, edgecolors="white")
        label = f"Niche {nid.split('_')[-1]}\n({len(nd.paper_ids)} papers)"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.85)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_stats(pdf: PdfPages, stats: dict, top_papers: list[tuple[str, float, str]]):
    """Summary statistics page."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ResearchBuddy — Session Summary", fontsize=14, fontweight="bold")

    # Left: stats table
    ax = axes[0]
    ax.axis("off")
    rows = [[k.replace("_", " ").title(), str(v)] for k, v in stats.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                    cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1])

    # Right: top-rated papers
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Top-Rated Papers", fontsize=10, fontweight="bold")
    if top_papers:
        y = 0.95
        for rating, title in [(r, t) for r, _, t in top_papers[:8]]:
            ax2.text(0.02, y, f"[{rating:.0f}/10]  {textwrap.shorten(title, 55)}",
                     transform=ax2.transAxes, fontsize=7, va="top")
            y -= 0.11
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Main entry point ───────────────────────────────────────────────────────────

def save_graph_pdf(graph: "HierarchicalResearchGraph", output_path: Path):
    """
    Render the full hierarchical research graph as a multi-page PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    papers       = graph.all_papers()
    paper_ids    = [p.paper_id for p in papers]
    cluster_map  = graph.paper_to_niche()    # paper_id → niche_node_id
    weights      = {p.paper_id: p.effective_weight for p in papers}
    titles       = {p.paper_id: p.title for p in papers}
    G            = graph.G

    # Build layout
    pos = _cluster_aware_pos(G, paper_ids, cluster_map)

    # Titles
    overview_title = (
        f"ResearchBuddy — Hierarchical Research Graph\n"
        f"{len(paper_ids)} papers  ·  "
        f"{len(graph._clusters)} clusters  ·  "
        f"{G.number_of_edges()} edges"
    )

    top_papers = sorted(
        [(m.user_rating, m.paper_id, m.title) for m in papers if m.user_rating],
        reverse=True
    )[:8]

    with PdfPages(output_path) as pdf:
        # Page metadata
        d = pdf.infodict()
        d["Title"]   = "ResearchBuddy Graph"
        d["Subject"] = "Hierarchical research literature graph"

        # Page 1: full overview
        _page_overview(pdf, G, pos, paper_ids, cluster_map, weights, overview_title)

        # Page 2: paper layer
        _page_paper_layer(pdf, paper_ids, pos, cluster_map, weights, titles, G)

        # Page 3: niche layer
        _page_niche_layer(pdf, G, graph._clusters, pos)

        # Last page: stats
        _page_stats(pdf, graph.stats(), top_papers)

    print(f"[viz] Graph saved → {output_path}")
