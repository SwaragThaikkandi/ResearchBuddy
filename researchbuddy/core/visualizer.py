"""
visualizer.py
Render the three research networks as separate multi-page PDFs.

  save_semantic_pdf   — NLP / Hierarchical Small World Network
  save_citation_pdf   — directed citation graph
  save_combined_pdf   — fused graph (semantic + citation edges together)
  save_all_pdfs       — convenience wrapper that calls all three

Each PDF is written to the path specified in config.py
(SEMANTIC_PDF, CITATION_PDF, COMBINED_PDF).

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
_LEVEL_COLOURS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
                   "#937860", "#DA8BC3", "#8C8C8C"]
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
    """Position papers within their niche cluster, niches in a ring."""
    if not cluster_map:
        return _spring_pos(G, seed)

    niche_ids  = sorted(set(cluster_map.values()))
    n_niches   = max(len(niche_ids), 1)
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

    for nid in niche_ids:
        pos[nid] = niche_centers[nid]

    # Higher-level clusters: rough centre of their children
    for node_id in G.nodes():
        if node_id not in pos:
            preds = [c for c in G.predecessors(node_id) if c in pos]
            if preds:
                xs = [pos[c][0] for c in preds]
                ys = [pos[c][1] for c in preds]
                pos[node_id] = (float(np.mean(xs)), float(np.mean(ys)))
            else:
                pos[node_id] = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    return pos


# ── Shared page helpers ────────────────────────────────────────────────────────

def _draw_edges(ax, G, pos, etype_colours: dict, default_colour: str = "#cccccc"):
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        etype  = data.get("etype", "")
        colour = etype_colours.get(etype, default_colour)
        alpha  = 0.5 if etype == "member" else (_SHORTCUT_ALPHA if etype == "shortcut" else _EDGE_ALPHA)
        lw     = 0.4 if etype in ("member", "shortcut") else 0.8
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", color=colour,
                                    alpha=alpha, lw=lw))


def _draw_arrows(ax, G, pos, colour="#DD8452", alpha=0.6, lw=0.8):
    """Draw directed arrows for citation edges."""
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    alpha=alpha, lw=lw,
                                    mutation_scale=8))


def _scatter_papers(ax, paper_ids, pos, cluster_map, weights,
                    niche_list=None, alpha=_NODE_ALPHA):
    if not paper_ids:
        return
    if niche_list is None:
        niche_list = sorted(set(cluster_map.values()))
    niche_idx = {nid: i for i, nid in enumerate(niche_list)}
    valid = [p for p in paper_ids if p in pos]
    if not valid:
        return
    ppos    = np.array([pos[p] for p in valid])
    sizes   = [max(30, weights.get(p, 5) * 20) for p in valid]
    colours = [_LEVEL_COLOURS[niche_idx.get(cluster_map.get(p, ""), 0) % len(_LEVEL_COLOURS)]
               for p in valid]
    ax.scatter(ppos[:, 0], ppos[:, 1], s=sizes, c=colours,
               alpha=alpha, zorder=3, linewidths=0.3, edgecolors="white")


def _page_stats(pdf: PdfPages, stats: dict, top_papers: list[tuple]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ResearchBuddy — Session Summary", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    rows = [[k.replace("_", " ").title(), str(v)] for k, v in stats.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                    cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1])

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Top-Rated Papers", fontsize=10, fontweight="bold")
    if top_papers:
        y = 0.95
        for rating, _, title in top_papers[:8]:
            ax2.text(0.02, y, f"[{rating:.0f}/10]  {textwrap.shorten(title, 55)}",
                     transform=ax2.transAxes, fontsize=7, va="top")
            y -= 0.11
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_level_summary(pdf: PdfPages, clusters, n_papers: int):
    """Bar chart: number of clusters at each hierarchy level."""
    from collections import Counter
    level_counts = Counter(nd.level for nd in clusters.values())
    if not level_counts:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    levels  = sorted(level_counts.keys())
    counts  = [level_counts[l] for l in levels]
    labels  = [f"Level {l}" for l in levels]
    colours = [_LEVEL_COLOURS[i % len(_LEVEL_COLOURS)] for i in range(len(levels))]
    bars    = ax.bar(labels, counts, color=colours, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title(
        f"Adaptive Hierarchy: {len(levels)} level(s) detected  "
        f"({n_papers} papers total)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Number of clusters")
    ax.set_xlabel("Hierarchy level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SEMANTIC / HSWN  PDF
# ═══════════════════════════════════════════════════════════════════════════════

def _sem_page_overview(pdf, G, pos, paper_ids, cluster_map, weights, title):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    niche_ids  = set(cluster_map.values())
    higher_ids = set(G.nodes()) - set(paper_ids) - niche_ids

    _draw_edges(ax, G, pos,
                etype_colours={"semantic": "#4C72B0", "shortcut": "#55A868",
                                "member": "#aaaaaa", "cluster_sim": "#C44E52"})
    _scatter_papers(ax, paper_ids, pos, cluster_map, weights)

    if niche_ids:
        npos = np.array([pos[n] for n in niche_ids if n in pos])
        if len(npos):
            ax.scatter(npos[:, 0], npos[:, 1], s=200, marker="D",
                       c="#DD8452", alpha=0.9, zorder=4, linewidths=0.5,
                       edgecolors="white")

    if higher_ids:
        hpos = np.array([pos[n] for n in higher_ids if n in pos])
        if len(hpos):
            ax.scatter(hpos[:, 0], hpos[:, 1], s=350, marker="*",
                       c="#C44E52", alpha=0.9, zorder=5, linewidths=0.5,
                       edgecolors="white")

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Semantic edge"),
        mpatches.Patch(color="#55A868", label="Shortcut edge (small-world)"),
        mpatches.Patch(color="#DD8452", label="Niche cluster"),
        mpatches.Patch(color="#C44E52", label="Area / higher cluster"),
        mpatches.Patch(color="#aaaaaa", label="Membership edge"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _sem_page_paper_layer(pdf, paper_ids, pos, cluster_map, weights, titles, G):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title("NLP Paper Layer  (coloured by research niche, size ∝ rating)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    for u, v, d in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        if d.get("etype") not in ("semantic", "shortcut"):
            continue
        alpha = 0.5 if d.get("etype") == "semantic" else 0.2
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#aaaaaa", alpha=alpha, lw=0.5, zorder=1)

    niche_list = sorted(set(cluster_map.values()))
    niche_idx  = {nid: i for i, nid in enumerate(niche_list)}
    for pid in paper_ids:
        if pid not in pos:
            continue
        x, y   = pos[pid]
        niche  = cluster_map.get(pid, "")
        colour = _LEVEL_COLOURS[niche_idx.get(niche, 0) % len(_LEVEL_COLOURS)]
        size   = max(40, weights.get(pid, 5) * 25)
        ax.scatter(x, y, s=size, c=colour, alpha=_NODE_ALPHA, zorder=3,
                   linewidths=0.4, edgecolors="white")
        if weights.get(pid, 0) >= 7:
            label = textwrap.shorten(titles.get(pid, pid), 30)
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=5, alpha=0.75)

    patches = [mpatches.Patch(color=_LEVEL_COLOURS[i % len(_LEVEL_COLOURS)],
                               label=f"Niche {nid}")
               for i, nid in enumerate(niche_list[:8])]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _sem_page_niche_layer(pdf, G, clusters, pos):
    niche_nodes = {nid: nd for nid, nd in clusters.items() if nd.level == 1}
    if not niche_nodes:
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title("Niche Layer  (research niches and their semantic relationships)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    niche_pos = {nid: pos[nid] for nid in niche_nodes if nid in pos}
    if len(niche_pos) < 2:
        plt.close(fig)
        return

    for u, v, d in G.edges(data=True):
        if u not in niche_pos or v not in niche_pos:
            continue
        if d.get("etype") != "cluster_sim":
            continue
        w = d.get("weight", 0.3)
        ax.plot([niche_pos[u][0], niche_pos[v][0]],
                [niche_pos[u][1], niche_pos[v][1]],
                color="#4C72B0", alpha=min(w * 0.8, 0.7), lw=w * 2, zorder=1)

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
        label = f"{nid}\n({len(nd.paper_ids)} papers)"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.85)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def save_semantic_pdf(graph: "HierarchicalResearchGraph", output_path: Path):
    """
    Render the NLP / Hierarchical Small World Network as a multi-page PDF.
    Pages: overview · paper layer · niche layer · level histogram · stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    papers    = graph.all_papers()
    paper_ids = [p.paper_id for p in papers]
    cluster_map = graph.paper_to_niche()
    weights   = {p.paper_id: p.effective_weight for p in papers}
    titles    = {p.paper_id: p.title for p in papers}
    G         = graph.G_semantic

    pos = _cluster_aware_pos(G, paper_ids, cluster_map)

    n_levels = max((nd.level for nd in graph._clusters.values()), default=0)
    overview_title = (
        f"ResearchBuddy — NLP / Semantic Network (HSWN)\n"
        f"{len(paper_ids)} papers  ·  "
        f"{len(graph._clusters)} clusters  ·  "
        f"{n_levels} level(s)  ·  "
        f"{G.number_of_edges()} edges"
    )

    top_papers = sorted(
        [(m.user_rating, m.paper_id, m.title) for m in papers if m.user_rating],
        reverse=True
    )[:8]

    with PdfPages(output_path) as pdf:
        d = pdf.infodict()
        d["Title"]   = "ResearchBuddy — Semantic Network"
        d["Subject"] = "NLP Hierarchical Small World Network"

        _sem_page_overview(pdf, G, pos, paper_ids, cluster_map, weights, overview_title)
        _sem_page_paper_layer(pdf, paper_ids, pos, cluster_map, weights, titles, G)
        _sem_page_niche_layer(pdf, G, graph._clusters, pos)
        _page_level_summary(pdf, graph._clusters, len(paper_ids))
        _page_stats(pdf, graph.stats(), top_papers)

    print(f"[viz] Semantic network PDF saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CITATION PDF
# ═══════════════════════════════════════════════════════════════════════════════

def _cit_page_overview(pdf, G_cit, paper_ids, weights, titles, cluster_map):
    """
    Citation network overview.
    Blue lines: bibliographic coupling (shared external references).
    Orange arrows: direct citation within corpus (rare for small sets).
    Node size ∝ user rating.
    """
    paper_nodes = [n for n in G_cit.nodes() if n in set(paper_ids)]
    if not paper_nodes:
        return

    sub = G_cit.subgraph(paper_nodes)
    n_bib = sum(1 for _, _, d in sub.edges(data=True) if d.get("etype") == "bib_coupling")
    n_cit = sum(1 for _, _, d in sub.edges(data=True) if d.get("etype") == "citation")

    # Use spring layout weighted by coupling strength so tightly-coupled papers cluster
    weight_dict = {(u, v): d.get("weight", 0.1)
                   for u, v, d in sub.edges(data=True)
                   if d.get("etype") == "bib_coupling"}
    if weight_dict:
        import networkx as nx_inner
        pos = nx_inner.spring_layout(sub, weight="weight", seed=42,
                                     k=2.0 / max(len(sub), 1) ** 0.5,
                                     iterations=80)
    else:
        pos = _spring_pos(sub)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(
        f"ResearchBuddy — Citation / Bibliographic Coupling Network\n"
        f"{len(paper_nodes)} papers  ·  "
        f"{n_bib} coupling edges  ·  {n_cit} direct citation edges",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.axis("off")

    # Draw bibliographic coupling edges (thickness ∝ coupling strength)
    for u, v, d in sub.edges(data=True):
        if u not in pos or v not in pos:
            continue
        etype = d.get("etype", "")
        if etype == "bib_coupling":
            w = d.get("weight", 0.1)
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#4C72B0", alpha=min(0.15 + w * 0.6, 0.75),
                    lw=0.5 + w * 2.5, zorder=1)

    # Draw direct citation arrows on top
    for u, v, d in sub.edges(data=True):
        if u not in pos or v not in pos:
            continue
        if d.get("etype") == "citation":
            ax.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="-|>", color="#DD8452",
                                        alpha=0.8, lw=1.2, mutation_scale=10))

    _scatter_papers(ax, paper_nodes, pos, cluster_map, weights)

    # Label highest-degree nodes by coupling
    degree = dict(sub.degree(weight="weight"))
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    for pid, deg in top_nodes:
        if pid in pos and deg > 0:
            ax.annotate(
                textwrap.shorten(titles.get(pid, pid), 28),
                pos[pid], textcoords="offset points",
                xytext=(4, 4), fontsize=5, alpha=0.8
            )

    legend_patches = [
        mpatches.Patch(color="#4C72B0",
                       label="Bibliographic coupling (shared references, width ∝ strength)"),
        mpatches.Patch(color="#DD8452",
                       label="Direct citation within corpus"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _cit_page_coupling_heatmap(pdf, G_cit, paper_ids, titles):
    """
    Coupling strength matrix and top-coupling pairs bar chart.
    Shows which pairs of papers share the most references.
    """
    paper_nodes = [n for n in G_cit.nodes() if n in set(paper_ids)]
    sub = G_cit.subgraph(paper_nodes)

    # Collect coupling pairs
    pairs = [(u, v, d["weight"], d.get("shared_refs", 0))
             for u, v, d in sub.edges(data=True)
             if d.get("etype") == "bib_coupling" and u < v]
    if not pairs:
        return

    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:15]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{textwrap.shorten(titles.get(u,'?'), 20)} ↔ "
              f"{textwrap.shorten(titles.get(v,'?'), 20)}"
              for u, v, w, sr in top]
    values = [w for _, _, w, _ in top]
    shared = [sr for _, _, _, sr in top]
    colours = [_LEVEL_COLOURS[i % len(_LEVEL_COLOURS)] for i in range(len(top))]
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1],
                   alpha=0.85, edgecolor="white")
    for bar, sr in zip(bars, shared[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{sr} shared refs", va="center", fontsize=7)
    ax.set_title("Top Bibliographically Coupled Pairs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coupling strength (Kessler normalised)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def save_citation_pdf(graph: "HierarchicalResearchGraph", output_path: Path):
    """
    Render the directed citation graph as a multi-page PDF.
    Pages: citation overview · in-degree ranking · stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    papers      = graph.all_papers()
    paper_ids   = [p.paper_id for p in papers]
    cluster_map = graph.paper_to_niche()
    weights     = {p.paper_id: p.effective_weight for p in papers}
    titles      = {p.paper_id: p.title for p in papers}
    G_cit       = graph.G_citation

    top_papers = sorted(
        [(m.user_rating, m.paper_id, m.title) for m in papers if m.user_rating],
        reverse=True
    )[:8]

    with PdfPages(output_path) as pdf:
        d = pdf.infodict()
        d["Title"]   = "ResearchBuddy — Citation Network"
        d["Subject"] = "Directed citation graph"

        _cit_page_overview(pdf, G_cit, paper_ids, weights, titles, cluster_map)
        _cit_page_coupling_heatmap(pdf, G_cit, paper_ids, titles)
        _page_stats(pdf, graph.stats(), top_papers)

    print(f"[viz] Citation network PDF saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COMBINED / FUSED PDF
# ═══════════════════════════════════════════════════════════════════════════════

def _comb_page_overview(pdf, G, pos, paper_ids, cluster_map, weights, title):
    """Combined graph: semantic edges in blue, citation edges in orange."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    niche_ids  = set(cluster_map.values())
    higher_ids = set(G.nodes()) - set(paper_ids) - niche_ids

    _draw_edges(ax, G, pos,
                etype_colours={
                    "semantic":    "#4C72B0",
                    "shortcut":    "#55A868",
                    "member":      "#cccccc",
                    "cluster_sim": "#C44E52",
                    "citation":    "#DD8452",
                    "bib_coupling": "#E8A45A",
                    "co_citation": "#E8A45A",
                })

    _scatter_papers(ax, paper_ids, pos, cluster_map, weights)

    if niche_ids:
        npos = np.array([pos[n] for n in niche_ids if n in pos])
        if len(npos):
            ax.scatter(npos[:, 0], npos[:, 1], s=200, marker="D",
                       c="#DD8452", alpha=0.9, zorder=4, linewidths=0.5,
                       edgecolors="white")

    if higher_ids:
        hpos = np.array([pos[n] for n in higher_ids if n in pos])
        if len(hpos):
            ax.scatter(hpos[:, 0], hpos[:, 1], s=350, marker="*",
                       c="#C44E52", alpha=0.9, zorder=5, linewidths=0.5,
                       edgecolors="white")

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Semantic edge (NLP similarity)"),
        mpatches.Patch(color="#DD8452", label="Citation edge (A cites B)"),
        mpatches.Patch(color="#55A868", label="Shortcut edge (small-world)"),
        mpatches.Patch(color="#C44E52", label="Area / higher cluster"),
        mpatches.Patch(color="#aaaaaa", label="Membership edge"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _comb_page_edge_breakdown(pdf, G_sem, G_cit, G_comb):
    """Pie / bar chart: edge type breakdown across all three networks."""
    from collections import Counter

    def count_etypes(G):
        return Counter(d.get("etype", "unknown") for _, _, d in G.edges(data=True))

    c_sem  = count_etypes(G_sem)
    c_cit  = count_etypes(G_cit)
    c_comb = count_etypes(G_comb)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Edge Type Breakdown Across Three Networks",
                 fontsize=13, fontweight="bold")

    def _pie(ax, counter, title):
        if not counter:
            ax.set_title(title + "\n(no edges)")
            ax.axis("off")
            return
        labels = list(counter.keys())
        values = list(counter.values())
        colours = [_LEVEL_COLOURS[i % len(_LEVEL_COLOURS)] for i in range(len(labels))]
        ax.pie(values, labels=labels, colors=colours, autopct="%1.0f%%",
               startangle=140, textprops={"fontsize": 8})
        ax.set_title(title, fontsize=10, fontweight="bold")

    _pie(axes[0], c_sem,  "Semantic Network")
    _pie(axes[1], c_cit,  "Citation Network")
    _pie(axes[2], c_comb, "Combined Network")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def save_combined_pdf(graph: "HierarchicalResearchGraph", output_path: Path):
    """
    Render the combined (semantic + citation) fused graph as a multi-page PDF.
    Pages: combined overview · paper layer (dual edges) · edge breakdown · stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    papers      = graph.all_papers()
    paper_ids   = [p.paper_id for p in papers]
    cluster_map = graph.paper_to_niche()
    weights     = {p.paper_id: p.effective_weight for p in papers}
    titles      = {p.paper_id: p.title for p in papers}
    G           = graph.G

    pos = _cluster_aware_pos(G, paper_ids, cluster_map)

    overview_title = (
        f"ResearchBuddy — Combined Network (Semantic + Citation)\n"
        f"{len(paper_ids)} papers  ·  "
        f"{graph.G_semantic.number_of_edges()} semantic edges  ·  "
        f"{graph.G_citation.number_of_edges()} citation edges  ·  "
        f"{G.number_of_edges()} total edges"
    )

    top_papers = sorted(
        [(m.user_rating, m.paper_id, m.title) for m in papers if m.user_rating],
        reverse=True
    )[:8]

    with PdfPages(output_path) as pdf:
        d = pdf.infodict()
        d["Title"]   = "ResearchBuddy — Combined Network"
        d["Subject"] = "Fused semantic and citation research graph"

        _comb_page_overview(pdf, G, pos, paper_ids, cluster_map, weights, overview_title)
        _comb_page_edge_breakdown(pdf, graph.G_semantic, graph.G_citation, G)
        _page_stats(pdf, graph.stats(), top_papers)

    print(f"[viz] Combined network PDF saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def save_all_pdfs(graph: "HierarchicalResearchGraph"):
    """Generate all three PDFs using paths from config."""
    from researchbuddy.config import SEMANTIC_PDF, CITATION_PDF, COMBINED_PDF
    save_semantic_pdf(graph, SEMANTIC_PDF)
    save_citation_pdf(graph, CITATION_PDF)
    save_combined_pdf(graph, COMBINED_PDF)
