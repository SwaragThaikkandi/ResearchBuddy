#!/usr/bin/env python3
"""
ResearchBuddy CLI — interactive literature search session.

Usage (after pip install):
    researchbuddy                         # load saved graph and start session
    researchbuddy --pdf <folder>          # import PDFs then start session
    researchbuddy --reset                 # clear saved state, start fresh

Tunable parameters (override config defaults):
    researchbuddy --alpha 0.7             # more weight on semantic similarity
    researchbuddy --exploration-ratio 0.4 # more exploratory suggestions
    researchbuddy --similarity-threshold 0.5
    researchbuddy --n-recommendations 15
    researchbuddy --no-plot               # skip PDF graph export

Run without installing:
    python -m researchbuddy [flags]
"""

from __future__ import annotations

import argparse
import textwrap
import time

import researchbuddy.config as cfg
from researchbuddy.core.graph_model   import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.state_manager import save, load, import_pdf_folder, resolve_seed_s2_ids
from researchbuddy.core.searcher      import find_candidates

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import box
    from rich.prompt  import Prompt
    console  = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

DIVIDER = "-" * 72


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_header(text: str):
    if HAS_RICH:
        console.rule(f"[bold cyan]{text}[/]")
    else:
        print(f"\n{'='*72}\n  {text}\n{'='*72}")

def print_info(text: str):
    if HAS_RICH:
        console.print(f"[dim]{text}[/]")
    else:
        print(text)

def print_success(text: str):
    if HAS_RICH:
        console.print(f"[green]{text}[/]")
    else:
        print(f"OK  {text}")

def print_warn(text: str):
    if HAS_RICH:
        console.print(f"[yellow]{text}[/]")
    else:
        print(f"!   {text}")

def ask(prompt: str, default: str = "") -> str:
    try:
        if HAS_RICH:
            return Prompt.ask(prompt, default=default)
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        return default


# ── Paper display ──────────────────────────────────────────────────────────────

def display_paper(idx: int, meta: PaperMeta, score: float, label: str):
    score_pct = f"{score * 100:.0f}%"
    year      = str(meta.year) if meta.year else "?"
    auth      = ", ".join(meta.authors[:2]) if meta.authors else "Unknown"
    if len(meta.authors) > 2:
        auth += " et al."
    snippet = textwrap.fill(
        meta.abstract[:300] + ("..." if len(meta.abstract) > 300 else ""),
        width=68, initial_indent="    ", subsequent_indent="    "
    )
    if HAS_RICH:
        if label == "explore":
            title_str = f"[bold magenta]* EXPLORE  {meta.title[:80]}[/]"
        else:
            title_str = f"[bold]{meta.title[:90]}[/]"
        console.print(f"\n  [{idx}] {title_str}")
        console.print(f"      [cyan]{auth}[/]  ({year})  match={score_pct}")
        console.print(f"[dim]{snippet}[/]")
        if meta.url:
            console.print(f"      [blue underline]{meta.url}[/]")
    else:
        tag = " [EXPLORE]" if label == "explore" else ""
        print(f"\n  [{idx}] {meta.title[:90]}{tag}")
        print(f"      {auth}  ({year})  match={score_pct}")
        print(snippet)
        if meta.url:
            print(f"      {meta.url}")


def display_results(results: list[tuple[PaperMeta, float, str]]):
    print_header("Search Results")
    if not results:
        print_warn("No new candidates found. Try different keywords or add more PDFs.")
        return
    for i, (meta, score, label) in enumerate(results, start=1):
        display_paper(i, meta, score, label)
    print(f"\n{DIVIDER}")


# ── Rating workflow ────────────────────────────────────────────────────────────

def rating_session(graph: HierarchicalResearchGraph, results: list[tuple[PaperMeta, float, str]]):
    if not results:
        return
    print_header("Rate Papers")
    print_info("Rate 1-10 (0 or Enter = skip, q = done).  * = exploratory paper.")
    print()

    rated_any = False
    for i, (meta, _, label) in enumerate(results, start=1):
        tag = "[*] " if label == "explore" else "    "
        if HAS_RICH:
            console.print(f"  [{i}] {tag}[bold]{meta.title[:80]}[/]")
        else:
            print(f"  [{i}] {tag}{meta.title[:80]}")

        raw = ask("      Rating", "0")
        if raw.lower() == "q":
            break
        try:
            rating = int(raw)
        except ValueError:
            rating = 0
        if rating == 0:
            continue
        rating = max(1, min(10, rating))

        meta.times_shown += 1
        meta.last_shown   = time.time()
        if meta.paper_id not in [p.paper_id for p in graph.all_papers()]:
            graph.embed_abstract(meta)
            graph.add_paper(meta, meta.embedding)
        graph.rate_paper(meta.paper_id, float(rating))
        rated_any = True

        if rating >= 7:
            print_success(f"      Highly relevant (weight={rating}).")
        elif rating >= 4:
            print_info(f"      Moderate relevance (weight={rating}).")
        else:
            print_warn(f"      Low relevance — will be used as negative example.")

    if rated_any:
        print_info("\n[graph] Rebuilding hierarchy with new ratings ...")
        graph.rebuild_hierarchy()


# ── Stats ──────────────────────────────────────────────────────────────────────

def show_stats(graph: HierarchicalResearchGraph):
    stats = graph.stats()
    print_header("Graph Statistics")
    if HAS_RICH:
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("Metric", style="cyan")
        t.add_column("Value",  style="bold")
        for k, v in stats.items():
            t.add_row(k.replace("_", " ").title(), str(v))
        console.print(t)
    else:
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")

    rated = sorted(
        [m for m in graph.rated_papers() if m.user_rating is not None],
        key=lambda m: m.user_rating, reverse=True
    )
    if rated:
        print_info("\n  Top-rated papers:")
        for m in rated[:5]:
            print(f"    [{m.user_rating:.0f}/10] {m.title[:70]}")

    print_info(f"\n  Active parameters:")
    print_info(f"    alpha (semantic weight) = {graph.alpha}")
    print_info(f"    hierarchy levels        = auto-detected ({stats.get('hierarchy_levels', 0)})")
    print_info(f"    exploration ratio       = {cfg.EXPLORATION_RATIO}")
    print_info(f"    similarity threshold    = {cfg.SIMILARITY_THRESHOLD}")


# ── Search session ─────────────────────────────────────────────────────────────

def search_session(graph: HierarchicalResearchGraph, plot: bool = True):
    if graph.context_vector() is None:
        print_warn("No context vector yet. Add seed PDFs (option 3) or rate some papers first.")
        return

    print_info("\nOptional: extra search keywords (comma-separated), or Enter to skip:")
    raw   = ask("Keywords", "")
    extra = [kw.strip() for kw in raw.split(",") if kw.strip()] if raw.strip() else []

    print()
    print_info("Searching ...  (may take 20-40 s)\n")
    candidates = find_candidates(graph, extra_keywords=extra)
    if not candidates:
        print_warn("No results returned. Check your internet connection.")
        return

    print_info("Ranking candidates (fused semantic + citation scores) ...")
    results = graph.rank_candidates(candidates, n=cfg.N_RECOMMENDATIONS,
                                    exploration_ratio=cfg.EXPLORATION_RATIO)
    display_results(results)

    if results:
        if ask("\nRate these papers? (y/n)", "y").lower() == "y":
            rating_session(graph, results)
        print_info("Auto-saving ...")
        save(graph)

    # Generate all three PDF graphs after each search session
    if plot and cfg.SAVE_GRAPH_PDF:
        _try_plot_all(graph)


def _try_plot_all(graph: HierarchicalResearchGraph):
    """Generate semantic, citation, and combined PDF graphs."""
    try:
        from researchbuddy.core.visualizer import save_all_pdfs
        print_info("Generating network PDFs (semantic / citation / combined) ...")
        save_all_pdfs(graph)
        print_success(f"PDFs saved to {cfg.DATA_DIR}/")
    except Exception as e:
        print_warn(f"Graph PDF generation skipped: {e}")


# ── Main menu ──────────────────────────────────────────────────────────────────

def main_menu(graph: HierarchicalResearchGraph, plot: bool = True):
    options = {
        "1": "Search for new papers",
        "2": "Show graph statistics",
        "3": "Add PDF folder",
        "4": "Fetch citation data (improves fusion quality)",
        "5": "Resolve Semantic Scholar IDs for seed papers",
        "6": "Rebuild hierarchy & regenerate all graph PDFs",
        "q": "Save & quit",
    }
    while True:
        print_header("ResearchBuddy")
        s = graph.stats()
        print_info(
            f"  {s['total_papers']} papers  |  {s['rated_papers']} rated  |  "
            f"{s['hierarchy_levels']} levels  |  "
            f"{s['niche_clusters']} niches  |  {s['area_clusters']} areas  |  "
            f"sem={s['semantic_edges']} edges  cit={s['citation_edges']} edges"
        )
        print()
        for key, desc in options.items():
            if HAS_RICH:
                console.print(f"  [cyan][{key}][/] {desc}")
            else:
                print(f"  [{key}] {desc}")
        print()

        choice = ask("Choose", "1").strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            search_session(graph, plot=plot)
        elif choice == "2":
            show_stats(graph)
        elif choice == "3":
            folder = ask("Path to PDF folder", "")
            if folder:
                import_pdf_folder(graph, folder)
                save(graph)
        elif choice == "4":
            graph.fetch_citations()
            graph.rebuild_hierarchy()
            save(graph)
        elif choice == "5":
            if ask("Make API calls to S2 for each seed paper? (y/n)", "n").lower() == "y":
                resolve_seed_s2_ids(graph)
                save(graph)
        elif choice == "6":
            print_info("Rebuilding hierarchy ...")
            graph.rebuild_hierarchy()
            save(graph)
            if plot:
                _try_plot_all(graph)
        else:
            print_warn("Unknown option.")


# ── CLI argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ResearchBuddy – hierarchical graph-based literature search assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf",    metavar="FOLDER", help="Import PDFs from folder on startup")
    p.add_argument("--reset",  action="store_true", help="Clear saved state and start fresh")

    # ── Tunable parameters ──────────────────────────────────────────────────
    g = p.add_argument_group("model parameters (override config.py defaults)")
    g.add_argument("--alpha", type=float, default=None,
                   metavar="FLOAT",
                   help=f"Semantic vs citation weight (0-1). Default: {cfg.FUSION_ALPHA}")
    g.add_argument("--exploration-ratio", type=float, default=None,
                   metavar="FLOAT",
                   help=f"Fraction of suggestions that are exploratory. Default: {cfg.EXPLORATION_RATIO}")
    g.add_argument("--similarity-threshold", type=float, default=None,
                   metavar="FLOAT",
                   help=f"Min cosine sim to draw a graph edge. Default: {cfg.SIMILARITY_THRESHOLD}")
    g.add_argument("--n-recommendations", type=int, default=None,
                   metavar="INT",
                   help=f"Papers shown per search session. Default: {cfg.N_RECOMMENDATIONS}")
    g.add_argument("--no-plot", action="store_true",
                   help="Disable PDF graph generation after each session")
    return p


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()

    # Apply CLI overrides to config module (affects all imports)
    if args.alpha                is not None: cfg.FUSION_ALPHA         = args.alpha
    if args.exploration_ratio    is not None: cfg.EXPLORATION_RATIO    = args.exploration_ratio
    if args.similarity_threshold is not None: cfg.SIMILARITY_THRESHOLD = args.similarity_threshold
    if args.n_recommendations    is not None: cfg.N_RECOMMENDATIONS    = args.n_recommendations
    if args.no_plot:                           cfg.SAVE_GRAPH_PDF       = False
    plot = not args.no_plot

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]ResearchBuddy[/]\n"
            "[dim]Hierarchical graph-based literature search assistant[/]",
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("  ResearchBuddy - Hierarchical literature search assistant")
        print("=" * 60)

    if args.reset and cfg.STATE_FILE.exists():
        cfg.STATE_FILE.unlink()
        print_info("Saved state cleared.")

    graph = load()
    if graph is None:
        graph = HierarchicalResearchGraph(alpha=cfg.FUSION_ALPHA)
        print_info("Starting with a fresh graph.")
    else:
        if args.alpha is not None:
            graph.alpha = cfg.FUSION_ALPHA
        s = graph.stats()
        print_success(
            f"Loaded: {s['total_papers']} papers, "
            f"{s['hierarchy_levels']} levels, "
            f"{s['niche_clusters']} niches, "
            f"{s['rated_papers']} rated."
        )

    if args.pdf:
        import_pdf_folder(graph, args.pdf)
        save(graph)
    elif graph.stats()["total_papers"] == 0:
        print_warn("\nNo papers yet. Let's add your seed PDFs.")
        folder = ask("Path to your PDF folder (or Enter to skip)", "")
        if folder:
            import_pdf_folder(graph, folder)
            save(graph)

    try:
        main_menu(graph, plot=plot)
    except KeyboardInterrupt:
        print("\n[Interrupted]")

    print_info("\nSaving before exit ...")
    save(graph)
    print_success("Session saved. See you next time!")


if __name__ == "__main__":
    main()
