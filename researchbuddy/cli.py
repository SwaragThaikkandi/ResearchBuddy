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
from researchbuddy.core.reasoner      import Reasoner, QueryResult

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


# ── Query / Reasoning mode ("Prefrontal Cortex") ──────────────────────────────

def _short_title(graph: HierarchicalResearchGraph, pid: str, n: int = 45) -> str:
    """Compact paper title from paper_id."""
    p = graph.get_paper(pid)
    if not p:
        return pid[:12]
    t = p.title[:n]
    return (t + "...") if len(p.title) > n else t


def display_query_result(result: QueryResult, graph: HierarchicalResearchGraph):
    """Pretty-print a QueryResult to the terminal."""
    print_header("Query Results")

    # ── Relevant papers (with centrality / role) ─────────────────────────
    if result.relevant_papers:
        if HAS_RICH:
            console.print("[bold]Most Relevant Papers[/]")
        else:
            print("  Most Relevant Papers")

        for i, (meta, score, info) in enumerate(result.relevant_papers, 1):
            pct  = f"{score * 100:.0f}%"
            year = str(meta.year) if meta.year else "?"
            auth = ", ".join(meta.authors[:2]) if meta.authors else "Unknown"
            if len(meta.authors) > 2:
                auth += " et al."
            role_tag = ""
            if info.get("role") == "hub":
                role_tag = "  [HUB]"
            elif info.get("role") == "isolated":
                role_tag = "  [isolated]"
            rated_tag = f"  rated {meta.user_rating:.0f}/10" if meta.user_rating else ""
            deg = info.get("degree", 0)

            if HAS_RICH:
                console.print(f"    [{i}] [bold]{meta.title[:80]}[/]")
                console.print(
                    f"        [cyan]{auth}[/] ({year})  "
                    f"relevance={pct}  {deg} connections{role_tag}{rated_tag}"
                )
            else:
                print(f"    [{i}] {meta.title[:80]}")
                print(f"        {auth} ({year})  relevance={pct}  "
                      f"{deg} connections{role_tag}{rated_tag}")

    # ── Research themes (cluster profiles) ────────────────────────────────
    if result.cluster_profiles:
        print()
        if HAS_RICH:
            console.print("[bold]Research Themes[/]")
        else:
            print("  Research Themes")

        for cp in result.cluster_profiles[:3]:
            pct = f"{cp.similarity * 100:.0f}%"
            yr  = f"avg {cp.avg_year:.0f}" if cp.avg_year else "?"
            central = (f"key: {cp.central_paper.title[:35]}..."
                       if cp.central_paper else "")
            if HAS_RICH:
                console.print(
                    f"    {cp.cluster.node_id}  "
                    f"({cp.n_papers} papers, match={pct}, "
                    f"{cp.maturity}, density={cp.density:.0%})"
                )
                if central:
                    console.print(f"      [dim]{central}[/]")
            else:
                print(f"    {cp.cluster.node_id}  "
                      f"({cp.n_papers} papers, match={pct}, "
                      f"{cp.maturity}, density={cp.density:.0%})")
                if central:
                    print(f"      {central}")

    # ── Research lineages (citation / semantic paths) ─────────────────────
    if result.lineages:
        print()
        if HAS_RICH:
            console.print("[bold]Research Lineages[/]")
        else:
            print("  Research Lineages")

        for lin in result.lineages:
            titles = [_short_title(graph, pid, 35) for pid in lin.path]
            chain  = " -> ".join(titles)
            label  = ("citation chain" if lin.path_type == "citation_chain"
                      else "semantic path")
            if HAS_RICH:
                console.print(f"    [dim]{chain}[/]")
                console.print(f"      ({label})")
            else:
                print(f"    {chain}")
                print(f"      ({label})")

    # ── Connections ───────────────────────────────────────────────────────
    if result.connections:
        print()
        if HAS_RICH:
            console.print("[bold]Connections[/]")
        else:
            print("  Connections")

        for a, b, desc in result.connections[:8]:
            if HAS_RICH:
                console.print(
                    f"    [dim]{_short_title(graph, a)}[/]  <->  "
                    f"[dim]{_short_title(graph, b)}[/]"
                )
                console.print(f"      ({desc})")
            else:
                print(f"    {_short_title(graph, a)}  <->  "
                      f"{_short_title(graph, b)}")
                print(f"      ({desc})")

    # ── Bridge papers ────────────────────────────────────────────────────
    if result.bridge_papers:
        print()
        if HAS_RICH:
            console.print("[bold]Bridge Papers[/] (connect multiple themes)")
        else:
            print("  Bridge Papers (connect multiple themes)")
        for meta in result.bridge_papers:
            if HAS_RICH:
                console.print(f"    [magenta]{meta.title[:70]}[/]")
            else:
                print(f"    {meta.title[:70]}")

    # ── Frontier papers (relevant but underconnected) ─────────────────────
    if result.frontier_papers:
        print()
        if HAS_RICH:
            console.print("[bold]Frontier[/] (relevant but underconnected)")
        else:
            print("  Frontier (relevant but underconnected)")
        for meta, sim in result.frontier_papers:
            pct = f"{sim * 100:.0f}%"
            if HAS_RICH:
                console.print(f"    [yellow]{meta.title[:65]}[/]  ({pct})")
            else:
                print(f"    {meta.title[:65]}  ({pct})")

    # ── Temporal narrative ────────────────────────────────────────────────
    if result.temporal_narrative:
        print()
        print_info(f"  Timeline: {result.temporal_narrative}")

    # ── Coverage note ────────────────────────────────────────────────────
    if result.gap_note:
        print()
        print_warn(result.gap_note)

    print(f"\n{DIVIDER}")


def query_session(graph: HierarchicalResearchGraph):
    """Interactive reasoning loop — the 'prefrontal cortex'."""
    if not graph.all_papers():
        print_warn("No papers in your graph yet. Add PDFs (option 3) first.")
        return

    reasoner = Reasoner(top_k=cfg.QUERY_TOP_K)
    print_header("Reasoning Mode")
    print_info("Ask questions about your research collection.")
    print_info("Type 'q' or 'quit' to return to the main menu.\n")

    while True:
        raw = ask("Your query", "")
        if not raw or raw.strip().lower() in ("q", "quit", "exit"):
            break

        print_info("\nThinking ...\n")
        result = reasoner.reason(raw.strip(), graph)
        display_query_result(result, graph)

        # ── Feedback ─────────────────────────────────────────────────────
        rating_raw = ask("Rate this response (1-10, 0=skip)", "0")
        try:
            rating = int(rating_raw)
        except ValueError:
            rating = 0
        if rating < 0 or rating > 10:
            rating = 0

        if rating > 0:
            paper_ids = [m.paper_id for m, _, _ in result.relevant_papers]
            graph.apply_query_feedback(
                result.query_embedding, paper_ids, float(rating)
            )
            if rating >= 7:
                print_success(
                    "Network updated — edges strengthened between relevant "
                    "papers. Future results will lean this way."
                )
            elif rating >= 4:
                print_info("Noted. Moderate interest recorded.")
            else:
                print_info(
                    "Network updated — relevance dampened for these papers."
                )
            save(graph)

        print()


# ── Main menu ──────────────────────────────────────────────────────────────────

def main_menu(graph: HierarchicalResearchGraph, plot: bool = True):
    options = {
        "1": "Search for new papers",
        "2": "Show graph statistics",
        "3": "Add PDF folder",
        "4": "Fetch citation data (improves fusion quality)",
        "5": "Resolve Semantic Scholar IDs for seed papers",
        "6": "Rebuild hierarchy & regenerate all graph PDFs",
        "7": "Query your research network (Reasoning Mode)",
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
        elif choice == "7":
            query_session(graph)
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
