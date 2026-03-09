#!/usr/bin/env python3
"""
ResearchBuddy CLI â€” interactive literature search session.

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


# â”€â”€ Output helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Paper display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def display_paper(idx: int, meta: PaperMeta, score: float, label: str):
    score_pct = f"{score * 100:.0f}%"
    year      = str(meta.year) if meta.year else "?"
    auth      = ", ".join(meta.authors[:2]) if meta.authors else "Unknown"
    if len(meta.authors) > 2:
        auth += " et al."
    # Peer-review indicator
    pr = getattr(meta, "is_peer_reviewed", None)
    pr_tag = ""
    if pr is True:
        pr_tag = " [PR]"
    elif pr is False:
        pr_tag = " [preprint]"
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
        console.print(f"      [cyan]{auth}[/]  ({year}{pr_tag})  match={score_pct}")
        console.print(f"[dim]{snippet}[/]")
        if meta.url:
            console.print(f"      [blue underline]{meta.url}[/]")
    else:
        tag = " [EXPLORE]" if label == "explore" else ""
        print(f"\n  [{idx}] {meta.title[:90]}{tag}")
        print(f"      {auth}  ({year}{pr_tag})  match={score_pct}")
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


# â”€â”€ Rating workflow â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            print_warn(f"      Low relevance â€” will be used as negative example.")

    if rated_any:
        print_info("\n[graph] Rebuilding hierarchy with new ratings ...")
        graph.rebuild_hierarchy()


# â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # Edge confidence + publication breakdown
    n_pr = stats.get("peer_reviewed", 0)
    n_pp = stats.get("preprints", 0)
    n_uk = stats["total_papers"] - n_pr - n_pp
    if n_pr or n_pp:
        print_info(f"\n  Publication status: {n_pr} peer-reviewed, {n_pp} preprints, {n_uk} unknown")

    n_xval = stats.get("multi_source_refs", 0)
    if n_xval:
        print_info(f"  Cross-validated citations: {n_xval} papers verified by 2+ sources")

    n_anom = stats.get("edge_anomalies", 0)
    if n_anom:
        print_warn(f"  Edge anomalies: {n_anom} (use option 10 to audit)")

    print_info(f"\n  Active parameters:")
    print_info(f"    alpha (semantic weight) = {graph.alpha}")
    print_info(f"    hierarchy levels        = auto-detected ({stats.get('hierarchy_levels', 0)})")
    print_info(f"    exploration ratio       = {cfg.EXPLORATION_RATIO}")
    print_info(f"    similarity threshold    = {cfg.SIMILARITY_THRESHOLD}")


# â”€â”€ Search session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def search_session(graph: HierarchicalResearchGraph, plot: bool = True):
    if graph.context_vector() is None:
        print_warn("No context vector yet. Add seed PDFs (option 3) or rate some papers first.")
        return

    # Ask for research intent (powers HyDE + query expansion + reranking)
    print_info("\nDescribe what you're looking for (research intent), or Enter to skip:")
    query_raw = ask("Research intent", "")
    query = query_raw.strip() if query_raw.strip() else None

    print_info("\nOptional: extra search keywords (comma-separated), or Enter to skip:")
    raw   = ask("Keywords", "")
    extra = [kw.strip() for kw in raw.split(",") if kw.strip()] if raw.strip() else []

    print()
    llm_note = " with LLM enhancements" if query and cfg.LLM_ENABLED else ""
    print_info(f"Searching{llm_note} ...  (may take 20-40 s)\n")
    candidates, hyde_embedding = find_candidates(
        graph, extra_keywords=extra, query=query
    )
    if not candidates:
        print_warn("No results returned. Check your internet connection.")
        return

    score_note = " + HyDE" if hyde_embedding is not None else ""
    print_info(f"Ranking candidates (fused semantic + citation{score_note} scores) ...")
    results = graph.rank_candidates(
        candidates, n=cfg.N_RECOMMENDATIONS,
        exploration_ratio=cfg.EXPLORATION_RATIO,
        hyde_embedding=hyde_embedding,
    )
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


# â”€â”€ Query / Reasoning mode ("Prefrontal Cortex") â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Relevant papers (with centrality / role) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Research themes (cluster profiles) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Research lineages (citation / semantic paths) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Connections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Bridge papers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Frontier papers (relevant but underconnected) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Temporal narrative â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if result.temporal_narrative:
        print()
        print_info(f"  Timeline: {result.temporal_narrative}")

    # â”€â”€ Coverage note â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if result.gap_note:
        print()
        print_warn(result.gap_note)

    print(f"\n{DIVIDER}")


def query_session(graph: HierarchicalResearchGraph):
    """Interactive reasoning loop â€” the 'prefrontal cortex'."""
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

        # â”€â”€ Feedback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                    "Network updated â€” edges strengthened between relevant "
                    "papers. Future results will lean this way."
                )
            elif rating >= 4:
                print_info("Noted. Moderate interest recorded.")
            else:
                print_info(
                    "Network updated â€” relevance dampened for these papers."
                )
            save(graph)

        print()


# â”€â”€ Creative Mode ("Creative Cortex") â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def creative_session(graph: HierarchicalResearchGraph):
    """
    Interactive argumentation loop â€” the 'Creative Cortex'.

    Generates argument paragraphs synthesising the literature and learns
    from user correctness / usefulness ratings via a StyleProfile.
    """
    if not graph.all_papers():
        print_warn("No papers in your graph yet. Add PDFs (option 3) first.")
        return
    if len(graph.all_papers()) < 3:
        print_warn("At least 3 papers are needed for argumentation. Add more PDFs first.")
        return

    from researchbuddy.core.arguer import Arguer, ArgumentInteraction

    reasoner = Reasoner(top_k=cfg.QUERY_TOP_K)
    arguer   = Arguer()

    print_header("Creative Mode â€” Argumentation Engine")
    print_info(
        "Ask a research question; the system will generate argument paragraphs\n"
        "that synthesise your literature using citation relationships.\n"
        "Rate each paragraph on Correctness and Usefulness (1-10) to help the\n"
        "engine learn which argument styles work best for your research.\n"
        "Type 'q' to return to the menu."
    )

    while True:
        print()
        raw = ask("Your query", "")
        if not raw or raw.strip().lower() in ("q", "quit", "exit"):
            break

        print_info("\nAnalysing research landscape ...")
        result = reasoner.reason(raw.strip(), graph)

        if not result.relevant_papers:
            print_warn("No relevant papers found for this query.")
            continue

        style_profile = graph.get_style_profile()
        paragraphs    = arguer.generate(
            raw.strip(), result, graph, style_profile, n=cfg.ARGUER_TOP_PARAGRAPHS
        )

        if not paragraphs:
            print_warn("Could not generate arguments (need more papers / connections).")
            continue

        rated_any = False
        for i, para in enumerate(paragraphs, 1):
            print()
            if HAS_RICH:
                console.rule(
                    f"[bold green]Argument {i}/{len(paragraphs)}  Â·  "
                    f"{para.arg_type_label}[/]"
                )
                from rich.panel import Panel as _Panel
                console.print(_Panel(para.text, border_style="green", padding=(1, 2)))
                console.print(f"[dim]  {para.explanation}[/]")
                if para.paper_refs:
                    console.print(
                        f"[dim]  Based on: {', '.join(para.paper_refs[:4])}[/]"
                    )
            else:
                print(f"\n{'â”€'*72}")
                print(f"  Argument {i}/{len(paragraphs)}  Â·  {para.arg_type_label}")
                print(f"{'â”€'*72}")
                print(f"\n  {para.text}\n")
                print(f"  [{para.explanation}]")
                if para.paper_refs:
                    print(f"  Based on: {', '.join(para.paper_refs[:4])}")

            # â”€â”€ Dual rating â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            print()
            raw_c = ask("  Correctness (1-10, 0=skip)", "0")
            raw_u = ask("  Usefulness  (1-10, 0=skip)", "0")

            try:
                correctness = float(raw_c)
            except ValueError:
                correctness = 0.0
            try:
                usefulness = float(raw_u)
            except ValueError:
                usefulness = 0.0

            if 1 <= correctness <= 10 and 1 <= usefulness <= 10:
                interaction = ArgumentInteraction(
                    argument_type = para.arg_type,
                    argument_text = para.text,
                    paper_ids     = para.paper_ids,
                    query         = raw.strip(),
                    correctness   = correctness,
                    usefulness    = usefulness,
                )
                graph.apply_argument_feedback(interaction)
                rated_any = True

                combined = (correctness + usefulness) / 2
                if combined >= 7:
                    print_success(
                        "  Excellent! Style profile updated â€” "
                        f"{para.arg_type} arguments boosted."
                    )
                elif combined >= 4:
                    print_info("  Noted. Moderate preference recorded.")
                else:
                    print_warn(
                        f"  Low rating recorded â€” {para.arg_type} arguments "
                        "will be deprioritised."
                    )

            elif raw_c.lower() in ("q", "quit"):
                break

        if rated_any:
            save(graph)

        # â”€â”€ Show current style preferences after a session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sp = graph.get_style_profile()
        if sp.total_interactions >= 3:
            print()
            print_info(
                f"  Style profile ({sp.total_interactions} interactions): "
                f"avg correctness={sp.avg_correctness:.1f}  "
                f"avg usefulness={sp.avg_usefulness:.1f}"
            )
            top_type = max(sp.type_weights, key=sp.type_weights.get)
            print_info(f"  Best-performing type so far: {top_type}")


# â”€â”€ Edge Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def audit_edges(graph: HierarchicalResearchGraph):
    """Show low-confidence edges and structural anomalies for user review."""
    print_header("Edge Audit â€” Graph Reliability")

    # â”€â”€ 1. Low-confidence citation edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    low_conf_edges = []
    for u, v, d in graph.G_citation.edges(data=True):
        conf = d.get("edge_confidence", 1.0)
        if conf < 0.5:
            low_conf_edges.append((u, v, conf, d.get("etype", "?")))

    if low_conf_edges:
        print_warn(f"\n  Low-confidence edges ({len(low_conf_edges)}):")
        for u, v, conf, etype in sorted(low_conf_edges, key=lambda x: x[2])[:15]:
            p_u = graph.get_paper(u)
            p_v = graph.get_paper(v)
            t_u = (p_u.title[:35] + "...") if p_u else u[:12]
            t_v = (p_v.title[:35] + "...") if p_v else v[:12]
            print(f"    {t_u}  -->  {t_v}")
            print(f"      confidence={conf:.2f}  type={etype}")
    else:
        print_success("  No low-confidence edges found.")

    # â”€â”€ 2. Temporal anomalies â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    anomalies = getattr(graph, "_edge_anomalies", [])
    if anomalies:
        print()
        print_warn(f"  Structural anomalies ({len(anomalies)}):")
        for src, tgt, reason, penalty in anomalies[:15]:
            p_s = graph.get_paper(src)
            p_t = graph.get_paper(tgt)
            y_s = p_s.year if p_s else "?"
            y_t = p_t.year if p_t else "?"
            t_s = (p_s.title[:30] + "...") if p_s else src[:12]
            t_t = (p_t.title[:30] + "...") if p_t else tgt[:12]
            print(f"    {t_s} ({y_s}) --> {t_t} ({y_t})")
            print(f"      reason={reason}  penalty={penalty:.2f}")
    else:
        print_success("  No structural anomalies detected.")

    # â”€â”€ 3. Publication breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_pr = sum(1 for m in graph.all_papers()
               if getattr(m, "is_peer_reviewed", None) is True)
    n_pp = sum(1 for m in graph.all_papers()
               if getattr(m, "is_peer_reviewed", None) is False)
    n_uk = len(graph.all_papers()) - n_pr - n_pp
    print()
    print_info(f"  Publication status: {n_pr} peer-reviewed, "
               f"{n_pp} preprints, {n_uk} unknown")

    # â”€â”€ 4. Cross-validation coverage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_xval = sum(1 for v in graph._ref_sources.values() if len(v) >= 2)
    n_total = len(graph._ref_sources)
    if n_total:
        print_info(f"  Citation cross-validation: {n_xval}/{n_total} papers "
                   f"verified by 2+ independent sources")
    print()


# â”€â”€ LLM Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _show_llm_status_banner():
    """Print a one-line LLM status at startup."""
    if not cfg.LLM_ENABLED:
        print_info("  LLM: disabled (--no-llm)")
        return
    try:
        from researchbuddy.core.llm import get_llm
        client = get_llm()
        st = client.status()
        if st.available:
            gpu_str = f" on {st.gpu_name}" if st.gpu_name else ""
            vram_str = f" ({st.gpu_vram_mb} MB VRAM)" if st.gpu_vram_mb else ""
            print_success(f"  LLM: {st.model_name}{gpu_str}{vram_str} -- ready")
        else:
            print_warn(f"  LLM: {st.error}")
            print_info("  (ResearchBuddy works without LLM â€” using template fallback)")
    except Exception as e:
        print_warn(f"  LLM: check failed ({e})")


def show_llm_status():
    """Detailed LLM status display (menu option 9)."""
    print_header("LLM Status & Setup")

    if not cfg.LLM_ENABLED:
        print_warn("LLM is disabled. Remove --no-llm flag to enable.")
        return

    try:
        from researchbuddy.core.llm import get_llm, detect_gpu
        client = get_llm()
        st = client.status()

        # GPU info
        gpu = detect_gpu()
        if gpu.get("available"):
            print_success(f"  GPU: {gpu['name']}  ({gpu['vram_mb']} MB VRAM)")
        else:
            gpu_err = gpu.get("error")
            if gpu_err:
                print_info(f"  GPU: {gpu_err} (using CPU)")
            else:
                print_info("  GPU: No CUDA GPU detected (using CPU)")

        # Ollama status
        if st.available:
            print_success(f"  Ollama: connected  (model: {st.model_name})")
        else:
            print_warn(f"  Ollama: {st.error}")

        # Feature toggles
        print()
        print_info("  Feature flags:")
        print_info(f"    LLM argumentation : {'ON' if cfg.LLM_ENABLED else 'OFF'}")
        print_info(f"    HyDE search       : {'ON' if cfg.HYDE_ENABLED else 'OFF'}")
        print_info(f"    Query expansion   : {'ON' if cfg.LLM_QUERY_EXPANSION else 'OFF'}")
        print_info(f"    LLM reranking     : {'ON' if cfg.LLM_RERANK_ENABLED else 'OFF'}")
        print_info(f"    Deterministic     : {'ON' if getattr(cfg, 'DETERMINISTIC_MODE', False) else 'OFF'}")
        print_info(f"    Search cache      : {'ON' if getattr(cfg, 'SEARCH_CACHE_ENABLED', False) else 'OFF'}")

        if not st.available:
            print()
            print_info("  Setup instructions:")
            print_info("    1. Install Ollama: https://ollama.ai")
            print_info("    2. Start server:   ollama serve")
            print_info(f"    3. Pull model:     ollama pull {cfg.LLM_MODEL}")
            print()
            print_info("  Recommended models by GPU VRAM:")
            print_info("    4 GB  (RTX 2050/3050):  ollama pull phi3.5")
            print_info("    8 GB  (RTX 3050/3060):  ollama pull mistral")
            print_info("   16 GB+ (RTX 3090/4090):  ollama pull llama3.1")
    except Exception as e:
        print_warn(f"  Error checking LLM status: {e}")


# â”€â”€ Main menu â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main_menu(graph: HierarchicalResearchGraph, plot: bool = True):
    options = {
        "1": "Search for new papers",
        "2": "Show graph statistics",
        "3": "Add PDF folder",
        "4": "Fetch citation data (improves fusion quality)",
        "5": "Resolve Semantic Scholar IDs for seed papers",
        "6": "Rebuild hierarchy & regenerate all graph PDFs",
        "7": "Query your research network (Reasoning Mode)",
        "8": "Creative Mode â€” generate & rate argument paragraphs",
        "9": "LLM status & setup",
        "10": "Audit graph edges (low-confidence & anomalies)",
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
        elif choice == "8":
            creative_session(graph)
        elif choice == "9":
            show_llm_status()
        elif choice == "10":
            audit_edges(graph)
        else:
            print_warn("Unknown option.")


# â”€â”€ CLI argument parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ResearchBuddy â€“ hierarchical graph-based literature search assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf",    metavar="FOLDER", help="Import PDFs from folder on startup")
    p.add_argument("--reset",  action="store_true", help="Clear saved state and start fresh")

    # â”€â”€ Tunable parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ LLM options â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    llm_g = p.add_argument_group("LLM options (local Ollama)")
    llm_g.add_argument("--llm-model", type=str, default=None,
                       metavar="NAME",
                       help=f"Ollama model name. Default: {cfg.LLM_MODEL}")
    llm_g.add_argument("--no-llm", action="store_true",
                       help="Disable all LLM features (pure graph-based mode)")
    return p


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    args = _build_parser().parse_args()

    # Apply CLI overrides to config module (affects all imports)
    if args.alpha                is not None: cfg.FUSION_ALPHA         = args.alpha
    if args.exploration_ratio    is not None: cfg.EXPLORATION_RATIO    = args.exploration_ratio
    if args.similarity_threshold is not None: cfg.SIMILARITY_THRESHOLD = args.similarity_threshold
    if args.n_recommendations    is not None: cfg.N_RECOMMENDATIONS    = args.n_recommendations
    if args.no_plot:                           cfg.SAVE_GRAPH_PDF       = False
    if args.no_llm:
        cfg.LLM_ENABLED        = False
        cfg.HYDE_ENABLED       = False
        cfg.LLM_QUERY_EXPANSION= False
        cfg.LLM_RERANK_ENABLED = False
    if args.llm_model is not None:
        cfg.LLM_MODEL = args.llm_model
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

    # â”€â”€ LLM status at startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _show_llm_status_banner()

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
