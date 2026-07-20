#!/usr/bin/env python3
"""
ResearchBuddy CLI - interactive literature search session.

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
import logging
import os
import textwrap
import time

logger = logging.getLogger(__name__)

import researchbuddy.config as cfg
from researchbuddy.core.graph_model   import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core.state_manager import save, load, import_pdf_folder, resolve_seed_s2_ids
from researchbuddy.core.searcher      import find_candidates
from researchbuddy.core.reasoner      import Reasoner, QueryResult
from researchbuddy.core import services as svc
from researchbuddy.core import audit

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

def _ingest_pdf_for_rated_paper(
    graph: HierarchicalResearchGraph,
    meta: PaperMeta,
    pdf_path: str,
) -> bool:
    """
    Upgrade a freshly-rated paper from "abstract embedding only" to a full
    GROBID-parsed node — section embeddings, parsed local references,
    figures, tables, equations.

    The paper_id stays the same so the rating travels with the upgrade.
    Returns True on a successful ingest.
    """
    from pathlib import Path
    from researchbuddy.core.ingest import ingest_pdf_into_meta, IngestError

    p = Path(pdf_path).expanduser().resolve()
    print_info(f"  Extracting {p.name} (GROBID -> pdfplumber fallback) ...")
    try:
        info = ingest_pdf_into_meta(graph, meta, p)
    except IngestError as e:
        print_warn(f"  {e}")
        return False

    print_success(
        f"  Ingested via {info['parser']}: "
        f"{info['n_sections']} section embeddings, "
        f"{info['n_refs']} parsed references."
    )
    audit.log_event(
        "fulltext", paper_id=meta.paper_id, title=meta.title[:200],
        doi=meta.doi, provider="user-supplied", license="", url="",
    )
    return True


def rating_session(graph: HierarchicalResearchGraph,
                   results: list[tuple[PaperMeta, float, str]]):
    """
    One-at-a-time review loop. After rating each paper, the user is asked
    whether they have the PDF locally. If yes, the paper is upgraded from
    abstract-only to a full GROBID-parsed graph node.

    This is what makes the graph actually grow with content (not just
    titles + abstracts) — a rating without a PDF still counts, but a
    rating WITH a PDF adds dimensionality to the recommender.
    """
    if not results:
        return

    print_header("Review Suggestions")
    print_info(
        "For each suggestion: rate 1-10  (s=skip, q=stop)\n"
        "  After a rating you'll be asked for the PDF. Providing the PDF\n"
        "  ingests its sections + references so the graph actually grows."
    )

    rated_any = False
    upgraded_any = False
    for i, (meta, score, label) in enumerate(results, start=1):
        print()
        print_header(f"Suggestion {i} of {len(results)}")
        display_paper(i, meta, score, label)
        if HAS_RICH:
            console.print()

        raw = ask("Rating (1-10, s=skip, q=stop)", "s").strip().lower()
        if raw == "q":
            break
        if raw in ("", "s", "0"):
            audit.log_event("screen", paper_id=meta.paper_id,
                            title=meta.title[:200], doi=meta.doi,
                            rating=None, decision="skipped")
            continue
        try:
            rating = int(raw)
        except ValueError:
            print_warn("Not a number — skipping.")
            continue
        rating = max(1, min(10, rating))
        audit.log_event("screen", paper_id=meta.paper_id,
                        title=meta.title[:200], doi=meta.doi,
                        rating=rating,
                        decision=audit.screen_decision(float(rating)))

        meta.times_shown += 1
        meta.last_shown   = time.time()
        # Ensure the paper exists in the graph (with abstract-only emb).
        # add_or_get resolves a title/DOI collision to the resident id so
        # rate_paper never hits KeyError on a candidate that duplicates an
        # existing paper under a different id.
        if graph.resolve_paper_id(meta) is None:
            graph.embed_abstract(meta)
        pid = graph.add_or_get(meta, meta.embedding)
        graph.rate_paper(pid, float(rating))
        rated_any = True

        if rating >= 7:
            print_success(f"  Rated {rating}/10 — strong positive example.")
        elif rating >= 4:
            print_info(f"  Rated {rating}/10 — moderate signal.")
        else:
            print_warn(f"  Rated {rating}/10 — used as negative example.")

        # ── Offer to ingest the PDF ───────────────────────────────────────
        pdf_path = ask(
            "  PDF path to ingest as full graph node (Enter to skip)", "",
        ).strip()
        if pdf_path:
            if _ingest_pdf_for_rated_paper(graph, meta, pdf_path):
                upgraded_any = True

    # ── End-of-session housekeeping ───────────────────────────────────────
    if rated_any:
        print()
        if upgraded_any:
            print_info("[graph] Rebuilding hierarchy with new content ...")
        else:
            print_info("[graph] Rebuilding hierarchy with new ratings ...")
        graph.rebuild_hierarchy()
        if graph.learn_signal_weights():
            print_success("[graph] Signal weights updated from your rating history.")


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

    health = stats.get("reliability_health", "good")
    mean_conf = stats.get("mean_edge_confidence", 1.0)
    low_conf_ratio = stats.get("low_conf_edge_ratio", 0.0)
    drift = stats.get("confidence_drift", None)
    drift_txt = f"{drift:+.3f}" if isinstance(drift, (int, float)) else "n/a"
    if health == "risk":
        print_warn(f"  Reliability: RISK (mean_conf={mean_conf:.2f}, low_conf={low_conf_ratio:.1%}, drift={drift_txt})")
    elif health == "watch":
        print_warn(f"  Reliability: WATCH (mean_conf={mean_conf:.2f}, low_conf={low_conf_ratio:.1%}, drift={drift_txt})")
    else:
        print_success(f"  Reliability: GOOD (mean_conf={mean_conf:.2f}, low_conf={low_conf_ratio:.1%}, drift={drift_txt})")

    print_info(f"\n  Active parameters:")
    print_info(f"    alpha (semantic weight) = {graph.alpha}")
    print_info(f"    hierarchy levels        = auto-detected ({stats.get('hierarchy_levels', 0)})")
    print_info(f"    exploration ratio       = {cfg.EXPLORATION_RATIO}")
    print_info(f"    similarity threshold    = {cfg.SIMILARITY_THRESHOLD}")

    # Scoring mode indicator
    n_papers = stats["total_papers"]
    if n_papers < cfg.COLD_START_THRESHOLD:
        print_warn(f"\n  Scoring: COLD-START mode ({n_papers}/{cfg.COLD_START_THRESHOLD} papers)")
        print_info("    Using simplified scoring (context similarity + per-paper matching).")
        print_info("    Add more papers for full multi-signal fusion.")
    elif graph._learned_signal_weights is not None:
        w = graph._learned_signal_weights
        base_labels = ["context", "niche", "area", "citation", "snf", "pub_qual", "recency"]
        # Per-section labels (only present in v2.3.0+ learned weights)
        sec_labels = [f"sec[{s[:4]}]" for s in cfg.SCORED_SECTION_TYPES]
        extra_labels = ["ppr", "impact", "equation"]
        labels = (base_labels + sec_labels + extra_labels)[: len(w)]
        w_str = "  ".join(f"{l}={v:.1f}" for l, v in zip(labels, w))
        print_success(f"\n  Scoring: LEARNED weights from your ratings")
        print_info(f"    {w_str}")
    else:
        print_info(f"\n  Scoring: default weights (need {cfg.WEIGHT_LEARNING_MIN_RATINGS}+ rated papers to learn)")

    # LLM status in stats
    if cfg.LLM_ENABLED and not _check_llm_available():
        print_warn("\n  LLM: UNAVAILABLE -- search is running without HyDE/expansion/reranking")


# â”€â”€ Search session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _pick_focus_papers(graph: HierarchicalResearchGraph) -> list[str]:
    """
    Focus mode: let the user anchor this discovery run on a few papers from
    their OWN library. The context vector and the Personalized-PageRank
    restart distribution then centre on that subset — multi-seed exploration
    around chosen anchors instead of the whole graph.
    """
    raw = ask(
        "Focus on specific papers? Keyword to search YOUR library "
        "(Enter = whole graph)", "",
    ).strip().lower()
    if not raw:
        return []
    hits = [m for m in graph.all_papers()
            if m.kind == "paper" and raw in m.title.lower()][:10]
    if not hits:
        print_warn(f"No papers in your library match '{raw}'.")
        return []
    for i, m in enumerate(hits, 1):
        r = f"  rated {m.user_rating:.0f}/10" if m.user_rating else ""
        print_info(f"  [{i}] {m.title[:70]} ({m.year or '?'}){r}")
    sel = ask("Anchor on which? (numbers, comma-separated; Enter = all shown)",
              "").strip()
    if not sel:
        chosen = hits
    else:
        idxs = []
        for tok in sel.split(","):
            try:
                idxs.append(int(tok.strip()) - 1)
            except ValueError:
                continue
        chosen = [hits[i] for i in idxs if 0 <= i < len(hits)]
    if chosen:
        print_success(f"Focus mode: discovery centred on {len(chosen)} "
                      "anchor paper(s).")
    return [m.paper_id for m in chosen]


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

    # Focus mode (multi-seed anchored discovery)
    focus_ids = _pick_focus_papers(graph)

    print()
    if query and cfg.LLM_ENABLED and not _check_llm_available():
        print_warn("LLM is enabled but unreachable -- searching without HyDE/expansion.")
        print_info("Start Ollama for better results: ollama serve\n")
    llm_note = " with LLM enhancements" if query and _check_llm_available() else ""
    print_info(f"Searching{llm_note} ...  (may take 20-40 s)\n")
    candidates, hyde_embedding = find_candidates(
        graph, extra_keywords=extra, query=query
    )
    audit.log_event("search", query=query or "", keywords=extra,
                    n_results=len(candidates))
    if not candidates:
        print_warn("No results returned. Check your internet connection.")
        return

    score_note = " + HyDE" if hyde_embedding is not None else ""
    n_papers = len(graph.all_papers())
    if n_papers < cfg.COLD_START_THRESHOLD:
        print_info(f"Cold-start mode ({n_papers}/{cfg.COLD_START_THRESHOLD} papers) -- using simplified scoring.")
        print_info("Add more papers & ratings for full fusion scoring.")
    print_info(f"Ranking candidates (fused semantic + citation{score_note} scores) ...")
    results = graph.rank_candidates(
        candidates, n=cfg.N_RECOMMENDATIONS,
        exploration_ratio=cfg.EXPLORATION_RATIO,
        hyde_embedding=hyde_embedding,
        focus_ids=focus_ids or None,
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
    """Interactive reasoning loop - the 'prefrontal cortex'."""
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
                    "Network updated - edges strengthened between relevant "
                    "papers. Future results will lean this way."
                )
            elif rating >= 4:
                print_info("Noted. Moderate interest recorded.")
            else:
                print_info(
                    "Network updated - relevance dampened for these papers."
                )
            save(graph)

        print()


# â”€â”€ Creative Mode ("Creative Cortex") â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def creative_session(graph: HierarchicalResearchGraph):
    """
    Interactive argumentation loop - the 'Creative Cortex'.

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

    print_header("Creative Mode - Argumentation Engine")
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
                        "  Excellent! Style profile updated - "
                        f"{para.arg_type} arguments boosted."
                    )
                elif combined >= 4:
                    print_info("  Noted. Moderate preference recorded.")
                else:
                    print_warn(
                        f"  Low rating recorded - {para.arg_type} arguments "
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
    print_header("Edge Audit - Graph Reliability")

    # â”€â”€ 1. Low-confidence citation edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    low_conf_edges = []
    for u, v, d in graph._backend.edges_data("citation"):
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


def quality_report_session(graph: HierarchicalResearchGraph):
    """Show model-value and reliability diagnostics in one place."""
    print_header("Quality & Reliability Report")

    rel = graph.reliability_report()
    health = rel.get("health", "good").upper()
    print_info(
        f"  Reliability: {health}  "
        f"(mean_conf={rel.get('mean_edge_confidence', 1.0):.2f}, "
        f"low_conf={rel.get('low_conf_ratio', 0.0):.1%}, "
        f"anomalies={rel.get('anomalies', 0)})"
    )
    drift = rel.get("confidence_drift", None)
    if isinstance(drift, (int, float)):
        print_info(f"  Confidence drift (vs recent): {drift:+.3f}")
    warns = rel.get("warnings", []) or []
    if warns:
        for w in warns:
            print_warn(f"  - {w}")

    print()
    q = graph.quality_report()
    if not q.get("ready"):
        print_warn(f"  Evaluation not ready: {q.get('note', 'Not enough data.')}")
        print_info(
            f"  Rated={q.get('rated', 0)}, "
            f"positives={q.get('positives', 0)}, negatives={q.get('negatives', 0)}"
        )
        print_info("  Tip: rate more papers across both relevant and irrelevant results.")
        print()
        return

    k = q.get("k", 0)
    b = q.get("baseline", {})
    g = q.get("graph", {})
    d = q.get("delta", {})

    print_info(
        f"  Rated set: {q.get('rated', 0)} "
        f"(pos={q.get('positives', 0)}, neg={q.get('negatives', 0)})"
    )
    print_info(f"  Metrics at k={k} (baseline semantic vs graph):")
    print_info(
        f"    AUC                 : {b.get('auc')} -> {g.get('auc')} "
        f"(delta={d.get('auc')})"
    )
    print_info(
        f"    Precision@{k:<2d}       : {b.get('precision_at_k')} -> {g.get('precision_at_k')} "
        f"(delta={d.get('precision_at_k')})"
    )
    print_info(
        f"    NDCG@{k:<2d}            : {b.get('ndcg_at_k')} -> {g.get('ndcg_at_k')} "
        f"(delta={d.get('ndcg_at_k')})"
    )
    print_info(
        f"    Rating correlation  : {b.get('rating_corr')} -> {g.get('rating_corr')} "
        f"(delta={d.get('rating_corr')})"
    )
    print()

# â”€â”€ LLM Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_LLM_AVAILABLE: bool | None = None  # session-level cache


def _check_llm_available() -> bool:
    """Check if LLM is actually reachable. Uses session cache."""
    global _LLM_AVAILABLE
    if _LLM_AVAILABLE is not None:
        return _LLM_AVAILABLE
    if not cfg.LLM_ENABLED:
        _LLM_AVAILABLE = False
        return False
    try:
        from researchbuddy.core.llm import get_llm
        client = get_llm()
        _LLM_AVAILABLE = client.is_available()
    except Exception:
        _LLM_AVAILABLE = False
    return _LLM_AVAILABLE


def _show_llm_status_banner():
    """Print LLM status at startup with clear degradation warnings."""
    global _LLM_AVAILABLE
    if not cfg.LLM_ENABLED:
        print_info("  LLM: disabled (--no-llm)")
        _LLM_AVAILABLE = False
        return
    try:
        from researchbuddy.core.llm import get_llm
        client = get_llm()
        st = client.status()
        if st.available:
            gpu_str = f" on {st.gpu_name}" if st.gpu_name else ""
            vram_str = f" ({st.gpu_vram_mb} MB VRAM)" if st.gpu_vram_mb else ""
            print_success(f"  LLM: {st.model_name}{gpu_str}{vram_str} -- ready")
            _LLM_AVAILABLE = True
        else:
            _LLM_AVAILABLE = False
            print_warn(f"  LLM: UNAVAILABLE -- {st.error}")
            print_warn("  Degraded mode: HyDE, query expansion, LLM reranking, and")
            print_warn("  argumentation are all disabled. Search quality is reduced.")
            print_info("  To fix: ollama serve && ollama pull " + cfg.LLM_MODEL)
    except Exception as e:
        _LLM_AVAILABLE = False
        print_warn(f"  LLM: UNAVAILABLE -- {e}")
        print_warn("  Running in degraded mode (no HyDE/expansion/reranking).")
        print_info("  To fix: ollama serve && ollama pull " + cfg.LLM_MODEL)


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


# ── Neo4j Browser launch ──────────────────────────────────────────────────────

def browse_in_neo4j(graph: HierarchicalResearchGraph) -> None:
    """
    Open Neo4j Browser at a starter Cypher query and tell the user how to
    apply the bundled stylesheet so the graph is actually readable.
    """
    import webbrowser
    from pathlib import Path

    backend_name = getattr(graph._backend, "backend_name", "?")
    if backend_name != "Neo4j":
        print_warn(
            "You're currently using the NetworkX backend. To browse in Neo4j, "
            "first start Neo4j (option from the startup prompt, or "
            "`docker start researchbuddy-neo4j`) and re-launch ResearchBuddy."
        )
        return

    # A starter query that returns a meaningful subgraph
    starter_cypher = (
        "MATCH (p:Paper)-[r:SEM_SIMILARITY|CIT_CITATION|CAUSAL_INFLUENCE]-(q:Paper) "
        "WHERE coalesce(r.weight, 0) > 0.4 "
        "RETURN p, r, q LIMIT 200"
    )
    browser_url = f"http://localhost:7474/browser/?cmd=play&arg={starter_cypher}"

    # Locate the bundled .grass file
    grass_path = Path(__file__).parent / "assets" / "researchbuddy.grass"

    print_header("Neo4j Browser")
    print_info("Opening http://localhost:7474 in your default browser ...")
    try:
        webbrowser.open(browser_url)
    except Exception:
        print_warn("Could not auto-open the browser. Please open this URL manually:")
        print_info(f"  {browser_url}")

    print()
    print_info("To make the graph readable (recommended, one-time setup):")
    print_info("  1. In Neo4j Browser, click the cog icon (bottom-left)")
    print_info("  2. Open the 'Document Settings' panel")
    print_info("  3. Drag the file below onto 'Graph Stylesheet':")
    if grass_path.exists():
        print_success(f"     {grass_path}")
    else:
        print_warn("     (stylesheet not bundled - re-install with `pip install -e .`)")

    print()
    print_info("Useful starter queries (paste into the Browser):")
    print_info("  Show top semantic neighbours of any paper:")
    print_info('    MATCH (p:Paper)-[r:SEM_SIMILARITY]->(q:Paper)')
    print_info("    WHERE p.title CONTAINS 'YOUR_KEYWORD'")
    print_info("    RETURN p, r, q ORDER BY r.weight DESC LIMIT 25")
    print()
    print_info("  Show citation chains (multi-hop):")
    print_info('    MATCH path = (p:Paper)-[:CIT_CITATION*1..3]->(q:Paper)')
    print_info("    RETURN path LIMIT 50")
    print()
    print_info("  Show clusters and their members:")
    print_info("    MATCH (c:Cluster)-[:SEM_MEMBER]-(p:Paper)")
    print_info("    RETURN c, p LIMIT 100")


# ── User-content upload ──────────────────────────────────────────────────────

def upload_thought_session(graph: HierarchicalResearchGraph) -> None:
    """
    Let the user feed their own writing into the graph as a 'thought'
    node. Anchors the user-context vector strongly, so subsequent
    recommendations align with the user's thinking — not just with what
    they've already read.
    """
    print_header("Upload Your Own Writing")
    print_info(
        "Add your essays, notes, draft sections, research questions, or\n"
        "outlines. Each one becomes a strongly-weighted node and re-shapes\n"
        "future suggestions toward how YOU think.\n"
    )
    print_info("  [1] Paste text inline")
    print_info("  [2] Read from a .txt / .md file")
    print_info("  [3] Read from a PDF (full GROBID extraction, tagged as 'draft')")
    print_info("  [b] Back")

    sub = ask("Choose", "1").strip().lower()
    if sub == "b":
        return

    kinds = ("essay", "note", "question", "outline", "draft")
    kind = ask(
        f"Kind ({'/'.join(kinds)})", "essay",
    ).strip().lower()
    if kind not in kinds:
        print_warn(f"Unknown kind {kind!r} — defaulting to 'essay'.")
        kind = "essay"

    if sub == "1":
        title = ask("Title", "untitled thought").strip()
        print_info(
            "Paste / type your text. End input with a single line containing 'END'."
        )
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)
        meta = graph.add_thought_from_text(text, title=title, kind=kind)
    elif sub == "2":
        path = ask("Path to .txt / .md file", "").strip()
        if not path:
            print_info("Cancelled.")
            return
        meta = graph.add_thought_from_file(path, kind=kind)
    elif sub == "3":
        from pathlib import Path
        path = ask("Path to your draft PDF", "").strip()
        if not path:
            print_info("Cancelled.")
            return
        # Reuse the rated-paper ingest helper, then retag as draft thought.
        # Build a fresh meta so the kind/source land correctly.
        from researchbuddy.core.pdf_processor import extract_from_pdf
        p = Path(path).expanduser().resolve()
        if not p.exists() or p.suffix.lower() != ".pdf":
            print_warn(f"Not a PDF: {p}")
            return
        ep = extract_from_pdf(p)
        if ep is None:
            print_warn("Extraction failed.")
            return
        # Concatenate all section text and run through add_thought_from_text
        # so it gets section_embeddings *and* the strong implicit weight.
        joined = "\n\n".join(s.text for s in ep.sections if s.text) \
                 or ep.full_text or ep.abstract
        # Build explicit section_text_map from GROBID's classification
        section_text_map = {}
        for s in ep.sections:
            if s.section_type and s.section_type != "other" and s.text:
                section_text_map.setdefault(s.section_type, []).append(s.text)
        section_text_map = {
            k: "\n\n".join(v) for k, v in section_text_map.items()
        }
        meta = graph.add_thought_from_text(
            joined, title=ep.title or p.stem,
            kind="draft", section_text_map=section_text_map,
        )
        if meta is not None:
            meta.filepath = str(p)
            if ep.doi and not meta.doi:
                meta.doi = ep.doi
    else:
        print_warn("Unknown option.")
        return

    if meta is None:
        print_warn("Could not add thought (text too short or empty).")
        return

    n_secs = len(getattr(meta, "section_embeddings", {}) or {})
    print_success(
        f"Added {kind} '{meta.title[:60]}' "
        f"(paper_id={meta.paper_id}, {n_secs} section embeddings, "
        f"weight={meta.user_rating:.0f})."
    )
    print_info("Run option 6 (Rebuild hierarchy) to refresh the graph layers.")


# ── Service management (on-demand) ────────────────────────────────────────────

def print_docker_help() -> None:
    """Step-by-step Docker setup for Neo4j + GROBID (shown whenever Docker
    or a service is missing, so nobody has to hunt through the README)."""
    print_info("How to set up the optional services with Docker:")
    print_info("  1. Install Docker Desktop:  https://www.docker.com/products/docker-desktop/")
    print_info("     (Windows: enable WSL2 when the installer asks; then start Docker Desktop)")
    print_info("  2. Verify it works:         docker --version")
    print_info("  3. Start Neo4j (graph database backend + browser):")
    print_info("       docker run -d --name researchbuddy-neo4j -p 7474:7474 -p 7687:7687 \\")
    print_info("         -e NEO4J_AUTH=neo4j/researchbuddy neo4j:5-community")
    print_info("  4. Start GROBID (scientific-PDF parser — much richer imports):")
    print_info("       docker run -d --name researchbuddy-grobid -p 8070:8070 lfoppiano/grobid:0.8.1")
    print_info("     First GROBID request loads its ML models (~30 s) — be patient.")
    print_info("  5. Re-launch ResearchBuddy (or use this menu) — services are")
    print_info("     auto-detected. Stop them anytime: docker stop <name>.")


def _apply_core_key_from_prefs() -> None:
    """Load a saved CORE API key (service prefs) into the live session."""
    try:
        key = (svc.load_prefs().get("core_api_key") or "").strip()
        if key and not os.environ.get("CORE_API_KEY"):
            from researchbuddy.core import core_fetcher
            core_fetcher.set_api_key(key)
            print_info("  CORE API key loaded from saved preferences.")
    except Exception as e:
        logger.debug("CORE key pref load skipped: %s", e)


def manage_services() -> None:
    """
    Interactive service control: start / stop Neo4j and GROBID, reset the
    'never' preference, show current status.

    NOTE: the menu renders even when Docker is down — several options
    (CORE key, docker help, history restore) don't need Docker at all.
    """
    print_header("Service Management")
    docker_ok = svc.docker_available()
    if not docker_ok:
        print_warn(
            "Docker is not detected — Neo4j / GROBID start/stop won't work "
            "until Docker Desktop is running (option 11 shows setup steps). "
            "The other options below still work."
        )

    while True:
        # Live status each iteration
        n_alive = docker_ok and svc._service_alive(svc.NEO4J_SPEC)
        g_alive = docker_ok and svc._service_alive(svc.GROBID_SPEC)
        prefs = svc.load_prefs()
        print()
        print_info(f"  Neo4j:  {'running' if n_alive else 'stopped'}  "
                   f"(auto-launch pref: {prefs.get('neo4j_auto_launch', 'ask')})")
        print_info(f"  GROBID: {'running' if g_alive else 'stopped'}  "
                   f"(auto-launch pref: {prefs.get('grobid_auto_launch', 'ask')})")
        print()
        print_info("  [1] Start Neo4j")
        print_info("  [2] Stop Neo4j")
        print_info("  [3] Start GROBID")
        print_info("  [4] Stop GROBID")
        print_info("  [5] Reset auto-launch preferences (will ask again next run)")
        print_info("  [6] Test Neo4j connection (bolt + auth)")
        print_info("  [7] Set Neo4j password for this session")
        print_info("  [8] Restore graph from a history snapshot")
        print_info("  [9] Compact history (delete old snapshots, keep stats log)")
        from researchbuddy.core import core_fetcher as _cf
        core_tag = "set" if _cf.has_api_key() else "not set — slower CORE access"
        print_info(f"  [10] Set CORE API key ({core_tag})")
        print_info("  [11] Docker setup instructions (Neo4j / GROBID)")
        print_info("  [b] Back to main menu")

        choice = ask("Choose", "b").strip().lower()

        if choice == "b":
            return
        if choice == "1":
            res = svc.ensure_running(svc.NEO4J_SPEC)
            if res.already_running:
                print_info("Neo4j already running.")
            elif res.started:
                print_success("Neo4j is up at http://localhost:7474")
                os.environ.setdefault("RESEARCHBUDDY_NEO4J_ENABLED", "true")
                os.environ.setdefault("RESEARCHBUDDY_NEO4J_PASSWORD", "researchbuddy")
            else:
                print_warn(f"Could not start Neo4j: {res.error}")
        elif choice == "2":
            if svc.stop_service(svc.NEO4J_SPEC):
                print_success("Neo4j stopped.")
            else:
                print_warn("Could not stop Neo4j (already stopped or container missing).")
        elif choice == "3":
            res = svc.ensure_running(svc.GROBID_SPEC)
            if res.already_running:
                print_info("GROBID already running.")
            elif res.started:
                print_success("GROBID is up at http://localhost:8070")
            else:
                print_warn(f"Could not start GROBID: {res.error}")
        elif choice == "4":
            if svc.stop_service(svc.GROBID_SPEC):
                print_success("GROBID stopped.")
            else:
                print_warn("Could not stop GROBID (already stopped or container missing).")
        elif choice == "5":
            svc.save_prefs({})
            print_success("Auto-launch preferences cleared.")
        elif choice == "6":
            password = os.environ.get("RESEARCHBUDDY_NEO4J_PASSWORD", "researchbuddy")
            print_info(f"Probing bolt://localhost:7687 as user 'neo4j' with the configured password ...")
            res = svc.probe_neo4j_bolt(password=password)
            if res.ok:
                print_success("Connection ok — Neo4j is usable as a backend.")
                print_info("Re-launch ResearchBuddy to switch the active backend to Neo4j.")
            else:
                print_warn(f"Connection failed: {res.reason}")
        elif choice == "7":
            new_pw = ask("New Neo4j password (leave blank to cancel)", "")
            if not new_pw:
                print_info("Cancelled.")
                continue
            os.environ["RESEARCHBUDDY_NEO4J_PASSWORD"] = new_pw
            os.environ.setdefault("RESEARCHBUDDY_NEO4J_ENABLED", "true")
            # Reload config so the next backend creation sees the new value
            import importlib, researchbuddy.config as _cfg_mod
            importlib.reload(_cfg_mod)
            res = svc.probe_neo4j_bolt(password=new_pw)
            if res.ok:
                print_success(
                    "Password accepted. Re-launch ResearchBuddy to use Neo4j "
                    "as the active backend."
                )
            else:
                print_warn(f"Still failing: {res.reason}")
        elif choice == "8":
            _restore_from_snapshot()
        elif choice == "9":
            _compact_history_menu()
        elif choice == "10":
            from researchbuddy.core import core_fetcher
            print_info("Free key: https://core.ac.uk/services/api "
                       "(anonymous works, but is rate-limited to ~1 req/s).")
            key = ask("CORE API key (blank to clear)", "").strip()
            core_fetcher.set_api_key(key)
            prefs = svc.load_prefs()
            if key:
                prefs["core_api_key"] = key
                print_success("CORE key applied to this session and saved.")
            else:
                prefs.pop("core_api_key", None)
                print_info("CORE key cleared.")
            svc.save_prefs(prefs)
        elif choice == "11":
            print()
            print_docker_help()
        else:
            print_warn("Unknown option.")


def _compact_history_menu() -> None:
    """
    Walk the history dir, fold every existing pickle's metrics into the
    JSONL log, then delete pickles beyond the configured retention. Print
    a clear summary of disk reclaimed.
    """
    from researchbuddy.core.state_manager import compact_history
    from researchbuddy.config import STATE_HISTORY_KEEP, HISTORY_DIR

    print_info(f"Compacting history in {HISTORY_DIR} ...")
    print_info(
        f"Each save still writes one ~1 KB line to history/evolution.jsonl. "
        f"Only the {STATE_HISTORY_KEEP} most recent full pickles are kept."
    )
    confirm = ask("Proceed? (y/n)", "y").strip().lower()
    if not confirm.startswith("y"):
        print_info("Cancelled.")
        return

    report = compact_history()
    mb_freed = report["bytes_freed"] / (1024 * 1024)
    print_success(
        f"Ingested {report['ingested']} new entries into evolution.jsonl, "
        f"deleted {report['deleted']} pickle(s), freed {mb_freed:.1f} MB."
    )
    print_info(
        f"Kept the {report['kept_pickles']} most recent pickles for recovery. "
        f"Long-term history lives in {report['log_path']}."
    )


def _restore_from_snapshot() -> None:
    """
    Walk through history snapshots in ~/.researchbuddy/history, show the user
    the most recent ones with edge counts, and let them pick one to copy
    over the canonical pickle.
    """
    import pickle
    from pathlib import Path
    from researchbuddy.core.state_manager import _total_edges
    from researchbuddy.config import HISTORY_DIR, STATE_FILE

    snaps = sorted(Path(HISTORY_DIR).glob("graph_*.pkl"), reverse=True)
    if not snaps:
        print_warn(f"No history snapshots found in {HISTORY_DIR}.")
        return

    # Show the most recent 20 with their edge counts so user picks the
    # last healthy one.
    print_info(f"Found {len(snaps)} snapshots. Most recent 20 (newest first):")
    candidates: list[tuple[int, Path, int]] = []
    for i, snap in enumerate(snaps[:20]):
        try:
            with open(snap, "rb") as f:
                g = pickle.load(f)
            edges = _total_edges(g)
            n_papers = len(getattr(g, "_papers", {}))
        except Exception as e:
            edges, n_papers = -1, -1
            print_info(f"  [{i+1:>2}]  {snap.name}  (could not read: {e})")
            continue
        flag = " <-- looks healthy" if edges > 0 else " <-- empty"
        print_info(f"  [{i+1:>2}]  {snap.name}  papers={n_papers}  edges={edges}{flag}")
        candidates.append((i + 1, snap, edges))

    if not candidates:
        print_warn("Could not read any snapshots.")
        return

    sel = ask("Pick a number to restore (or blank to cancel)", "").strip()
    if not sel:
        print_info("Cancelled.")
        return
    try:
        idx = int(sel)
    except ValueError:
        print_warn("Invalid selection.")
        return
    chosen = next((c for c in candidates if c[0] == idx), None)
    if chosen is None:
        print_warn("Selection out of range.")
        return

    _, snap_path, edges = chosen
    confirm = ask(
        f"Overwrite {STATE_FILE.name} with {snap_path.name} ({edges} edges)? (y/n)",
        "n",
    ).strip().lower()
    if not confirm.startswith("y"):
        print_info("Cancelled.")
        return

    import shutil
    backup = STATE_FILE.with_suffix(".pkl.before-restore")
    if STATE_FILE.exists():
        shutil.copy2(STATE_FILE, backup)
        print_info(f"Backed up current pickle to {backup.name}")
    shutil.copy2(snap_path, STATE_FILE)
    print_success(
        f"Restored {snap_path.name} -> {STATE_FILE.name}. "
        "Quit and re-launch ResearchBuddy to load the restored graph."
    )


# ── Open-Access harvest session ───────────────────────────────────────────────

def harvest_session(graph: HierarchicalResearchGraph) -> None:
    """
    Auto-fetch legal open-access full texts for graph papers that lack one
    (arXiv -> Unpaywall -> OpenAlex -> Europe PMC). Each success upgrades an
    abstract-only node into a full node with section embeddings and parsed
    references — the graph feeds itself, no manual PDF hunting.
    """
    from researchbuddy.core import oa_harvester as oh

    print_header("Open-Access Full-Text Harvest")
    print_info(
        "Fetches PDFs ONLY from legal open-access locations (Unpaywall,\n"
        "OpenAlex, arXiv, Europe PMC). Paywalled papers are skipped — never\n"
        "circumvented. License + provenance are recorded for every download."
    )
    if not cfg.UNPAYWALL_EMAIL:
        print_warn(
            "Tip: set OPENALEX_MAILTO=you@example.com to unlock Unpaywall\n"
            "  (the single best OA resolver) and faster OpenAlex responses."
        )

    todo = oh.harvestable_papers(graph)
    if not todo:
        print_warn("Nothing to harvest — every paper either has a local file "
                   "or lacks a DOI/arXiv id.")
        return
    print_info(f"\n{len(todo)} paper(s) could gain full text "
               f"(rated papers first).")
    raw = ask(f"How many to try this run? (max {len(todo)})",
              str(min(len(todo), cfg.HARVEST_MAX_PER_RUN))).strip()
    try:
        n = max(1, min(int(raw), len(todo)))
    except ValueError:
        n = min(len(todo), cfg.HARVEST_MAX_PER_RUN)

    report = oh.harvest(graph, papers=todo, max_papers=n, progress=print_info)

    print()
    print_success(
        f"Harvest done: {report.ingested} ingested / "
        f"{report.downloaded} downloaded / {report.checked} checked."
    )
    if report.no_oa:
        print_info(f"  {report.no_oa} paper(s) have no legal OA copy "
                   "(skipped; try again later — OA status changes).")
    if report.by_provider:
        prov = ", ".join(f"{k}: {v}" for k, v in report.by_provider.items())
        print_info(f"  Sources: {prov}")
    for err in report.errors[:5]:
        print_warn(f"  {err}")
    print_info(f"  PDFs + provenance records: {cfg.OA_LIBRARY_DIR}")

    if report.ingested:
        print_info("\n[graph] Rebuilding hierarchy with new full-text content ...")
        graph.rebuild_hierarchy()
        save(graph)


# ── Snowball session ──────────────────────────────────────────────────────────

def snowball_session(graph: HierarchicalResearchGraph) -> None:
    """
    Backward/forward citation snowballing from the user's best-rated papers,
    feeding straight into the normal one-at-a-time rating loop.
    """
    from researchbuddy.core import snowball as sb

    print_header("Citation Snowballing")
    print_info(
        "Backward: follow the reference lists of your best papers.\n"
        "Forward:  find newer papers citing them.\n"
        "Uses bibliographic metadata only (OpenAlex / Semantic Scholar)."
    )

    used = sb.load_used_seeds()
    if used and ask(f"{len(used)} seed(s) already expanded in past rounds. "
                    "Reset and start over? (y/n)", "n").lower() == "y":
        sb.reset_used_seeds()
        used = set()

    seeds = sb.pick_seeds(graph, exclude=used)
    if not seeds:
        print_warn("No fresh seeds — every candidate has been expanded. Rate "
                   "more papers, or answer 'y' above to reset the frontier.")
        return
    print_info(f"\nSeeding from {len(seeds)} paper(s) (walking the frontier "
               "outward, skipping already-expanded seeds):")
    for m in seeds[:5]:
        if m.user_rating:
            r = f"rated {m.user_rating:.0f}/10"
        elif m.source in ("snowball", "discovered"):
            r = "frontier"
        else:
            r = "seed PDF"
        print_info(f"  - {m.title[:65]} ({r})")
    if len(seeds) > 5:
        print_info(f"  ... and {len(seeds) - 5} more")

    dir_raw = ask("Direction (b=backward, f=forward, both)", "both").strip().lower()
    directions = {"b": ("backward",), "f": ("forward",)}.get(dir_raw,
                  ("backward", "forward"))

    print_info("\nSnowballing ... (one OpenAlex round-trip per seed)\n")
    candidates, stats = sb.snowball_round(
        graph, seeds=seeds, directions=directions, progress=print_info
    )
    audit.log_event("snowball", directions=list(directions),
                    n_seeds=stats["n_seeds"], fetched=stats["fetched"],
                    new_unique=stats["new_unique"],
                    saturation=stats["saturation_ratio"])

    print()
    print_success(
        f"Round complete: {stats['new_unique']} new unique papers "
        f"from {stats['fetched']} fetched references/citations."
    )
    remaining = stats.get("seeds_remaining")
    if stats["saturated"]:
        print_success(
            "  SATURATION reached — little new AND no fresh frontier left. "
            "Your coverage of this literature is solid."
        )
    else:
        print_info(
            f"  Saturation ratio: {stats['saturation_ratio']:.0%} new"
            + (f"  |  {remaining} seed(s) still on the frontier — run again "
               "to walk further out." if remaining else "")
        )

    if not candidates:
        return

    n_show = cfg.N_RECOMMENDATIONS
    raw = ask(f"\nReview top how many now? (0 to skip)", str(max(n_show, 5))).strip()
    try:
        n_show = max(0, int(raw))
    except ValueError:
        pass
    if n_show <= 0:
        return

    results = graph.rank_candidates(candidates, n=n_show, exploration_ratio=0.0)
    display_results(results)
    if results:
        rating_session(graph, results)
        print_info("Auto-saving ...")
        save(graph)


# ── Review pack export session ────────────────────────────────────────────────

def review_pack_session(graph: HierarchicalResearchGraph) -> None:
    """Export the full review pack: BibTeX, RIS, matrix, scaffold, PRISMA."""
    from researchbuddy.core.review_builder import export_review_pack, review_papers

    print_header("Export Review Pack")
    papers = review_papers(graph)
    if not papers:
        print_warn("No papers qualify yet — import PDFs or rate some "
                   "suggestions first.")
        return

    n_rated = sum(1 for m in papers if m.user_rating is not None)
    print_info(
        f"{len(papers)} papers will be exported ({n_rated} rated).\n"
        "Pack contents: review.bib, review.ris, synthesis_matrix.csv,\n"
        "review_scaffold.md (themed skeleton), prisma_flow.md.\n"
        "Only bibliographic facts + your own ratings + locally-generated\n"
        "synthesis text are written — no third-party text is reproduced."
    )
    use_llm = _check_llm_available()
    if use_llm:
        print_info("Local LLM detected — themes get original synthesis "
                   "paragraphs (slower).")
        if ask("Use LLM for theme synthesis? (y/n)", "y").lower() != "y":
            use_llm = False

    if ask("Export now? (y/n)", "y").lower() != "y":
        return
    print_info("Building pack ...")
    pack = export_review_pack(graph, use_llm=use_llm)
    print_success(f"Review pack written to: {pack}")
    for f in sorted(pack.iterdir()):
        print_info(f"  - {f.name}")


# ── Living review session ─────────────────────────────────────────────────────

def living_review_session(graph: HierarchicalResearchGraph) -> None:
    """Manage watch queries and check for new papers since the last visit."""
    from researchbuddy.core import watcher as wt

    while True:
        watches = wt.load_watches()
        print_header("Living Review — Watch Queries")
        if watches:
            for i, w in enumerate(watches, 1):
                kw = f"  +[{', '.join(w['keywords'])}]" if w.get("keywords") else ""
                print_info(f"  [{i}] '{w['query']}'{kw}  "
                           f"(last checked {w['last_checked']})")
        else:
            print_info("  No watches yet. A watch re-runs a query against "
                       "OpenAlex,\n  restricted to papers published since "
                       "your last check.")
        print()
        print_info("  [c] Check all watches now")
        print_info("  [a] Add a watch")
        print_info("  [d] Delete a watch")
        print_info("  [b] Back")

        choice = ask("Choose", "c" if watches else "a").strip().lower()
        if choice == "b":
            return
        elif choice == "a":
            q = ask("Watch query (research topic)", "").strip()
            if not q:
                continue
            raw = ask("Extra keywords (comma-separated, optional)", "").strip()
            kws = [k.strip() for k in raw.split(",") if k.strip()]
            wt.add_watch(q, kws)
            print_success(f"Watch added. First check covers the last 30 days.")
        elif choice == "d":
            raw = ask("Watch number to delete", "").strip()
            try:
                idx = int(raw) - 1
            except ValueError:
                continue
            if wt.remove_watch(idx):
                print_success("Watch removed.")
            else:
                print_warn("No such watch.")
        elif choice == "c":
            if not watches:
                print_warn("Add a watch first.")
                continue
            print()
            reports = wt.check_watches(graph, progress=print_info)
            any_new = False
            for rep in reports:
                results = rep["results"]
                print()
                print_header(f"Watch: {rep['watch']['query']}")
                if not results:
                    print_info("  Nothing new (or nothing relevant enough).")
                    continue
                any_new = True
                display_results(results)
                if ask("Rate these now? (y/n)", "y").lower() == "y":
                    rating_session(graph, results)
            if any_new:
                print_info("Auto-saving ...")
                save(graph)


# ── Graph capsule session (social-psyche interop) ─────────────────────────────

def capsule_session(graph: HierarchicalResearchGraph) -> None:
    """
    Package this graph as a privacy-scrubbed capsule, or merge a collaborator's
    capsule into yours — measuring how reliable the merge is with standard
    graph-theory error measures (spectral distance, DeltaCon, degree-KS,
    Jaccard, Gromov–Wasserstein).
    """
    from researchbuddy.core import capsule as cap

    print_header("Graph Capsules — Share & Merge (social-psyche)")
    print_info(
        "A capsule packages your graph for a collaborator WITHOUT leaking\n"
        "private data: thought/draft nodes are always dropped; DOIs, titles,\n"
        "and ratings are shared only if you opt in. Embeddings + structure\n"
        "always travel, so graphs can be aligned even fully anonymised."
    )
    print()
    print_info("  [1] Export my graph as a capsule (file)")
    print_info("  [2] Inspect a capsule file")
    print_info("  [3] Merge a capsule file into my graph (with reliability report)")
    print_info("  [4] Merge LIVE with a collaborator (secure network, no files)")
    print_info("  [b] Back")
    choice = ask("Choose", "1").strip().lower()
    if choice == "b":
        return

    if choice == "1":
        ids = ask("Share DOIs + titles? (y/n)  [needed to IMPORT new papers; "
                  "off = fully private structural share]", "n").lower() == "y"
        rates = ask("Share your ratings? (y/n)", "n").lower() == "y"
        cfg.CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
        default = cfg.CAPSULE_DIR / f"mygraph_{int(time.time())}.rbcapsule"
        out = ask("Output path", str(default)).strip() or str(default)
        capsule = cap.export_capsule(
            graph, share_identifiers=ids, share_ratings=rates)
        path = cap.write_capsule(capsule, out)
        print_success(f"Capsule written: {path}")
        print_info(f"  {capsule.stats['n_papers']} papers, "
                   f"{capsule.stats['n_sem_edges']} semantic edges, "
                   f"{capsule.stats['n_clusters']} clusters.")
        print_info(f"  Privacy: identifiers={'on' if ids else 'OFF'}, "
                   f"ratings={'on' if rates else 'OFF'}.")
        print_info("  Send this file to a collaborator; they merge it with "
                   "option [3].")

    elif choice == "2":
        path = ask("Capsule path", "").strip()
        if not path:
            return
        try:
            capsule = cap.load_capsule(path)
        except Exception as e:
            print_warn(f"Could not read capsule: {e}")
            return
        print_success(f"Capsule v{capsule.version} created {capsule.created}")
        for k, v in capsule.stats.items():
            print_info(f"  {k}: {v}")
        print_info(f"  Privacy: {capsule.privacy}")

    elif choice == "3":
        path = ask("Capsule path to merge", "").strip()
        if not path:
            return
        try:
            capsule = cap.load_capsule(path)
        except Exception as e:
            print_warn(f"Could not read capsule: {e}")
            return
        print_info("Merging + computing graph-theoretic reliability ...")
        report = cap.merge_capsule(graph, capsule, rebuild=True)
        _print_merge_report(report)
        if report.imported:
            save(graph)
            print_success("Graph saved with imported papers. Harvest their "
                          "OA full text with option 15.")

    elif choice == "4":
        networked_merge_session(graph)


def _print_merge_report(report) -> None:
    """Shared pretty-printer for a capsule MergeReport (file or networked)."""
    print_header("Merge Reliability Report")
    print_info(f"  Shared by DOI         : {report.shared_by_doi}")
    print_info(f"  Shared by embedding   : {report.shared_by_embedding}")
    print_success(f"  New papers imported   : {report.imported}")
    print_info(f"  Novel regions (unmerged): {report.novel_regions}")
    print()

    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

    print_info("  Graph-theory error / similarity measures:")
    print_info(f"    Jaccard (DOI overlap)   : {_fmt(report.jaccard_doi)}")
    print_info(f"    Spectral distance        : {_fmt(report.spectral_distance)}")
    print_info(f"    DeltaCon similarity      : {_fmt(report.deltacon_similarity)}")
    print_info(f"    Degree-dist KS           : {_fmt(report.degree_ks)}")
    print_info(f"    Gromov–Wasserstein dist  : {_fmt(report.gw_distortion)}")
    print_info(f"    Modularity (mine)        : {_fmt(report.modularity_self)}")
    for note in report.notes:
        print_warn(f"  - {note}")


def networked_merge_session(graph: HierarchicalResearchGraph) -> None:
    """
    Secure, live merge with a collaborator over the network — capsules never
    touch disk. Delegates the protocol + crypto to the social-psyche package
    (authenticated, forward-secret, AEAD channel + PSI). Optional dependency:
    ResearchBuddy works fully without it; this one menu path needs it.
    """
    try:
        from social_psyche import netmerge
        from social_psyche.identity import Identity
    except ImportError:
        print_warn("Live networked merge needs the 'social-psyche' package "
                   "(separate, optional).")
        print_info("  Install it (editable, with secure transport):")
        print_info("    pip install -e <path-to>/social-psyche[net]")
        print_info("  Then re-open this menu. File-based merge ([1]/[3]) works "
                   "without it.")
        return

    ident = Identity.load_or_create()
    print_header("Secure Networked Merge (social-psyche)")
    print_success("YOUR identity fingerprint — read it to your collaborator over")
    print_success("a trusted channel (phone / in person) so they can verify you:")
    print_info(f"    {ident.fingerprint()}")
    print()

    role = ask("Role: [s]erve (wait for peer) or [c]onnect (dial peer)?", "s")\
        .strip().lower()
    peer_fp = ask("Pin peer's fingerprint (paste it, or Enter for "
                  "trust-on-first-use)", "").strip() or None
    if peer_fp is None:
        print_warn("No fingerprint pinned: the channel is encrypted but NOT "
                   "verified against a known peer — a man-in-the-middle is "
                   "possible. Confirm the peer fingerprint shown after connecting.")
    share_ids = ask("Share DOIs + titles? (y/n)  [needed so the peer can IMPORT "
                    "your new papers]", "y").lower() == "y"
    share_rt = ask("Share your ratings? (y/n)", "n").lower() == "y"
    kwargs = dict(identity=ident, expected_peer_fp=peer_fp,
                  share_identifiers=share_ids, share_ratings=share_rt)

    try:
        if role.startswith("c"):
            host = ask("Peer host / IP", "127.0.0.1").strip()
            port = int(ask("Port", "9333").strip() or "9333")
            print_info(f"Connecting to {host}:{port} (handshake + PSI + merge) ...")
            result = netmerge.connect(graph, host, port, **kwargs)
        else:
            host = ask("Bind address", "0.0.0.0").strip()
            port = int(ask("Port", "9333").strip() or "9333")
            print_info(f"Listening on {host}:{port} — waiting for peer "
                       "(Ctrl-C to cancel) ...")
            result = netmerge.serve(graph, host, port, **kwargs)
    except KeyboardInterrupt:
        print_warn("Cancelled.")
        return
    except Exception as e:
        print_warn(f"Networked merge failed: {e}")
        return

    print()
    print_info(f"Peer fingerprint (verify this matches your collaborator): "
               f"{result.peer_fingerprint}")
    print_info(f"Shared papers found via PSI: {len(result.shared_dois)}")
    _print_merge_report(result.report)
    if result.report.imported:
        save(graph)
        print_success("Graph saved with imported papers. Harvest their OA full "
                      "text with option 15.")


# ── Open archive session (anti-lock-in) ───────────────────────────────────────

def archive_session(graph: HierarchicalResearchGraph) -> Optional[HierarchicalResearchGraph]:
    """
    Export the full graph to an open, hash-verified archive (JSONL + NPZ), or
    rebuild a graph from one. This is the guarantee that your knowledge
    topology never depends on ResearchBuddy, pickle, or any single vendor.
    Returns a new graph when an import happened, else None.
    """
    from pathlib import Path
    from researchbuddy.core import archive as ar

    print_header("Open Archive — Your Topology, No Lock-In")
    print_info(
        "Exports the ENTIRE graph as open formats anyone can read without\n"
        "ResearchBuddy: papers.jsonl, edges.jsonl, embeddings.npz (pickle-\n"
        "free), state.json, manifest.json with sha256 integrity hashes.\n"
        "Personal backup — contains abstracts; publish capsules (19), not this."
    )
    print()
    print_info("  [1] Export archive")
    print_info("  [2] Verify an archive (integrity check)")
    print_info("  [3] Import archive (REPLACES the current graph)")
    print_info("  [b] Back")
    choice = ask("Choose", "1").strip().lower()

    if choice == "1":
        default = cfg.ARCHIVE_DIR / f"archive_{time.strftime('%Y%m%d_%H%M%S')}"
        out = ask("Output directory", str(default)).strip() or str(default)
        print_info("Exporting ...")
        path = ar.export_archive(graph, out)
        print_success(f"Archive written: {path}")
        for f in sorted(Path(path).iterdir()):
            print_info(f"  - {f.name}  ({f.stat().st_size:,} bytes)")
    elif choice == "2":
        d = ask("Archive directory", "").strip()
        if not d:
            return None
        try:
            man = ar.verify_archive(d)
        except ar.ArchiveError as e:
            print_warn(f"VERIFY FAILED: {e}")
            return None
        print_success(
            f"Archive OK: {man['n_papers']} papers, {man['n_edges']} edges, "
            f"created {man['created']} — all hashes match.")
    elif choice == "3":
        d = ask("Archive directory to import", "").strip()
        if not d:
            return None
        print_warn("This REPLACES your current in-memory graph. Your existing "
                   "pickle is backed up before the new graph is saved.")
        if ask("Proceed? (y/n)", "n").lower() != "y":
            return None
        try:
            new_graph = ar.import_archive(d)
        except ar.ArchiveError as e:
            print_warn(f"Import failed: {e}")
            return None
        import shutil
        if cfg.STATE_FILE.exists():
            backup = cfg.STATE_FILE.with_suffix(".pkl.before-archive-import")
            shutil.copy2(cfg.STATE_FILE, backup)
            print_info(f"Backed up current state to {backup.name}")
        save(new_graph)
        s = new_graph.stats()
        print_success(
            f"Imported: {s['total_papers']} papers, "
            f"{s['rated_papers']} rated. Graph saved.")
        return new_graph
    return None


# ── Main menu ─────────────────────────────────────────────────────────────────

def main_menu(graph: HierarchicalResearchGraph, plot: bool = True):
    options = {
        "1": "Search for new papers",
        "2": "Show graph statistics",
        "3": "Add PDF folder",
        "4": "Fetch citation data (improves fusion quality)",
        "5": "Resolve Semantic Scholar IDs for seed papers",
        "6": "Rebuild hierarchy & regenerate all graph PDFs",
        "7": "Query your research network (Reasoning Mode)",
        "8": "Creative Mode - generate & rate argument paragraphs",
        "9": "LLM status & setup",
        "10": "Audit graph edges (low-confidence & anomalies)",
        "11": "Quality & reliability report",
        "12": "Browse graph in Neo4j (open Browser + load style)",
        "13": "Manage services (Neo4j / GROBID)",
        "14": "Upload my own writing / thoughts (essay, note, draft, ...)",
        "15": "Harvest open-access full texts (legal OA only — graph feeds itself)",
        "16": "Snowball citations (backward/forward from your best papers)",
        "17": "Export review pack (BibTeX / RIS / matrix / scaffold / PRISMA)",
        "18": "Living review (watch queries — what's new since last check)",
        "19": "Graph capsules — share / merge with a collaborator (social-psyche)",
        "20": "Open archive — export/import full graph, zero lock-in (JSONL/NPZ)",
        "q": "Save & quit",
    }
    while True:
        print_header("ResearchBuddy")
        s = graph.stats()
        mode_tag = ""
        if s['total_papers'] < cfg.COLD_START_THRESHOLD:
            mode_tag = "  [COLD-START]"
        elif graph._learned_signal_weights is not None:
            mode_tag = "  [LEARNED]"
        llm_tag = ""
        if cfg.LLM_ENABLED and not _check_llm_available():
            llm_tag = "  [NO LLM]"
        backend_tag = f"  [{graph._backend.backend_name}]"
        print_info(
            f"  {s['total_papers']} papers  |  {s['rated_papers']} rated  |  "
            f"{s['hierarchy_levels']} levels  |  "
            f"{s['niche_clusters']} niches  |  {s['area_clusters']} areas  |  "
            f"sem={s['semantic_edges']} edges  cit={s['citation_edges']} edges"
            f"{mode_tag}{llm_tag}{backend_tag}"
        )
        print()
        for key, desc in options.items():
            if HAS_RICH:
                # Escape the brackets so Rich doesn't treat e.g. "[q]" as
                # markup for an (unknown) tag and silently strip it.
                console.print(rf"  [cyan]\[{key}][/] {desc}")
            else:
                print(f"  [{key}] {desc}")
        print()

        choice = ask("Choose", "1").strip().lower()

        if choice == "q":
            return graph          # caller must save THIS object (archive
                                  # import may have replaced the original)
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
        elif choice == "11":
            quality_report_session(graph)
        elif choice == "12":
            browse_in_neo4j(graph)
        elif choice == "13":
            manage_services()
        elif choice == "14":
            upload_thought_session(graph)
            save(graph)
        elif choice == "15":
            harvest_session(graph)
        elif choice == "16":
            snowball_session(graph)
        elif choice == "17":
            review_pack_session(graph)
        elif choice == "18":
            living_review_session(graph)
        elif choice == "19":
            capsule_session(graph)
        elif choice == "20":
            new_graph = archive_session(graph)
            if new_graph is not None:
                graph = new_graph
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
    p.add_argument("--no-services", action="store_true",
                   help="Skip the Docker auto-launch prompt for Neo4j / GROBID")

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

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Set logging verbosity (default: INFO)")
    return p


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI argument ranges before applying to config."""
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("Error: --alpha must be between 0.0 and 1.0")
    if args.exploration_ratio is not None and not (0.0 <= args.exploration_ratio <= 1.0):
        raise SystemExit("Error: --exploration-ratio must be between 0.0 and 1.0")
    if args.similarity_threshold is not None and not (0.0 <= args.similarity_threshold <= 1.0):
        raise SystemExit("Error: --similarity-threshold must be between 0.0 and 1.0")
    if args.n_recommendations is not None and args.n_recommendations < 1:
        raise SystemExit("Error: --n-recommendations must be at least 1")


def _print_service_status() -> None:
    """
    One-line status for each optional service. Prints unconditionally so the
    user always knows what's plugged in without reading container logs.

    For Neo4j, we don't just check the HTTP endpoint — we also do a bolt
    auth probe, since 'http reachable' does not imply 'usable as backend'.
    Returns nothing; emits a follow-up actionable line if bolt failed.
    """
    grobid_alive = svc._service_alive(svc.GROBID_SPEC)
    docker_ok    = svc.docker_available()

    # Neo4j: bolt probe (the one that actually matters for the backend)
    neo4j_password = os.environ.get("RESEARCHBUDDY_NEO4J_PASSWORD", "researchbuddy")
    bolt = svc.probe_neo4j_bolt(password=neo4j_password)
    http_alive = svc._service_alive(svc.NEO4J_SPEC)

    def status(label: str, alive: bool, extra: str = "") -> str:
        if HAS_RICH:
            mark = "[green]ok[/]" if alive else "[red]down[/]"
        else:
            mark = "ok" if alive else "down"
        return f"{label}: {mark}{extra}"

    if bolt.ok:
        neo4j_part = status("Neo4j", True)
    elif http_alive:
        # HTTP works, bolt doesn't — that's a real story worth telling
        if HAS_RICH:
            neo4j_part = "Neo4j: [yellow]http only[/]"
        else:
            neo4j_part = "Neo4j: http only"
    else:
        neo4j_part = status("Neo4j", False)

    parts = [neo4j_part, status("GROBID", grobid_alive)]
    if not docker_ok:
        parts.append("Docker: [yellow]not detected[/]" if HAS_RICH else "Docker: not detected")
    line = "  Services -- " + "  |  ".join(parts)
    if HAS_RICH:
        console.print(line)
    else:
        print(line)

    # If Neo4j is half-up, print a clear next step
    if http_alive and not bolt.ok:
        if HAS_RICH:
            console.print(f"  [yellow]→ Neo4j bolt: {bolt.reason}[/]")
        else:
            print(f"  -> Neo4j bolt: {bolt.reason}")


def _ensure_services_at_startup() -> None:
    """
    Detect Neo4j and GROBID. If either is missing, ask the user once whether
    to auto-launch it (using Docker). Choices are remembered across runs.

    Sets the Neo4j env vars on success so the rest of the app picks it up
    without the user having to export anything.
    """
    if not svc.docker_available():
        # Tell the user why we didn't ask, instead of silently skipping —
        # and show exactly how to fix it.
        print_warn(
            "Docker not detected -- can't auto-launch Neo4j / GROBID. "
            "ResearchBuddy still works (NetworkX backend + pdfplumber "
            "extraction), but here's how to unlock the full stack:"
        )
        print_docker_help()
        return

    prefs = svc.load_prefs()
    use_neo4j = False    # set to True once we decide a Neo4j endpoint is usable

    # ── Neo4j ─────────────────────────────────────────────────────────────
    if svc._service_alive(svc.NEO4J_SPEC):
        # Already running (Desktop, manual `docker run`, or a previous session)
        use_neo4j = True
    else:
        choice = prefs.get("neo4j_auto_launch")
        if choice is None:
            ans = ask(
                "Neo4j not running. Start it via Docker now? (y/n/never)",
                "y",
            ).strip().lower()
            if ans in ("never", "no-never"):
                prefs["neo4j_auto_launch"] = "never"
                svc.save_prefs(prefs)
            elif ans.startswith("y"):
                prefs["neo4j_auto_launch"] = "yes"
                svc.save_prefs(prefs)
                choice = "yes"
            else:
                choice = "no"
        if choice == "yes":
            res = svc.ensure_running(svc.NEO4J_SPEC)
            if res.already_running:
                print_info("Neo4j already running.")
                use_neo4j = True
            elif res.started:
                print_success("Neo4j is up at http://localhost:7474")
                use_neo4j = True
            else:
                print_warn(f"Could not start Neo4j: {res.error}")

    # Configure ResearchBuddy to talk to Neo4j: this MUST happen in both
    # "started fresh" and "already running" paths, and the config module
    # must be reloaded so create_backend() sees the updated values.
    if use_neo4j:
        os.environ.setdefault("RESEARCHBUDDY_NEO4J_ENABLED", "true")
        os.environ.setdefault("RESEARCHBUDDY_NEO4J_PASSWORD", "researchbuddy")
        import importlib, researchbuddy.config as _cfg_mod
        importlib.reload(_cfg_mod)

    # ── GROBID ────────────────────────────────────────────────────────────
    if not svc._service_alive(svc.GROBID_SPEC):
        choice = prefs.get("grobid_auto_launch")
        if choice is None:
            ans = ask(
                "GROBID not running. Start it via Docker now? (y/n/never)",
                "y",
            ).strip().lower()
            if ans in ("never", "no-never"):
                prefs["grobid_auto_launch"] = "never"
                svc.save_prefs(prefs)
            elif ans.startswith("y"):
                prefs["grobid_auto_launch"] = "yes"
                svc.save_prefs(prefs)
                choice = "yes"
            else:
                choice = "no"
        if choice == "yes":
            res = svc.ensure_running(svc.GROBID_SPEC)
            if res.already_running:
                print_info("GROBID already running.")
            elif res.started:
                print_success("GROBID is up at http://localhost:8070")
            else:
                print_warn(f"Could not start GROBID: {res.error}")


def main():
    args = _build_parser().parse_args()

    # ── Logging setup ──────────────────────────────────────────────────────
    log_level = getattr(logging, (getattr(args, "log_level", None) or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="[%(name)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    _validate_cli_args(args)

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

    # ── LLM status at startup ─────────────────────────────────────────────
    _show_llm_status_banner()

    # ── Saved CORE API key (service menu option 10) ───────────────────────
    _apply_core_key_from_prefs()

    # ── Auto-launch optional Docker services (Neo4j, GROBID) ──────────────
    # If Docker is installed, offer to start any missing service. The user's
    # answer (yes/no/never) is remembered across runs.
    if not getattr(args, "no_services", False):
        try:
            _ensure_services_at_startup()
        except Exception as e:
            logger.debug("Service auto-launch skipped: %s", e)
    # Always show one-line status so the user knows what's connected.
    _print_service_status()

    if args.reset and cfg.STATE_FILE.exists():
        cfg.STATE_FILE.unlink()
        print_info("Saved state cleared.")

    graph = load()
    if graph is None:
        graph = HierarchicalResearchGraph(alpha=cfg.FUSION_ALPHA)
        print_info("Starting with a fresh graph.")
    else:
        # Reapply kept self-tuning experiments (researchbuddy-sentinel
        # --autotune runs overnight; its wins carry into every session).
        try:
            from researchbuddy.core.autotune import apply_saved_tuning
            applied = apply_saved_tuning(graph)
            if applied:
                print_info(f"  Self-tuned parameters applied: "
                           f"{', '.join(applied)}")
        except Exception as e:
            logger.debug("autotune apply skipped: %s", e)
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
        result = main_menu(graph, plot=plot)
        if result is not None:
            graph = result    # archive import may have swapped the graph
    except KeyboardInterrupt:
        print("\n[Interrupted]")

    print_info("\nSaving before exit ...")
    save(graph)
    print_success("Session saved. See you next time!")


if __name__ == "__main__":
    main()
