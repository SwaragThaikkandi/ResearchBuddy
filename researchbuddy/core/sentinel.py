"""
Sentinel — continuous literature surveillance.

Watches (living review) answer "what's new since I last looked?" — but YOU
still have to look. Sentinel closes the loop: it looks for you on a
schedule, triages what it finds with your learned scoring weights, files
the keepers into an inbox, and writes a dated digest you can read like a
morning briefing. Fully local, fully yours — the surveillance runs on YOUR
machine watching PUBLIC literature; nobody is watching you.

Pieces it composes (all existing):
    watcher.check_watches   what appeared since the last scan (OpenAlex)
    rank_candidates         your personal learned relevance model
    audit trail             every scan logged (PRISMA-grade provenance)

State:
    ~/.researchbuddy/sentinel.json          config + last_run
    ~/.researchbuddy/sentinel_inbox.jsonl   triaged finds awaiting review
    ~/.researchbuddy/digests/               one Markdown briefing per scan
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from researchbuddy.config import DATA_DIR
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

SENTINEL_FILE = DATA_DIR / "sentinel.json"
INBOX_FILE    = DATA_DIR / "sentinel_inbox.jsonl"
DIGEST_DIR    = DATA_DIR / "digests"

DEFAULT_CONFIG = {
    "enabled": False,          # autonomous scanning on/off
    "interval_hours": 24,      # how often to scan when enabled
    "min_score": 0.35,         # triage threshold (learned-relevance score)
    "last_run": 0.0,           # unix time of the last completed scan
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> dict:
    p = Path(path or SENTINEL_FILE)
    cfg = dict(DEFAULT_CONFIG)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("sentinel config unreadable: %s", e)
    return cfg


def save_config(cfg: dict, path: Optional[Path] = None) -> None:
    p = Path(path or SENTINEL_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def is_due(cfg: dict, now: Optional[float] = None) -> bool:
    """Scan due? True when enabled and interval has elapsed since last_run."""
    if not cfg.get("enabled"):
        return False
    now = time.time() if now is None else now
    interval_s = max(1, float(cfg.get("interval_hours", 24))) * 3600
    return (now - float(cfg.get("last_run", 0.0))) >= interval_s


# ── Inbox ─────────────────────────────────────────────────────────────────────

def inbox_list(path: Optional[Path] = None) -> list[dict]:
    p = Path(path or INBOX_FILE)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _inbox_write(entries: list[dict], path: Optional[Path] = None) -> None:
    p = Path(path or INBOX_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def inbox_remove(token: str, path: Optional[Path] = None) -> Optional[dict]:
    """Pop one entry by token; returns it (or None)."""
    entries = inbox_list(path)
    kept, popped = [], None
    for e in entries:
        if popped is None and e.get("token") == token:
            popped = e
        else:
            kept.append(e)
    if popped is not None:
        _inbox_write(kept, path)
    return popped


def entry_to_meta(entry: dict) -> PaperMeta:
    """Rebuild a PaperMeta from an inbox line (for accept-into-graph)."""
    return PaperMeta(
        paper_id=entry.get("token", ""),
        title=entry.get("title", ""),
        abstract=entry.get("abstract", ""),
        authors=list(entry.get("authors") or []),
        year=entry.get("year"),
        doi=entry.get("doi", ""),
        url=entry.get("url", ""),
        venue=entry.get("venue", ""),
        source="sentinel",
    )


# ── The scan ──────────────────────────────────────────────────────────────────

def run_scan(
    graph: HierarchicalResearchGraph,
    config: Optional[dict] = None,
    progress: Optional[Callable[[str], None]] = None,
    inbox_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    digest_dir: Optional[Path] = None,
) -> dict:
    """
    One surveillance sweep: run every watch, triage by learned relevance,
    file keepers into the inbox (deduped), write a digest. Returns a report.
    """
    from researchbuddy.core import watcher as wt

    say = progress or (lambda s: None)
    cfg = config if config is not None else load_config(config_path)
    threshold = float(cfg.get("min_score", 0.35))

    say("Scanning your watched topics for new publications…")
    reports = wt.check_watches(graph, progress=say)

    existing = {e.get("token") for e in inbox_list(inbox_path)}
    in_graph = {m.paper_id for m in graph.all_papers()}
    now_iso = time.strftime("%Y-%m-%d %H:%M")

    new_entries: list[dict] = []
    per_watch: list[dict] = []
    for rep in reports:
        kept_here = []
        for meta, score, _label in rep["results"]:
            if score < threshold:
                continue
            if meta.paper_id in existing or meta.paper_id in in_graph:
                continue
            existing.add(meta.paper_id)
            entry = {
                "token": meta.paper_id,
                "title": meta.title,
                "abstract": (meta.abstract or "")[:400],
                "authors": meta.authors[:4],
                "year": meta.year,
                "doi": meta.doi,
                "url": meta.url,
                "venue": meta.venue,
                "score": round(float(score), 3),
                "watch": rep["watch"]["query"],
                "found_at": now_iso,
            }
            new_entries.append(entry)
            kept_here.append(entry)
        per_watch.append({"query": rep["watch"]["query"],
                          "found": rep["n_found"],
                          "kept": len(kept_here)})

    if new_entries:
        _inbox_write(inbox_list(inbox_path) + new_entries, inbox_path)

    # ── Digest (the morning briefing) ───────────────────────────────────
    digest_path = None
    if new_entries:
        ddir = Path(digest_dir or DIGEST_DIR)
        ddir.mkdir(parents=True, exist_ok=True)
        digest_path = ddir / f"digest_{time.strftime('%Y%m%d_%H%M')}.md"
        L = [f"# Literature digest — {now_iso}",
             "",
             f"{len(new_entries)} new paper(s) cleared your relevance "
             f"threshold ({threshold:.2f}).", ""]
        for w in per_watch:
            hits = [e for e in new_entries if e["watch"] == w["query"]]
            if not hits:
                continue
            L.append(f"## {w['query']}  ({len(hits)} kept / "
                     f"{w['found']} found)")
            L.append("")
            for e in hits:
                doi = f" — https://doi.org/{e['doi']}" if e["doi"] else ""
                L.append(f"- **{e['title']}** "
                         f"({e['year'] or '?'}) · score {e['score']}{doi}")
            L.append("")
        digest_path.write_text("\n".join(L), encoding="utf-8")

    cfg["last_run"] = time.time()
    save_config(cfg, config_path)
    audit.log_event("sentinel_scan",
                    n_watches=len(per_watch),
                    n_found=sum(w["found"] for w in per_watch),
                    n_kept=len(new_entries))
    say(f"Scan complete: {len(new_entries)} paper(s) filed to the inbox.")
    return {"new": len(new_entries),
            "per_watch": per_watch,
            "digest": str(digest_path) if digest_path else None}
