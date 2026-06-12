"""
PRISMA-grade audit trail.

Systematic reviews live or die on reproducibility: which queries were run,
which records were screened, which were included or excluded and why. This
module is an append-only JSONL event log that every discovery / screening
surface writes to, plus an aggregator that folds the log into PRISMA-2020
flow counts for export.

Only bibliographic facts (titles, DOIs, counts) and the user's own
decisions are stored — no copyrighted expression.

Event vocabulary (all carry a unix `ts`):
    search      {query, keywords, n_results, sources}
    snowball    {direction(s), n_seeds, fetched, new_unique, saturation}
    screen      {paper_id, title, doi, rating, decision}
    fulltext    {paper_id, title, doi, provider, license, url}
    watch_check {query, n_new}
    export      {pack_dir, n_papers, n_included}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from researchbuddy.config import PRISMA_LOG, REVIEW_MIN_RATING_INCLUDE

logger = logging.getLogger(__name__)


def log_event(event: str, log_path: Optional[Path] = None, **fields) -> None:
    """Append one event line. Never raises — auditing must not break the app."""
    path = Path(log_path or PRISMA_LOG)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "event": event, **fields}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("audit log_event failed: %s", e)


def read_events(event: Optional[str] = None,
                log_path: Optional[Path] = None) -> list[dict]:
    """Read all events (optionally filtered by type), oldest first."""
    path = Path(log_path or PRISMA_LOG)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event is None or rec.get("event") == event:
                    out.append(rec)
    except OSError as e:  # pragma: no cover - defensive
        logger.debug("audit read_events failed: %s", e)
    return out


def screen_decision(rating: Optional[float]) -> str:
    """Map a 1-10 rating to a PRISMA screening decision."""
    if rating is None:
        return "skipped"
    return "included" if rating >= REVIEW_MIN_RATING_INCLUDE else "excluded"


def prisma_counts(log_path: Optional[Path] = None) -> dict:
    """
    Fold the event log into PRISMA-2020 flow numbers.

    `screened` counts unique papers with a screen event; a paper's decision
    is its most recent one (re-rating updates the verdict).
    """
    events = read_events(log_path=log_path)

    n_searches = 0
    identified_search = 0
    identified_snowball = 0
    by_source: dict[str, int] = {}
    fulltext_ids: set[str] = set()
    watch_new = 0
    latest_decision: dict[str, str] = {}   # paper_id -> decision

    for rec in events:
        ev = rec.get("event")
        if ev == "search":
            n_searches += 1
            identified_search += int(rec.get("n_results", 0) or 0)
            for src, n in (rec.get("sources") or {}).items():
                by_source[src] = by_source.get(src, 0) + int(n or 0)
        elif ev == "snowball":
            identified_snowball += int(rec.get("new_unique", 0) or 0)
        elif ev == "watch_check":
            watch_new += int(rec.get("n_new", 0) or 0)
        elif ev == "screen":
            pid = rec.get("paper_id") or rec.get("title") or ""
            if pid:
                latest_decision[pid] = rec.get("decision", "skipped")
        elif ev == "fulltext":
            pid = rec.get("paper_id") or rec.get("doi") or rec.get("title") or ""
            if pid:
                fulltext_ids.add(pid)

    decisions = list(latest_decision.values())
    return {
        "n_searches":          n_searches,
        "identified_search":   identified_search,
        "identified_snowball": identified_snowball,
        "identified_watch":    watch_new,
        "identified_total":    identified_search + identified_snowball + watch_new,
        "identified_by_source": by_source,
        "screened":            len(decisions),
        "included":            sum(1 for d in decisions if d == "included"),
        "excluded":            sum(1 for d in decisions if d == "excluded"),
        "skipped":             sum(1 for d in decisions if d == "skipped"),
        "fulltext_retrieved":  len(fulltext_ids),
    }
