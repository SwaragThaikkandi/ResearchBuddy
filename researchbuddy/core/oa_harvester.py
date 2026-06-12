"""
Legal open-access full-text autopilot.

ResearchBuddy's graph gets dramatically smarter when papers carry full text
(section embeddings, parsed references) instead of just abstracts. This
module closes that loop WITHOUT touching paywalled or pirated copies: it
asks only services that index author/publisher-sanctioned open-access
locations, in this order:

  1. arXiv       — preprints the authors themselves uploaded
  2. Unpaywall   — DOI -> legal OA copies only (explicitly excludes Sci-Hub
                   and other infringing hosts)
  3. OpenAlex    — best_oa_location across 250M+ works
  4. Europe PMC  — the open-access subset of biomedical literature

Every download writes a provenance sidecar (provider, URL, license, version,
retrieved_at) next to the PDF, so the user can always show where a file came
from and under what terms. Papers with no legal OA copy are simply reported
as such — the harvester never tries to work around a paywall.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import requests

from researchbuddy.config import (
    UNPAYWALL_URL, UNPAYWALL_EMAIL, EUROPEPMC_URL, OPENALEX_URL,
    OA_LIBRARY_DIR, OA_DOWNLOAD_TIMEOUT, OA_MAX_PDF_MB,
    HARVEST_MAX_PER_RUN, REQUEST_TIMEOUT, REQUEST_DELAY,
)
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ResearchBuddy/0.6 (open-access harvester; legal OA only)"}


@dataclass
class OALocation:
    """A legal open-access location for one paper."""
    pdf_url: str
    provider: str               # arxiv | unpaywall | openalex | europepmc
    license: str = ""           # e.g. cc-by, cc-by-nc; "" = unknown/host terms
    version: str = ""           # publishedVersion | acceptedVersion | submittedVersion
    host_type: str = ""         # publisher | repository
    landing_url: str = ""


@dataclass
class HarvestReport:
    checked: int = 0
    resolved: int = 0
    downloaded: int = 0
    ingested: int = 0
    no_oa: int = 0
    errors: list[str] = field(default_factory=list)
    by_provider: dict[str, int] = field(default_factory=dict)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, headers=_HEADERS,
                         timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug("OA lookup failed for %s: %s", url, e)
    return None


# ── Resolvers (one per provider, tried in order) ──────────────────────────────

def _resolve_arxiv(meta: PaperMeta) -> Optional[OALocation]:
    if not meta.arxiv_id:
        return None
    aid = meta.arxiv_id.strip().replace("arXiv:", "")
    return OALocation(
        pdf_url     = f"https://arxiv.org/pdf/{aid}",
        provider    = "arxiv",
        license     = "arxiv-nonexclusive",
        version     = "submittedVersion",
        host_type   = "repository",
        landing_url = f"https://arxiv.org/abs/{aid}",
    )


def _resolve_unpaywall(meta: PaperMeta) -> Optional[OALocation]:
    if not meta.doi or not UNPAYWALL_EMAIL:
        return None
    data = _get_json(f"{UNPAYWALL_URL}/{meta.doi}",
                     params={"email": UNPAYWALL_EMAIL})
    if not data or not data.get("is_oa"):
        return None
    loc = data.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf") or ""
    if not pdf_url:
        return None
    return OALocation(
        pdf_url     = pdf_url,
        provider    = "unpaywall",
        license     = (loc.get("license") or "")[:60],
        version     = loc.get("version") or "",
        host_type   = loc.get("host_type") or "",
        landing_url = loc.get("url_for_landing_page") or "",
    )


def _resolve_openalex(meta: PaperMeta) -> Optional[OALocation]:
    if not meta.doi:
        return None
    data = _get_json(f"{OPENALEX_URL}/doi:{meta.doi}")
    if not data:
        return None
    loc = data.get("best_oa_location") or {}
    pdf_url = loc.get("pdf_url") or ""
    if not pdf_url or not loc.get("is_oa"):
        return None
    src = loc.get("source") or {}
    return OALocation(
        pdf_url     = pdf_url,
        provider    = "openalex",
        license     = (loc.get("license") or "")[:60],
        version     = loc.get("version") or "",
        host_type   = (src.get("type") or ""),
        landing_url = loc.get("landing_page_url") or "",
    )


def _resolve_europepmc(meta: PaperMeta) -> Optional[OALocation]:
    if not meta.doi:
        return None
    data = _get_json(
        f"{EUROPEPMC_URL}/search",
        params={
            "query":  f'DOI:"{meta.doi}" AND OPEN_ACCESS:Y',
            "format": "json",
            "pageSize": 1,
        },
    )
    if not data:
        return None
    hits = ((data.get("resultList") or {}).get("result")) or []
    if not hits:
        return None
    hit = hits[0]
    pmcid = hit.get("pmcid") or ""
    if not pmcid:
        return None
    return OALocation(
        pdf_url     = f"https://europepmc.org/articles/{pmcid}?pdf=render",
        provider    = "europepmc",
        license     = (hit.get("license") or "")[:60],
        version     = "publishedVersion",
        host_type   = "repository",
        landing_url = f"https://europepmc.org/articles/{pmcid}",
    )


_RESOLVERS: list[Callable[[PaperMeta], Optional[OALocation]]] = [
    _resolve_arxiv,
    _resolve_unpaywall,
    _resolve_openalex,
    _resolve_europepmc,
]


def resolve_oa(meta: PaperMeta) -> Optional[OALocation]:
    """Find the best legal OA location for a paper, or None if there is none."""
    for resolver in _RESOLVERS:
        loc = resolver(meta)
        if loc is not None:
            return loc
    return None


# ── Download + provenance ─────────────────────────────────────────────────────

def _safe_filename(meta: PaperMeta) -> str:
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", meta.title or meta.paper_id)[:80].strip()
    base = re.sub(r"\s+", "_", base) or meta.paper_id
    return f"{base}_{meta.paper_id[:10]}.pdf"


def download_pdf(loc: OALocation, dest: Path) -> bool:
    """
    Stream-download an OA PDF. Verifies the payload actually is a PDF
    (magic bytes) and enforces a size ceiling. Returns False on any problem.
    """
    max_bytes = OA_MAX_PDF_MB * 1024 * 1024
    ok = False
    try:
        with requests.get(loc.pdf_url, headers=_HEADERS, stream=True,
                          timeout=OA_DOWNLOAD_TIMEOUT, allow_redirects=True) as r:
            if r.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            first = True
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if first:
                        if not chunk.startswith(b"%PDF"):
                            # HTML landing page / error body — not a PDF.
                            break
                        first = False
                    written += len(chunk)
                    if written > max_bytes:
                        logger.warning("OA download exceeded %d MB, aborting: %s",
                                       OA_MAX_PDF_MB, loc.pdf_url)
                        break
                    f.write(chunk)
                else:
                    ok = not first      # at least one valid chunk written
    except Exception as e:
        logger.debug("OA download failed (%s): %s", loc.pdf_url, e)
        ok = False
    if not ok:
        # File handle is closed by now — safe to remove partial output.
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
    return ok


def _write_provenance(pdf_path: Path, meta: PaperMeta, loc: OALocation) -> None:
    sidecar = pdf_path.with_suffix(".provenance.json")
    record = {
        "title":        meta.title,
        "doi":          meta.doi,
        "paper_id":     meta.paper_id,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **asdict(loc),
        "note": ("Downloaded from a legal open-access location surfaced by "
                 f"{loc.provider}. Kept for personal research use under the "
                 "recorded license / host terms."),
    }
    try:
        sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    except OSError as e:  # pragma: no cover - defensive
        logger.debug("Could not write provenance sidecar: %s", e)


# ── The harvest loop ──────────────────────────────────────────────────────────

def harvestable_papers(graph: HierarchicalResearchGraph) -> list[PaperMeta]:
    """Papers that could gain full text: no local file yet, and identifiable."""
    cands = [
        m for m in graph.all_papers()
        if m.kind == "paper" and not m.filepath and (m.doi or m.arxiv_id)
    ]
    # Highest-value first: rated papers, then by recency.
    cands.sort(key=lambda m: (-(m.effective_weight), -(m.year or 0)))
    return cands


def harvest(
    graph: HierarchicalResearchGraph,
    papers: Optional[list[PaperMeta]] = None,
    max_papers: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> HarvestReport:
    """
    Resolve + download + ingest legal OA full texts for graph papers that
    lack one. Each success turns an abstract-only node into a full node
    with section embeddings and parsed references — the graph feeds itself.
    """
    from researchbuddy.core.ingest import ingest_pdf_into_meta, IngestError

    say = progress or (lambda s: None)
    todo = papers if papers is not None else harvestable_papers(graph)
    todo = todo[: (max_papers or HARVEST_MAX_PER_RUN)]

    report = HarvestReport()
    OA_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    for meta in todo:
        report.checked += 1
        say(f"[{report.checked}/{len(todo)}] {meta.title[:60]}")

        loc = resolve_oa(meta)
        time.sleep(REQUEST_DELAY)
        if loc is None:
            report.no_oa += 1
            say("    no legal OA copy found — skipping (never circumventing paywalls)")
            continue
        report.resolved += 1

        dest = OA_LIBRARY_DIR / _safe_filename(meta)
        if not dest.exists():
            if not download_pdf(loc, dest):
                report.errors.append(f"download failed: {meta.title[:50]}")
                say(f"    {loc.provider} location did not yield a valid PDF")
                continue
        report.downloaded += 1
        report.by_provider[loc.provider] = report.by_provider.get(loc.provider, 0) + 1
        _write_provenance(dest, meta, loc)

        try:
            info = ingest_pdf_into_meta(graph, meta, dest)
        except IngestError as e:
            report.errors.append(f"ingest failed: {meta.title[:50]} ({e})")
            say(f"    downloaded but ingest failed: {e}")
            continue
        report.ingested += 1
        lic = loc.license or "host terms"
        say(f"    OK via {loc.provider} ({lic}) — "
            f"{info['n_sections']} sections, {info['n_refs']} refs [{info['parser']}]")

        audit.log_event(
            "fulltext",
            paper_id=meta.paper_id, title=meta.title[:200], doi=meta.doi,
            provider=loc.provider, license=loc.license, url=loc.pdf_url,
        )

    return report
