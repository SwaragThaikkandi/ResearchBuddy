"""
grobid_client.py — Client for the GROBID document parsing service.

GROBID (https://grobid.readthedocs.io/) is a machine-learning library for
extracting structured information from scholarly PDFs. It is purpose-built
for academic papers and far outperforms pdfplumber on:

  * Two-column / mixed layouts
  * Title / abstract / section detection
  * Figure and table captions
  * Mathematical formulae
  * Reference parsing (with DOI / arXiv ID resolution)

GROBID runs as an HTTP service (typically in a Docker container). This
module talks to it over HTTP and converts the returned TEI-XML into our
ExtractedPaper dataclass.

If GROBID is unavailable, the caller is expected to fall back to pdfplumber.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

from researchbuddy.core.pdf_processor import (
    ExtractedPaper, Section, Figure, Table, Reference,
    _stable_id, _to_chunks, _fix_ligatures,
)

logger = logging.getLogger(__name__)

# TEI namespace used in GROBID output
TEI_NS = "{http://www.tei-c.org/ns/1.0}"


# ── Availability ──────────────────────────────────────────────────────────────

def is_available(base_url: str, timeout: float = 2.0) -> bool:
    """
    Check whether a GROBID instance is reachable and ready.

    Returns True if the /api/isalive endpoint returns 200 with body "true".
    Logs at debug level on failure (callers handle the fallback themselves).
    """
    try:
        r = requests.get(
            f"{base_url.rstrip('/')}/api/isalive",
            timeout=timeout,
        )
        return r.status_code == 200 and r.text.strip().lower() == "true"
    except requests.RequestException as e:
        logger.debug("GROBID not reachable at %s: %s", base_url, e)
        return False


def warmup(base_url: str, timeout: float = 90.0) -> bool:
    """
    Trigger GROBID model loading by sending a no-op header request.

    The first /api/processHeaderDocument call after a fresh GROBID start
    spends ~20-40s loading transformer models (CPU-only setups can take
    longer). Calling this once means the first real PDF parse is fast.

    Returns True if a 200 came back within the timeout, False otherwise.
    Failures are non-fatal — the caller will still try real requests.
    """
    # Tiny synthetic PDF (1-page, just a title). This is the smallest
    # parseable PDF I could produce. GROBID's response is irrelevant —
    # we only care that the models got loaded.
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Warmup Document) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000052 00000 n \n"
        b"0000000095 00000 n \n0000000179 00000 n \n0000000273 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n333\n%%EOF\n"
    )
    url = f"{base_url.rstrip('/')}/api/processHeaderDocument"
    try:
        files = {"input": ("warmup.pdf", minimal_pdf, "application/pdf")}
        r = requests.post(url, files=files, timeout=timeout)
        # Even a 4xx means the server got the request and loaded models;
        # we treat anything that returns within the timeout as "warmed".
        return r.status_code < 500
    except requests.RequestException as e:
        logger.debug("GROBID warmup failed (%s) — first PDF may be slow.", e)
        return False


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _text(elem: Optional[ET.Element]) -> str:
    """Recursively gather text from a TEI element, ignoring tags."""
    if elem is None:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail)
    return _fix_ligatures(" ".join(parts).strip())


def _extract_doi_from_idno(elem: ET.Element) -> str:
    """Find a DOI from a TEI <idno> sub-element, if present."""
    for idno in elem.iter(f"{TEI_NS}idno"):
        if idno.get("type", "").lower() == "doi" and idno.text:
            return idno.text.strip().lower()
    return ""


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)


def _scan_doi(text: str) -> str:
    m = _DOI_RE.search(text)
    return m.group(0).lower() if m else ""


# ── TEI → ExtractedPaper ──────────────────────────────────────────────────────

def _parse_header(root: ET.Element) -> tuple[str, str, str]:
    """Pull (title, abstract, doi) from <teiHeader>."""
    header = root.find(f"{TEI_NS}teiHeader")
    if header is None:
        return "", "", ""

    # Title — under fileDesc/titleStmt/title or sourceDesc/biblStruct/.../title
    title_elem = header.find(
        f".//{TEI_NS}titleStmt/{TEI_NS}title"
    )
    title = _text(title_elem)[:300] if title_elem is not None else ""

    # Abstract — under profileDesc/abstract
    abstract_elem = header.find(f".//{TEI_NS}abstract")
    abstract = _text(abstract_elem) if abstract_elem is not None else ""

    # DOI — from <idno type="DOI">, anywhere in header
    doi = _extract_doi_from_idno(header)
    if not doi:
        doi = _scan_doi(_text(header))

    return title, abstract, doi


def _parse_sections(root: ET.Element) -> list[Section]:
    """Pull body sections (<div>) with their <head>."""
    out: list[Section] = []
    body = root.find(f".//{TEI_NS}text/{TEI_NS}body")
    if body is None:
        return out
    for div in body.findall(f"{TEI_NS}div"):
        head_elem = div.find(f"{TEI_NS}head")
        heading = _text(head_elem) if head_elem is not None else ""
        # Concatenate <p> contents only (skip the head)
        para_texts: list[str] = []
        for p in div.findall(f"{TEI_NS}p"):
            t = _text(p)
            if t:
                para_texts.append(t)
        body_text = "\n\n".join(para_texts).strip()
        if heading or body_text:
            out.append(Section(heading=heading, text=body_text))
    return out


def _parse_figures(root: ET.Element) -> tuple[list[Figure], list[Table]]:
    """Pull figures and tables (TEI puts both under <figure>; tables have type='table')."""
    figs: list[Figure] = []
    tabs: list[Table] = []
    for fig in root.iter(f"{TEI_NS}figure"):
        is_table = fig.get("type") == "table"
        head_elem = fig.find(f"{TEI_NS}head")
        label = _text(head_elem) if head_elem is not None else ""
        # GROBID also has <label>N</label> separate from <head> — combine if present
        label_elem = fig.find(f"{TEI_NS}label")
        if label_elem is not None and label_elem.text:
            num = label_elem.text.strip()
            if not label.lower().startswith(("figure", "table")):
                kind = "Table" if is_table else "Figure"
                label = f"{kind} {num}: {label}".rstrip(": ")
        fig_desc = fig.find(f"{TEI_NS}figDesc")
        caption = _text(fig_desc) if fig_desc is not None else ""
        if is_table:
            tbl_text = ""
            tbl_elem = fig.find(f"{TEI_NS}table")
            if tbl_elem is not None:
                # Flatten cell text
                cells: list[str] = []
                for cell in tbl_elem.iter(f"{TEI_NS}cell"):
                    t = _text(cell)
                    if t:
                        cells.append(t)
                tbl_text = " | ".join(cells)
            tabs.append(Table(label=label, caption=caption, text=tbl_text))
        else:
            figs.append(Figure(label=label, caption=caption))
    return figs, tabs


def _parse_equations(root: ET.Element) -> list[str]:
    """Pull standalone equations (<formula>)."""
    out: list[str] = []
    for f in root.iter(f"{TEI_NS}formula"):
        t = _text(f)
        if t:
            out.append(t)
    return out


def _parse_references(root: ET.Element) -> list[Reference]:
    """
    Pull bibliography from <text>/<back>/<div>/<listBibl>/<biblStruct>.

    Important: scope to <listBibl> only — there are biblStruct elements in
    <teiHeader>/<sourceDesc> that describe the paper itself (its own DOI,
    journal, etc.) and must NOT be returned as references.
    """
    out: list[Reference] = []
    for list_bibl in root.iter(f"{TEI_NS}listBibl"):
        for bibl in list_bibl.findall(f"{TEI_NS}biblStruct"):
            title_elem = bibl.find(f".//{TEI_NS}title[@type='main']")
            if title_elem is None:
                title_elem = bibl.find(f".//{TEI_NS}title")
            title = _text(title_elem) if title_elem is not None else ""
            # Authors
            authors: list[str] = []
            for author in bibl.iter(f"{TEI_NS}author"):
                persName = author.find(f"{TEI_NS}persName")
                if persName is None:
                    continue
                forename = persName.find(f"{TEI_NS}forename")
                surname = persName.find(f"{TEI_NS}surname")
                f_txt = _text(forename) if forename is not None else ""
                s_txt = _text(surname) if surname is not None else ""
                full = f"{f_txt} {s_txt}".strip()
                if full:
                    authors.append(full)
            # Year
            year = ""
            date_elem = bibl.find(f".//{TEI_NS}date")
            if date_elem is not None:
                year = (date_elem.get("when") or _text(date_elem) or "")[:4]
            # DOI
            doi = _extract_doi_from_idno(bibl)
            if not doi:
                doi = _scan_doi(_text(bibl))
            if title or doi or authors:
                raw = f"{', '.join(authors)} ({year}). {title}".strip(" .")
                out.append(Reference(
                    raw=raw, title=title, doi=doi, year=year, authors=authors,
                ))
    return out


def _full_text_from_sections(sections: list[Section]) -> str:
    """Flatten sections into a single text blob (for chunking + 'full_text')."""
    parts: list[str] = []
    for sec in sections:
        if sec.heading:
            parts.append(sec.heading)
        if sec.text:
            parts.append(sec.text)
    return "\n\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

class GrobidTimeout(Exception):
    """Raised when a GROBID request exceeds the configured timeout."""


def extract(
    filepath: str | Path,
    base_url: str,
    timeout: float = 180.0,
) -> Optional[ExtractedPaper]:
    """
    Send a PDF to GROBID's processFulltextDocument endpoint and return an
    ExtractedPaper, or None on failure.

    Raises GrobidTimeout if the read times out (so callers can retry with
    a larger budget). All other errors are converted to None and logged.

    Note: this does NOT check availability first. Call is_available() at
    the call site if you want a quick health probe before bulk imports.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    url = f"{base_url.rstrip('/')}/api/processFulltextDocument"
    try:
        with open(filepath, "rb") as f:
            files = {"input": (filepath.name, f, "application/pdf")}
            data = {
                # Ask GROBID to consolidate citations against CrossRef where possible
                "consolidateCitations": "1",
                # Include figures, tables and formulas in the output
                "includeRawCitations": "1",
                "segmentSentences": "0",
            }
            # Short connect timeout (we already health-probed), generous
            # read timeout for the actual ML inference.
            r = requests.post(
                url, files=files, data=data, timeout=(10.0, float(timeout)),
            )
        if r.status_code != 200:
            logger.warning(
                "GROBID returned %d for %s (%s)",
                r.status_code, filepath.name, r.text[:200],
            )
            return None
    except (requests.ReadTimeout, requests.ConnectTimeout, requests.Timeout) as e:
        # Surface timeouts so the caller can retry with a longer budget.
        raise GrobidTimeout(str(e)) from e
    except requests.RequestException as e:
        logger.warning("GROBID request failed for %s: %s", filepath.name, e)
        return None

    # Parse TEI-XML
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        logger.warning("GROBID returned malformed XML for %s: %s", filepath.name, e)
        return None

    title, abstract, doi = _parse_header(root)
    sections = _parse_sections(root)
    figs, tabs = _parse_figures(root)
    equations = _parse_equations(root)
    refs = _parse_references(root)

    full_text = _full_text_from_sections(sections)

    # Chunks for embedding: abstract + section paragraphs (each section's
    # text broken into ~300-word chunks). Keeps each chunk topically coherent.
    chunks: list[str] = []
    if abstract:
        chunks.append(abstract)
    for sec in sections:
        if not sec.text:
            continue
        for ch in _to_chunks(sec.text, chunk_size=300, overlap=60):
            chunks.append(ch)
    # cap to keep embedding work bounded
    chunks = chunks[:32]

    return ExtractedPaper(
        filepath   = str(filepath),
        paper_id   = _stable_id(str(filepath)),
        title      = title or "",
        abstract   = abstract or "",
        full_text  = full_text[:5000],
        chunks     = chunks,
        doi        = doi or "",
        sections   = sections,
        figures    = figs,
        tables     = tabs,
        equations  = equations,
        references = refs,
        parser     = "grobid",
    )
