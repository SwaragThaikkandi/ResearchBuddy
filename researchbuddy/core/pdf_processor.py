"""
pdf_processor.py
Extract clean text from PDF files, split into meaningful chunks,
and pull basic metadata (title / authors when available).

Title resolution priority:
  1. PDF document-level metadata (most reliable for published papers)
  2. Font-size analysis on page 1 (largest text = title, works for 2-column)
  3. Heuristic first-line scanning (legacy fallback)
"""

from __future__ import annotations
import re
import hashlib
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A logical section parsed from the document (intro, methods, results, …)."""
    heading: str
    text: str
    # Classified section type. Heuristic mapping over the heading text.
    # One of: "abstract", "introduction", "related_work", "background",
    # "methods", "experiments", "results", "discussion", "conclusion",
    # "limitations", "acknowledgements", "appendix", "other"
    section_type: str = "other"
    # Original `n` attribute from <head n="3.2">3.2. Methods</head>, when given
    number: str = ""


@dataclass
class Figure:
    label: str             # e.g. "Figure 1"
    caption: str           # full caption text


@dataclass
class Table:
    label: str             # e.g. "Table 1"
    caption: str
    text: str = ""         # backwards-compat flattened cells
    rows: list[list[str]] = field(default_factory=list)   # structured rows


@dataclass
class CitationContext:
    """
    A single in-text citation marker located within the paper. Captures
    *where* a reference was cited (section type + sentence-ish snippet),
    not just *that* it was cited. This makes downstream tasks far richer:
      * "show me papers that cite Pearl 2009 in their methods"
      * citation classification (supportive vs contrastive vs methodological)
      * co-citation by paragraph
    """
    ref_index: str             # GROBID's #b0, #b1, ... target id
    section_type: str          # the containing section's classified type
    section_heading: str       # the literal heading text
    snippet: str               # paragraph text around the citation marker


@dataclass
class Reference:
    raw: str               # the as-printed reference string (best effort)
    title: str = ""
    doi: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    # Where in the citing paper this reference was cited
    contexts: list[CitationContext] = field(default_factory=list)


@dataclass
class ExtractedPaper:
    filepath: str
    paper_id: str          # SHA-1 of filepath (stable local ID)
    title: str
    abstract: str
    full_text: str         # first ~5000 chars of body text
    chunks: list[str] = field(default_factory=list)  # paragraphs for embedding
    doi: str = ""          # DOI extracted from PDF text (e.g. 10.1234/abc)

    # ── Structured fields (populated by GROBID; empty when pdfplumber falls back)
    sections:  list[Section]   = field(default_factory=list)
    figures:   list[Figure]    = field(default_factory=list)
    tables:    list[Table]     = field(default_factory=list)
    equations: list[str]       = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    parser:    str             = "pdfplumber"   # "grobid" | "pdfplumber"


# ── Ligature / Unicode normalisation ─────────────────────────────────────────

def _fix_ligatures(text: str) -> str:
    """
    Replace Unicode ligature characters with ASCII equivalents.
    MUST run before _clean() strips all non-ASCII, otherwise the
    characters vanish and leave broken words (e.g. 'Uni ed' for 'Unified').
    """
    text = text.replace('\ufb01', 'fi')     # fi ligature
    text = text.replace('\ufb02', 'fl')     # fl ligature
    text = text.replace('\ufb00', 'ff')     # ff ligature
    text = text.replace('\ufb03', 'ffi')    # ffi ligature
    text = text.replace('\ufb04', 'ffl')    # ffl ligature
    text = text.replace('\u2013', '-')      # en-dash
    text = text.replace('\u2014', '-')      # em-dash
    text = text.replace('\u2018', "'")      # left single quote
    text = text.replace('\u2019', "'")      # right single quote
    text = text.replace('\u201c', '"')      # left double quote
    text = text.replace('\u201d', '"')      # right double quote
    return text


# ── helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = _fix_ligatures(text)             # ← ligatures FIRST
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


# ── PDF metadata extraction ──────────────────────────────────────────────────

# Garbage values commonly found in PDF Title metadata
_META_TITLE_GARBAGE = [
    'microsoft', 'untitled', '.pdf', '.doc', '.tex', '.dvi',
    'latex', 'pdftex', 'tex output', 'acrobat', 'word',
    'powerpoint', 'openoffice', 'libreoffice', 'preview',
    'scanned', 'unknown', 'title', 'full text',
]


def _metadata_title(pdf) -> str:
    """Extract title from PDF document properties (embedded metadata)."""
    meta = pdf.metadata or {}
    raw = meta.get('Title', '') or meta.get('title', '') or ''
    raw = _fix_ligatures(raw.strip())
    if not raw or len(raw) < 10 or len(raw) > 400:
        return ''
    lower = raw.lower()
    if any(x in lower for x in _META_TITLE_GARBAGE):
        return ''
    # Skip if it looks like a filename
    if raw.endswith(('.pdf', '.doc', '.docx', '.tex')):
        return ''
    # Skip production IDs like "npgrj_nn_1790 1432..1438"
    alpha = sum(c.isalpha() for c in raw)
    if alpha < len(raw) * 0.5:
        return ''
    # Skip if mostly underscores/digits (internal filenames)
    if raw.count('_') > 2:
        return ''
    return raw[:300]


def _metadata_doi(pdf) -> str:
    """Extract DOI from PDF document properties."""
    meta = pdf.metadata or {}
    for key in ('doi', 'DOI', 'Subject', 'subject', 'Keywords', 'keywords',
                'Description', 'description', 'WPS-ARTICLEDOI'):
        val = meta.get(key, '') or ''
        if '10.' in val:
            doi = _extract_doi(val)
            if doi:
                return doi
    return ''


# ── Font-size based title extraction (handles two-column layouts) ─────────────

def _title_by_font_size(page) -> str:
    """
    Extract the paper title by finding the LARGEST font on page 1.

    In two-column papers (PNAS, Psych Review, J. Neuroscience, etc.),
    the title spans the full page width and uses a noticeably larger
    font than both body text and journal-header text.

    Only uses the single largest font size — avoids picking up the
    journal name which is typically an intermediate size.

    Ligature glyphs (fi, fl, ff) are often at slightly offset vertical
    positions (±5pt); we group chars into lines using a 6pt tolerance
    to keep them with their parent word.
    """
    try:
        chars = page.chars
    except Exception:
        return ''
    if not chars or len(chars) < 20:
        return ''

    # Get font sizes (round to nearest 0.5pt for grouping)
    sizes = [round(c.get('size', 0) * 2) / 2
             for c in chars if c.get('size', 0) > 4]
    if not sizes:
        return ''

    size_counts = Counter(sizes)
    # Most common size = body text
    body_size = size_counts.most_common(1)[0][0]
    # Single largest font size on the page
    max_size = max(size_counts.keys())

    # Title must be noticeably larger than body (≥1.3× body)
    if max_size < body_size * 1.3:
        return ''

    # Get chars at the MAX font size only, in the top 50% of page
    page_h = page.height or 800
    title_chars = [
        c for c in chars
        if round(c.get('size', 0) * 2) / 2 == max_size
        and c.get('top', page_h) < page_h * 0.50
        and c.get('text', '')
    ]

    if not title_chars:
        return ''

    # ── Group chars into lines using 6pt vertical tolerance ──────────
    # Ligature glyphs often have a ±5pt vertical offset from other chars
    # on the same visual line. Grouping by proximity keeps them together.
    title_chars.sort(key=lambda c: c.get('top', 0))
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    line_top = -99.0
    for c in title_chars:
        top = c.get('top', 0)
        if current_line and abs(top - line_top) > 6:
            lines.append(current_line)
            current_line = [c]
            line_top = top
        else:
            current_line.append(c)
            if not current_line[1:]:   # first char sets the line_top
                line_top = top
    if current_line:
        lines.append(current_line)

    # ── Build string: sort each line by x0, insert spaces at gaps ────
    parts: list[str] = []
    for line in lines:
        line.sort(key=lambda c: c.get('x0', 0))
        if parts:
            parts.append(' ')  # space between lines
        for i, c in enumerate(line):
            txt = c.get('text', '')
            if not txt:
                continue
            if i > 0:
                prev_x1 = line[i - 1].get('x1', 0)
                x0 = c.get('x0', 0)
                if x0 - prev_x1 > 2:
                    parts.append(' ')
            parts.append(txt)

    # Apply ligature fix + collapse whitespace
    title = _fix_ligatures(''.join(parts))
    title = re.sub(r'\s+', ' ', title).strip()

    # Validate: must be reasonable length and mostly alphabetic
    if len(title) < 15 or len(title) > 300:
        return ''
    alpha = sum(c.isalpha() for c in title)
    if alpha < len(title) * 0.5:
        return ''
    return title


# ── Title / journal-header detection ──────────────────────────────────────────

# Patterns that indicate a line is journal/publisher metadata, not a paper title
_TITLE_SKIP = [
    re.compile(r'available\s+online', re.I),
    re.compile(r'www\.\S+\.\w{2,}', re.I),
    re.compile(r'https?://'),
    re.compile(r'article\s+in\s+press', re.I),
    re.compile(r'Vol\.\s*\d+|Volume\s+\d+', re.I),
    re.compile(r'\bNo\.\s*\d+', re.I),
    re.compile(r'\bpp\.\s*\d+', re.I),
    re.compile(r'doi\s*[:=]\s*10\.', re.I),
    re.compile(r'^10\.\d{4,9}/'),
    re.compile(r'received\s+\d|accepted\s+\d', re.I),
    re.compile(r'[©®™]'),
    re.compile(r'copyright\s*\d{4}', re.I),
    re.compile(r'this\s+article\s+was\s+download', re.I),
    re.compile(r'^\d{4}[-~/]\d{3}[\dxX]'),               # ISSN
    re.compile(r'\$\d+\.\d{2}'),                          # price
    re.compile(r'elsevier|pergamon|springer|wiley|academic\s+press', re.I),
    re.compile(r'contents\s+lists?\s+available', re.I),
    re.compile(r'journal\s+homepage', re.I),
    re.compile(r'(?:January|February|March|April|May|June|July|August|'
               r'September|October|November|December)\s+\d{1,2},?\s*\d{4}', re.I),
    re.compile(r'^\d+\s*$'),                              # page number
    re.compile(r'Proc\.\s*Natl', re.I),
]

# "JournalName91(2019)1437" — journal + volume(year)pages
_JOURNAL_HEADER_RE = re.compile(
    r'^[A-Za-z][A-Za-z\s,&]+\d{1,4}\s*\(\d{4}\)\s*\d+'
)


def _is_journal_line(line: str) -> bool:
    """Return True if line is journal/publisher metadata, not paper content."""
    s = line.strip()
    if not s:
        return True
    for pat in _TITLE_SKIP:
        if pat.search(s):
            return True
    if _JOURNAL_HEADER_RE.match(s):
        return True
    # Line with very few alpha chars → likely codes/numbers
    alpha = sum(c.isalpha() for c in s)
    if len(s) < 60 and alpha < len(s) * 0.5:
        return True
    return False


def _guess_title(lines: list[str]) -> str:
    for line in lines[:30]:
        line = line.strip()
        if len(line) < 15 or len(line) > 250:
            continue
        if line.startswith('http'):
            continue
        if _is_journal_line(line):
            continue
        # Skip lines that are ALL-CAPS (section headers like "ARTICLES")
        if line.isupper() and len(line) < 40:
            continue
        return line
    # Fallback: return first line with reasonable length
    for line in lines[:10]:
        line = line.strip()
        if 10 < len(line) < 250:
            return line
    return lines[0].strip() if lines else "Unknown Title"


def _extract_abstract(text: str) -> str:
    m = re.search(
        r'abstract[:\s\n]+(.*?)(?=\n\s*(?:introduction|keywords|1[\.\s]|background))',
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        return _clean(m.group(1))[:1200]
    return _clean(text)[:500]


def _extract_doi(text: str) -> str:
    """Return the first DOI found in text, stripped of trailing punctuation."""
    m = re.search(r'\b(10\.\d{4,9}/[^\s"<>\[\]{}|\\^`]+)', text)
    if m:
        doi = m.group(1).strip(".,;:)(")
        # Strip supplementary-material suffixes like ".supp", ".Supplemental"
        doi = re.sub(r'\.supp(?:lemental)?$', '', doi, flags=re.I)
        return doi
    return ""


def _to_chunks(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = ' '.join(words[i: i + chunk_size])
        if len(chunk) > 60:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _stable_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:12]


# ── public API ─────────────────────────────────────────────────────────────────

# Module-level cache of the last GROBID availability probe so we don't pay the
# health-check round-trip on every PDF in a bulk import.
_grobid_probe_cache: dict[str, tuple[float, bool]] = {}
_GROBID_PROBE_TTL = 300.0   # seconds


def _grobid_available(base_url: str) -> bool:
    """
    Cached availability probe — refreshed every 5 minutes. The first time
    a probe succeeds, we also send a warmup request so the first real PDF
    isn't slowed by GROBID's one-off model-loading cost (~30s on CPU).
    """
    import time
    from researchbuddy.core import grobid_client

    now = time.monotonic()
    cached = _grobid_probe_cache.get(base_url)
    if cached is not None:
        ts, ok = cached
        if (now - ts) < _GROBID_PROBE_TTL:
            return ok
    ok = grobid_client.is_available(base_url)
    _grobid_probe_cache[base_url] = (now, ok)
    if ok:
        logger.info("GROBID is reachable at %s — using it for PDF parsing.", base_url)
        try:
            from researchbuddy.config import GROBID_WARMUP
        except ImportError:
            GROBID_WARMUP = True
        if GROBID_WARMUP:
            logger.info("[GROBID] Loading models (one-time warmup, ~30s) ...")
            grobid_client.warmup(base_url)
    else:
        logger.info("GROBID not reachable at %s — falling back to pdfplumber.", base_url)
    return ok


def _extract_via_pdfplumber(filepath: Path) -> Optional[ExtractedPaper]:
    """Original pdfplumber-based extraction. Used when GROBID is unavailable."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")
    try:
        pages_text: list[str] = []
        with pdfplumber.open(filepath) as pdf:
            # ── Title: metadata → font-size → line heuristic ─────────────
            meta_title = _metadata_title(pdf)
            font_title = ''

            for page in pdf.pages[:40]:
                t = page.extract_text()
                if t:
                    pages_text.append(t)

            if not meta_title and pdf.pages:
                font_title = _title_by_font_size(pdf.pages[0])

            # ── DOI: metadata → first pages text ─────────────────────────
            meta_doi = _metadata_doi(pdf)

        if not pages_text:
            return None

        full_raw   = '\n'.join(pages_text)
        full_clean = _clean(full_raw)       # _clean now calls _fix_ligatures
        lines      = [l for l in full_clean.split('\n') if l.strip()]

        # Title priority: PDF metadata > font-size > line heuristic
        title    = meta_title or font_title or _guess_title(lines)
        abstract = _extract_abstract(full_clean)
        body     = full_clean[:5000]

        paras  = [p.strip() for p in full_clean.split('\n\n') if len(p.strip()) > 80]
        chunks = [abstract] + paras[:8]

        # DOI: metadata > first 3 pages text > full text
        first_pages = '\n'.join(pages_text[:3])
        doi = meta_doi or _extract_doi(first_pages) or _extract_doi(full_clean)

        return ExtractedPaper(
            filepath  = str(filepath),
            paper_id  = _stable_id(str(filepath)),
            title     = title[:200],
            abstract  = abstract,
            full_text = body,
            chunks    = chunks,
            doi       = doi,
            parser    = "pdfplumber",
        )
    except Exception as e:
        logger.warning("Could not read %s: %s", filepath.name, e)
        return None


def extract_from_pdf(filepath: str | Path) -> Optional[ExtractedPaper]:
    """
    Parse a PDF and return an ExtractedPaper, or None on failure.

    Tries GROBID first (when configured and reachable) for high-quality
    structured extraction. Falls back to pdfplumber otherwise.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        from researchbuddy.config import GROBID_ENABLED, GROBID_URL, GROBID_TIMEOUT
    except ImportError:
        GROBID_ENABLED, GROBID_URL, GROBID_TIMEOUT = False, "", 180

    if GROBID_ENABLED and GROBID_URL and _grobid_available(GROBID_URL):
        from researchbuddy.core import grobid_client
        from researchbuddy.core.grobid_client import GrobidTimeout
        # Try once at the configured timeout; if it timed out, retry once
        # with double the budget. After the second failure, fall back to
        # pdfplumber. This handles cold-start spikes and unusually long PDFs
        # without blocking forever on a stuck GROBID.
        for attempt, t in enumerate((float(GROBID_TIMEOUT), float(GROBID_TIMEOUT) * 2.0), 1):
            try:
                ep = grobid_client.extract(filepath, base_url=GROBID_URL, timeout=t)
                if ep is not None and (ep.title or ep.full_text):
                    return ep
                # Empty result usually means a non-PDF or a corrupt file —
                # don't retry, just fall back.
                logger.info(
                    "GROBID returned an empty result for %s — falling back to pdfplumber.",
                    filepath.name,
                )
                break
            except GrobidTimeout:
                if attempt == 1:
                    logger.info(
                        "GROBID timed out on %s after %.0fs — retrying with %.0fs.",
                        filepath.name, t, t * 2.0,
                    )
                    continue
                logger.warning(
                    "GROBID timed out twice on %s — falling back to pdfplumber.",
                    filepath.name,
                )
                break
            except Exception as e:
                logger.warning(
                    "GROBID extraction failed for %s (%s) — falling back to pdfplumber.",
                    filepath.name, e,
                )
                break

    return _extract_via_pdfplumber(filepath)


def reextract_title_doi(filepath: str | Path) -> tuple[str, str]:
    """
    Re-extract title and DOI from a PDF using metadata + font analysis.
    Used to retroactively fix garbage titles in existing pickles.
    Returns (title, doi) — either or both may be empty.
    """
    filepath = Path(filepath)
    if not filepath.exists() or pdfplumber is None:
        return '', ''

    try:
        with pdfplumber.open(filepath) as pdf:
            # 1. PDF metadata title
            title = _metadata_title(pdf)

            # 2. Font-size analysis on page 1
            if not title and pdf.pages:
                title = _title_by_font_size(pdf.pages[0])

            # 3. DOI from metadata
            doi = _metadata_doi(pdf)

            # 4. DOI from first page text if not in metadata
            if not doi and pdf.pages:
                text = pdf.pages[0].extract_text() or ''
                doi = _extract_doi(_fix_ligatures(text))

            return title, doi
    except Exception:
        return '', ''


def extract_from_folder(folder: str | Path) -> list[ExtractedPaper]:
    """Extract all PDFs in a folder (non-recursive)."""
    folder = Path(folder)
    papers = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s", folder)
        return papers

    for pdf in pdf_files:
        result = extract_from_pdf(pdf)
        if result:
            papers.append(result)
            logger.info("  + %s  [%d chunks]", pdf.name, len(result.chunks))
        else:
            logger.warning("  x %s  (could not read)", pdf.name)
    return papers
