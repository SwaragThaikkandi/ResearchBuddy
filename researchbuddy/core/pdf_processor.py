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

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore


@dataclass
class ExtractedPaper:
    filepath: str
    paper_id: str          # SHA-1 of filepath (stable local ID)
    title: str
    abstract: str
    full_text: str         # first ~5000 chars of body text
    chunks: list[str] = field(default_factory=list)  # paragraphs for embedding
    doi: str = ""          # DOI extracted from PDF text (e.g. 10.1234/abc)


# ── helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
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
    raw = raw.strip()
    if not raw or len(raw) < 10 or len(raw) > 400:
        return ''
    lower = raw.lower()
    if any(x in lower for x in _META_TITLE_GARBAGE):
        return ''
    # Skip if it looks like a filename
    if raw.endswith(('.pdf', '.doc', '.docx', '.tex')):
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
    Extract the paper title by finding the largest text on page 1.

    In two-column papers (PNAS, Psych Review, J. Neuroscience, etc.),
    the title spans the full page width and uses a larger font than body
    text. This approach works regardless of column layout.
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

    # Title must be at least 1.25× larger than body text
    min_title_size = body_size * 1.25
    title_sizes = {s for s in size_counts if s >= min_title_size}

    if not title_sizes:
        return ''

    # Get chars at title font sizes in the top 45% of page (title region)
    page_h = page.height or 800
    title_chars = [
        c for c in chars
        if round(c.get('size', 0) * 2) / 2 in title_sizes
        and c.get('top', page_h) < page_h * 0.45
        and c.get('text', '').strip()
    ]

    if not title_chars:
        return ''

    # Sort by vertical position then horizontal
    title_chars.sort(key=lambda c: (round(c['top']), c.get('x0', 0)))

    # Build string from individual chars
    parts: list[str] = []
    prev_top = -99.0
    prev_x1 = 0.0
    for c in title_chars:
        top = c.get('top', 0)
        x0  = c.get('x0', 0)
        txt = c.get('text', '')
        if not txt:
            continue
        # New line (vertical gap > 3pt)
        if abs(top - prev_top) > 3:
            if parts:
                parts.append(' ')
        # Word gap (horizontal gap > 2pt)
        elif x0 - prev_x1 > 2:
            parts.append(' ')
        parts.append(txt)
        prev_top = top
        prev_x1  = c.get('x1', x0 + 5)

    title = re.sub(r'\s+', ' ', ''.join(parts)).strip()

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
        return m.group(1).strip(".,;:)(")
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

def extract_from_pdf(filepath: str | Path) -> Optional[ExtractedPaper]:
    """Parse a PDF and return an ExtractedPaper, or None on failure."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")

    filepath = Path(filepath)
    if not filepath.exists():
        return None

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
        full_clean = _clean(full_raw)
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
        )
    except Exception as e:
        print(f"[pdf_processor] Could not read {filepath.name}: {e}")
        return None


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
                doi = _extract_doi(text)

            return title, doi
    except Exception:
        return '', ''


def extract_from_folder(folder: str | Path) -> list[ExtractedPaper]:
    """Extract all PDFs in a folder (non-recursive)."""
    folder = Path(folder)
    papers = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"[pdf_processor] No PDF files found in {folder}")
        return papers

    for pdf in pdf_files:
        result = extract_from_pdf(pdf)
        if result:
            papers.append(result)
            print(f"  + {pdf.name}  [{len(result.chunks)} chunks]")
        else:
            print(f"  x {pdf.name}  (could not read)")
    return papers
