"""
pdf_processor.py
Extract clean text from PDF files, split into meaningful chunks,
and pull basic metadata (title / authors when available).
"""

from __future__ import annotations
import re
import hashlib
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


def _guess_title(lines: list[str]) -> str:
    for line in lines[:30]:
        line = line.strip()
        if 20 < len(line) < 200 and not line.startswith('http') and not line.isupper():
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
            for page in pdf.pages[:40]:
                t = page.extract_text()
                if t:
                    pages_text.append(t)

        if not pages_text:
            return None

        full_raw   = '\n'.join(pages_text)
        full_clean = _clean(full_raw)
        lines      = [l for l in full_clean.split('\n') if l.strip()]

        title    = _guess_title(lines)
        abstract = _extract_abstract(full_clean)
        body     = full_clean[:5000]

        paras  = [p.strip() for p in full_clean.split('\n\n') if len(p.strip()) > 80]
        chunks = [abstract] + paras[:8]

        # Try to extract DOI from first 3 pages (most likely location)
        first_pages = '\n'.join(pages_text[:3])
        doi = _extract_doi(first_pages) or _extract_doi(full_clean)

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
