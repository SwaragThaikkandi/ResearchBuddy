"""
arguer.py — Creative argumentation engine ("Creative Cortex").

Generates structured argumentative paragraphs that synthesize the literature by
combining:
  - Citation relationship types (supports / contradicts / mentions) from the
    context-aware semantic network
  - Graph topology (which papers are connected and why)
  - Temporal structure (how the field has evolved)

The Arguer learns from user feedback via a StyleProfile, which tracks which
argument types produce high ratings and biases future generation accordingly.

Argument types (inspired by scientific discourse patterns):
  CONVERGENCE : Multiple papers independently support the same conclusion
  TENSION     : Papers present conflicting evidence or competing interpretations
  EVOLUTION   : The field's understanding has changed over time
  SYNTHESIS   : Combining insights from different methodological angles
  GAP         : Identifying what is still missing or underexplored

User rates each paragraph on:
  - Correctness (1–10): factual accuracy of the argument
  - Usefulness  (1–10): how helpful it is for the user's research
"""

from __future__ import annotations

import re
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta
    from researchbuddy.core.reasoner import QueryResult

# ── Argument type constants ────────────────────────────────────────────────────

ARG_CONVERGENCE = "convergence"
ARG_TENSION     = "tension"
ARG_EVOLUTION   = "evolution"
ARG_SYNTHESIS   = "synthesis"
ARG_GAP         = "gap"

_ALL_ARG_TYPES = [ARG_CONVERGENCE, ARG_TENSION, ARG_EVOLUTION, ARG_SYNTHESIS, ARG_GAP]


# ── Persistent data classes (stored in graph pickle) ──────────────────────────

@dataclass
class ArgumentInteraction:
    """Persisted record of one generated argument and its user ratings."""
    argument_type : str
    argument_text : str
    paper_ids     : list[str]
    query         : str
    correctness   : float       # 1–10 (0 = unrated)
    usefulness    : float       # 1–10 (0 = unrated)
    timestamp     : float = field(default_factory=time.time)


@dataclass
class StyleProfile:
    """
    Tracks which argument types perform well for this user.

    type_weights maps ARG_* → weight in [0.05, 2.0].
      weight = 1.0  → neutral (no feedback yet)
      weight > 1.0  → user found this type useful/correct
      weight < 1.0  → user found this type less useful/correct

    Updated via exponential moving average after each rated interaction
    (learning rate controlled by ARGUER_STYLE_LR in config).
    """
    type_weights       : dict[str, float] = field(
        default_factory=lambda: {t: 1.0 for t in _ALL_ARG_TYPES}
    )
    total_interactions : int   = 0
    avg_correctness    : float = 5.0
    avg_usefulness     : float = 5.0

    def update(
        self,
        arg_type   : str,
        correctness: float,
        usefulness : float,
        lr         : float = 0.20,
    ):
        """EMA update: combined rating shifts type weight toward [0, 2]."""
        combined = (correctness + usefulness) / 20.0    # normalise to [0, 1]
        old_w = self.type_weights.get(arg_type, 1.0)
        # Target: combined * 2 (so 10/10 → weight 2.0, 0/0 → weight 0.0)
        self.type_weights[arg_type] = max(0.05, old_w * (1 - lr) + combined * 2 * lr)

        n = self.total_interactions + 1
        self.total_interactions = n
        self.avg_correctness = (self.avg_correctness * (n - 1) + correctness) / n
        self.avg_usefulness  = (self.avg_usefulness  * (n - 1) + usefulness ) / n

    def weighted_sample(self, available: list[str], n: int = 3) -> list[str]:
        """
        Sample up to n argument types from available, weighted by performance.
        Ensures diversity (no type repeated).
        """
        if not available:
            return []
        weights = [max(0.05, self.type_weights.get(t, 1.0)) for t in available]
        total   = sum(weights)
        probs   = [w / total for w in weights]

        chosen: list[str] = []
        pool = list(zip(available, probs))

        for _ in range(min(n, len(available))):
            if not pool:
                break
            r   = random.random()
            cum = 0.0
            for idx, (t, p) in enumerate(pool):
                cum += p
                if r <= cum:
                    chosen.append(t)
                    pool.pop(idx)
                    # Re-normalise remaining probabilities
                    rem_total = sum(pp for _, pp in pool) or 1.0
                    pool = [(tt, pp / rem_total) for tt, pp in pool]
                    break

        return chosen


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ArgumentParagraph:
    """One generated argument paragraph with metadata."""
    arg_type        : str          # ARG_* constant
    arg_type_label  : str          # human-readable label
    text            : str          # the paragraph itself
    paper_ids       : list[str]    # IDs of papers referenced
    paper_refs      : list[str]    # citation strings ("Smith et al. (2020)")
    connection_type : str          # "supports" / "contradicts" / "mixed" / "gap"
    explanation     : str          # brief note on why this argument type was chosen


# ── PDF metadata cleaning ──────────────────────────────────────────────────────
#
# Real-world abstracts from pdfplumber are heavily polluted with journal headers,
# publisher boilerplate, ISSNs, URLs, etc.  The regexes below strip all of that
# before the text is used in argument templates.

_METADATA_STRIP: list[re.Pattern] = [
    # Journal+vol(year)pages: "CognitivePsychology62(2011)193-222"
    re.compile(r'[A-Za-z]+\d{1,4}\s*\(\d{4}\)\s*[\d\-–]+\s*'),
    # Journal vol (year) pages: "Neuropsychologia 123 (2019) 5-18"
    re.compile(r'[A-Za-z]+\s+\d{1,4}\s*\(\d{4}\)\s*[\d\-–]+\s*'),
    # "Contents lists available at ScienceDirect"
    re.compile(r'[Cc]ontents?\s*lists?\s*available\s*at\s*\S+', re.I),
    # "journal homepage: www..."
    re.compile(r'journal\s*homepage\s*:?\s*\S+', re.I),
    # "Available online at www..."
    re.compile(r'Available\s+online\s+at\s+\S+', re.I),
    # URLs
    re.compile(r'https?://\S+'),
    re.compile(r'www\.\S+'),
    # DOI strings
    re.compile(r'doi\s*:?\s*10\.\S+', re.I),
    # Email addresses
    re.compile(r'E-mail\s*:\s*\S+', re.I),
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\S+'),
    # ISSN: "0270-6474/82/..." or "0270~6474"
    re.compile(r'\d{4}[-~/]\d{3}[\dxX]\S*'),
    # Price: "$02.00/O"
    re.compile(r'\$\d+\.\d{2}\S*'),
    # "Vol. 81, No. 5, 338-364"
    re.compile(r'Vol\.\s*\d+\s*,?\s*No\.\s*\d+[^.]*', re.I),
    # "pp. 338-364"
    re.compile(r'pp\.\s*\d+\s*[-–]\s*\d+', re.I),
    # Publisher names
    re.compile(
        r'Elsevier\s*(B\.V\.|Science|Ltd)?|Pergamon(\s+Press)?|'
        r'Academic\s+Press|Springer[-\s]Verlag|'
        r'Cambridge\s+University\s+Press|Wiley[-\s]Liss',
        re.I,
    ),
    # Copyright line
    re.compile(r'©\s*\d{4}[^.]*\.?'),
    re.compile(r'Copyright\s+.{0,40}Society\s+\w+', re.I),
    # "Printed in U.S.A."
    re.compile(r'Printed\s+in\s+\S+', re.I),
    # "ARTICLE IN PRESS"
    re.compile(r'\bARTICLE\s+IN\s+PRESS\b', re.I),
    # Section slugs: "Behavioral/Systems/Cognitive"
    re.compile(r'Behavioral/\w+/\w+'),
    # Standalone ALL-CAPS labels (≥5 chars): "ABSTRACT", "REVIEW" etc.
    re.compile(r'\b[A-Z]{5,}\b'),
    # Month-date-year in metadata context
    re.compile(
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s*\d{4}',
        re.I,
    ),
    # Preamble journal names: "The Journal of Neuroscience, …"
    re.compile(
        r'^(?:The\s+)?(?:Journal\s+of\s+[\w\s,]+|'
        r'Cognitive\s+Psychology|Neuropsychologia|'
        r'Psychological\s+(?:Review|Bulletin|Science))'
        r'[\s,]*(?:\d{4}|Vol\.)',
        re.I,
    ),
    # Squished-together metadata blobs (pdfplumber merges words)
    re.compile(r'Contentslistsavailable\S*', re.I),
    re.compile(r'journalhomepage\S*', re.I),
    # Spaced-out section headers: "a r t i c l e   i n f o", "a b s t r a c t"
    re.compile(r'a\s+r\s+t\s+i\s+c\s+l\s+e\s+i\s+n\s+f\s+o', re.I),
    re.compile(r'a\s+b\s+s\s+t\s+r\s+a\s+c\s+t', re.I),
    re.compile(r'k\s+e\s+y\s+w\s+o\s+r\s+d\s+s', re.I),
    # "Email:" without hyphen (pdfplumber sometimes strips hyphen from "E-mail")
    re.compile(r'Email\s*:\s*\S+', re.I),
    # Bracket placeholders: "[Your Email]", "[University Name]", "[Last Name]"
    re.compile(r'\[[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\]'),
    # "Author Note" / "Correspondence" blocks (stop at sentence boundary)
    re.compile(r'Author\s+Note\b[^.]{0,300}\.?', re.I),
    re.compile(r'Correspondence\s+concerning\b[^.]{0,300}\.?', re.I),
    # Running header labels: "THERMODYNAMIC GEOMETRY OF CHOICE 1" etc.
    re.compile(r'(?:[A-Z]+\s+){3,}\d+\b'),
    # "published: 28 February 2022" style dates
    re.compile(r'published\s*:\s*\d+\s*\w+\s*\d{4}', re.I),
    # Frontiers-style header: "fnins-16-827888 February22,2022 Time:15:2 #1"
    re.compile(r'[a-z]+-\d+-\d+\s+\w+\d+,?\d*\s+Time:\d+:\d+\s*#\d+', re.I),
    re.compile(r'\bORIGINALRESEARCH\b|\bOriginal\s+Research\b', re.I),
    # Stray "doi:10.xxxx/..." patterns
    re.compile(r'doi:\s*10\.\d{4,}/\S+', re.I),
    # "rspa.royalsocietypublishing.org" and similar
    re.compile(r'\w+\.\w+publishing\.\w+', re.I),
    # "Cite this article:" blocks
    re.compile(r'Cite\s*this\s*article\s*:.*', re.I),
]


def _clean_for_display(text: str) -> str:
    """Aggressively strip PDF metadata noise; return readable prose only."""
    if not text:
        return ""
    t = text
    # Unicode ligatures → ASCII
    for old, new in [('\ufb01', 'fi'), ('\ufb02', 'fl'), ('\ufb00', 'ff'),
                     ('\ufb03', 'ffi'), ('\ufb04', 'ffl'),
                     ('\u2013', '-'), ('\u2014', '-'),
                     ('\u2018', "'"), ('\u2019', "'"),
                     ('\u201c', '"'), ('\u201d', '"')]:
        t = t.replace(old, new)
    for pat in _METADATA_STRIP:
        t = pat.sub(' ', t)
    # Remove pdfplumber (cid:N) artefacts
    t = re.sub(r'\(cid:\d+\)', '', t)
    # Squished-together dates: "Accepted7December2007", "Received21October2004"
    t = re.sub(r'(?:Accepted|Received)\d+\w+\d{4}', ' ', t, flags=re.I)
    t = re.sub(r'(?:Articlehistory|Article\s*history)\s*:', ' ', t, flags=re.I)
    # Remove lone "Abstract" header
    t = re.sub(r'\bAbstract\b', ' ', t)
    # Remove long lowercase blobs (≥16 chars, garbled PDF like
    # "departmentofpsychology" or "contentslistsavailableatsciencedirect")
    t = re.sub(r'\b[a-z]{16,}\b', ' ', t)
    # Squished text with hyphens: "wepresentananalyticsolution...accumula-tors"
    t = re.sub(r'\b[a-z][a-z-]{20,}\b', ' ', t)
    # Squished CamelCase (lowercase→uppercase transition inside a word, >10 chars)
    # Catches "MariusUsher", "SchoolofPsychology", "UniversityofNewcastle"
    t = re.sub(r'\b\w*[a-z][A-Z]\w{4,}\b', ' ', t)
    # CamelCase with apostrophe: "JosephO'Neill" → partial remnant "josepho'"
    t = re.sub(r"\b[A-Z][a-z]+[A-Z][a-z]*'[A-Z][a-z]+\b", ' ', t)
    # Lone remnants: "josepho'" (lowercase + apostrophe at end of word)
    t = re.sub(r"\b\w+[a-z][A-Z]\w*'\w*\b", ' ', t)
    # Comma-separated short codes: ",PA,USA", ",NSW2308,Australia"
    t = re.sub(r',\s*[A-Z]{2,5}\d*\s*,\s*[A-Za-z]+', ' ', t)
    # Lone comma-prefixed state/country codes: ",USA", ",Australia"
    t = re.sub(r',\s*(?:USA|UK|Australia|Canada|Germany|France|China|Japan)\b',
               ' ', t, flags=re.I)
    # Stray author markers: "Scott D.", "Andrew Heathcote" after title
    # (We keep these — they don't break readability)
    # Collapse whitespace
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t


def _is_readable(text: str, min_words: int = 5) -> bool:
    """Return True if *text* looks like a real English sentence, not garbled metadata."""
    if not text or len(text) < 20:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # Mostly-alphabetic check
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.55:
        return False
    # Obvious metadata markers
    tl = text.lower()
    for marker in ('sciencedirect', 'elsevier', 'homepage', 'doi.org',
                    'doi:', 'issn', 'copyright', 'downloaded', 'contents list',
                    'journal homepage', 'available online', 'article in press',
                    'printed in u.s.a', 'email:', '[your email]',
                    'correspondence concerning', 'author note',
                    'rspa.royalsociety', 'citethisarticle'):
        if marker in tl:
            return False
    # Too many CamelCase blobs (garbled PDF text like "BirkbeckCollege")
    camel = sum(1 for w in words
                if len(w) > 12 and any(c.isupper() for c in w[1:]))
    if camel > max(1, len(words) * 0.25):
        return False
    # Long lowercase blobs without spaces ("departmentofpsychology")
    long_garbled = sum(1 for w in words if len(w) > 15 and w.islower())
    if long_garbled >= 1:
        return False
    # Comma-separated camelcase ("UniversityofLondon,Callaghan,NSW2308")
    if any(len(w) > 10 and ',' in w for w in words):
        return False
    return True


def _readable_sentences(text: str) -> list[str]:
    """Split text into sentences and keep only the readable ones."""
    cleaned = _clean_for_display(text)
    if not cleaned:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
    return [s for s in sentences if _is_readable(s)]


# ── Year / author extraction from noisy abstracts ─────────────────────────────

def _try_extract_year(abstract: str) -> Optional[int]:
    """Try to pull a publication year from the metadata noise in the abstract."""
    if not abstract:
        return None
    # Year in parentheses — most reliable (journal header)
    m = re.search(r'\((\d{4})\)', abstract[:250])
    if m:
        y = int(m.group(1))
        if 1950 <= y <= 2030:
            return y
    # Any plausible 4-digit year
    m = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', abstract[:300])
    if m:
        return int(m.group(1))
    return None


def _try_extract_first_author(abstract: str, title: str) -> Optional[str]:
    """
    Heuristic: locate the paper title inside the abstract, then look for a
    name-like pattern immediately after it.

    Typical PDF layout:
      "…JournalHeader… <Title> <FirstName M. LastName>, <next author>…"
    """
    if not abstract or not title:
        return None
    # Use a 40-char prefix of the title for a fuzzy search
    prefix = title[:40].lower().strip()
    abs_lower = abstract.lower()
    idx = abs_lower.find(prefix)
    if idx < 0:
        return None
    # Skip past the title
    after = abstract[idx + len(prefix):]
    if not after:
        return None
    # Match patterns like "Scott D. Brown" or "Darryl W. Schneider"
    m = re.match(
        r'[^A-Z]{0,5}'                         # minor junk after title
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+'     # first name (+ optional middle)
        r'[A-Z][a-z]{2,})',                     # surname
        after,
    )
    if m:
        parts = m.group(1).split()
        return parts[-1]   # surname
    return None


# ── Text extraction helpers ────────────────────────────────────────────────────

def _clean_title_for_claim(title: str) -> str:
    """Clean a paper title for use as a fallback claim/finding text."""
    t = title.strip()
    # Strip running header prefixes: "THERMODYNAMIC GEOMETRY OF CHOICE 1 The..."
    t = re.sub(r'^(?:[A-Z]+\s+){3,}\d+\s*', '', t).strip()
    # Strip journal name prefixes that leak into the title
    # Greedy: "Journal of Mathematical Psychology" → strip all capitalized words
    t = re.sub(
        r'^(?:The\s+)?Journal\s+of\s+(?:[A-Z]\w+\s+){1,4}',
        '', t, flags=re.I,
    ).strip()
    t = re.sub(
        r'^(?:Cognitive\s+Psychology|Neuropsychologia|'
        r'Psychological\s+(?:Review|Bulletin|Science)|'
        r'Psychology|Neuroscience|Mathematics|Physics|'
        r'Annual\s+Review\s+of\s+\w+)\s*',
        '', t, flags=re.I,
    ).strip()
    # Strip leading articles
    t = re.sub(r'^(?:The|A|An)\s+', '', t, flags=re.I).strip()
    # Truncate at author names / affiliations that leak into the title
    # Pattern: "Swarag" or "Department of" or "[Last Name]" etc.
    t = re.sub(r'\s+(?:Department|Division|Institute|School)\s+of\b.*', '', t, flags=re.I)
    t = re.sub(r'\s+\[.*', '', t)  # bracket placeholders
    t = re.sub(r'\s+Autho\b.*', '', t, flags=re.I)  # truncated "Author"
    # Remove CamelCase blobs
    t = re.sub(r'\b\w*[a-z][A-Z]\w{4,}\b', ' ', t)
    # Remove long garbled words
    t = re.sub(r'\b[a-z]{16,}\b', ' ', t)
    # Collapse whitespace
    t = re.sub(r'\s{2,}', ' ', t).strip()
    # Truncate to reasonable claim length
    if len(t) > 90:
        t = t[:90].rsplit(' ', 1)[0]
    return t


def _claim_looks_clean(text: str) -> bool:
    """Return True if extracted claim text looks like real prose, not garbled PDF."""
    if not text or len(text) < 15:
        return False
    words = text.split()
    # Reject if any word is a squished blob (>25 chars, all lowercase)
    if any(len(w) > 25 and w.replace('-', '').isalpha() for w in words):
        return False
    # Reject if too many commas relative to words (author/affiliation lists)
    comma_count = text.count(',')
    if comma_count > max(2, len(words) * 0.3):
        return False
    # Reject if contains email or bracket placeholders
    if re.search(r'email|@|\[your|\[last\s+name\]|\[university', text, re.I):
        return False
    # Reject if contains typical affiliation patterns
    if re.search(r'(?:university|department|princeton|callaghan|newcastle|'
                 r'institute|division|semel|los\s+angeles)',
                 text, re.I):
        return False
    # Reject if contains stray author-name patterns: "Tyler , , Philip"
    if re.search(r'[A-Z][a-z]+\s*,\s*,', text):
        return False
    # Reject if starts with the generic fallback
    if text.startswith("investigates key aspects"):
        return False
    # Reject if starts with or contains a journal name (abstract header leaked)
    if re.match(
        r'(?:cognitive\s+psychology|neuropsychologia|psychological\s+'
        r'(?:review|bulletin|science)|journal\s+of|neuroscience|'
        r'annual\s+review|quarterly\s+journal|psychonomic\s+bulletin|'
        r'biology\s+&?\s*philosophy)',
        text, re.I,
    ):
        return False
    # Journal-name patterns anywhere in the text
    if re.search(r'\bjournal\s+of\s+\w+', text, re.I):
        return False
    if re.search(r'\b(?:quarterly|bulletin|proceedings)\s+(?:of|and)\b',
                 text, re.I):
        return False
    # Reject if ends with an author-name fragment (first name or initial)
    if re.search(r'\b[a-z]{3,10}\s+[a-z]\.?\s*$', text):
        return False
    # Reject if ends with a word followed by apostrophe (name remnant like "josepho'")
    if re.search(r"\w+'\s*$", text):
        return False
    # Reject if starts with garbled running-header remnant: "of 1 the...", "1 the..."
    if re.match(r'^(?:of|and|or|the|in|on|for|to|a|an)\s+\d+\s+', text, re.I):
        return False
    if re.match(r'^\d+\s+', text):
        return False
    return True


_CLAIM_STARTERS = re.compile(
    r'(?:we|this study|this paper|this work|our|here,?\s+we|results?|findings?|'
    r'data|analysis|evidence|the\s+present)\s+'
    r'(?:show|find|demonstrate|propose|report|present|argue|suggest|reveal|'
    r'found|showed|demonstrated|reported|observed|identified|indicates?|suggests?)\s+'
    r'(?:that\s+)?',
    re.IGNORECASE,
)


def _extract_main_claim(abstract: str, max_len: int = 110, title: str = "") -> str:
    """Extract the main claim, cleaning PDF noise first. Falls back to title."""
    good = _readable_sentences(abstract)

    # Prefer sentences with explicit claim language
    for sent in good:
        m = _CLAIM_STARTERS.search(sent)
        if m:
            claim = sent[m.end():].strip()[:max_len]
            if len(claim) > 20:
                claim = claim.rstrip('.,;:')
                if not claim.endswith('.'):
                    claim += '...'
                candidate = claim.lower()
                if _claim_looks_clean(candidate):
                    return candidate

    # Second readable sentence is often the main finding
    if len(good) >= 2:
        s = good[1].strip()[:max_len]
        if len(s) > 25:
            candidate = s.lower().rstrip('.,;:')
            if _claim_looks_clean(candidate):
                return candidate

    # First readable sentence
    if good:
        s = good[0].strip()[:max_len]
        if len(s) > 20:
            candidate = s.lower().rstrip('.,;:')
            if _claim_looks_clean(candidate):
                return candidate

    # Fall back to the paper title (titles are usually clean)
    if title:
        clean_title = _clean_title_for_claim(title)
        if len(clean_title) > 10:
            return clean_title.lower()

    return "investigates key aspects of the research question"


def _extract_key_finding(abstract: str, max_len: int = 90, title: str = "") -> str:
    """Extract a key result or finding from an abstract. Falls back to title."""
    good = _readable_sentences(abstract)

    _RESULT_RE = re.compile(
        r'(?:we\s+(?:found|observed|showed|demonstrated|report)|'
        r'results?\s+(?:show|indicate|suggest|reveal|demonstrate)|'
        r'our\s+(?:results?|findings?|analysis)\s+(?:show|indicate|suggest)|'
        r'here\s+we)',
        re.IGNORECASE,
    )
    for sent in good[1:]:
        m = _RESULT_RE.search(sent)
        if m:
            fragment = sent[m.end():].strip()[:max_len]
            if len(fragment) > 20:
                candidate = fragment.lower().rstrip('.,;:') + '...'
                if _claim_looks_clean(candidate):
                    return candidate

    # Fall back to last readable sentence (often conclusion)
    if good:
        last = good[-1].strip()[:max_len]
        if len(last) > 20:
            candidate = last.lower().rstrip('.,;:')
            if _claim_looks_clean(candidate):
                return candidate

    # Fall back to title
    if title:
        clean_title = _clean_title_for_claim(title)
        if len(clean_title) > 10:
            return clean_title.lower()

    return "contributes important evidence to this area"


def _short_author(paper: "PaperMeta") -> str:
    """Return a short author string, with robust fallbacks for noisy data."""
    # 1. Use parsed authors if available
    if paper.authors:
        first = paper.authors[0]
        if "," in first:
            last = first.split(",")[0].strip()
        else:
            parts = first.split()
            last  = parts[-1] if parts else first
        return f"{last} et al." if len(paper.authors) > 1 else last

    # 2. Try to extract the first author's surname from the abstract
    surname = _try_extract_first_author(paper.abstract or "", paper.title or "")
    if surname:
        return f"{surname} et al."

    # 3. Fall back to a clean short-title reference
    title = (paper.title or "").strip()
    # Strip running header prefixes: "THERMODYNAMIC GEOMETRY OF CHOICE 1 The..."
    title = re.sub(r'^(?:[A-Z]+\s+){3,}\d+\s*', '', title).strip()
    # Strip journal name prefixes that leak into the title field
    title = re.sub(
        r'^(?:The\s+)?Journal\s+of\s+(?:[A-Z]\w+\s+){1,4}',
        '', title, flags=re.I,
    ).strip()
    title = re.sub(
        r'^(?:Cognitive\s+Psychology|Neuropsychologia|'
        r'Psychological\s+(?:Review|Bulletin|Science)|'
        r'Psychology|Neuroscience|Mathematics)\s*',
        '', title, flags=re.I,
    ).strip()
    # Strip author/affiliation junk after title
    title = re.sub(r'\s+(?:Department|Division|Institute)\s+of\b.*', '', title, flags=re.I)
    title = re.sub(r'\s+\[.*', '', title)
    title = re.sub(r'\s+Autho\b.*', '', title, flags=re.I)
    title = re.sub(r'^(?:The|A|An)\s+', '', title, flags=re.I)
    words = [w for w in title.split()
             if (not w.isupper() or len(w) <= 3) and len(w) < 20]
    if not words:
        words = title.split()[:3]
    short = ' '.join(words[:4])
    if short:
        return f'"{short}"'
    return "the authors"


def _paper_ref(paper: "PaperMeta") -> str:
    """Format a concise in-text citation string."""
    author = _short_author(paper)
    year   = paper.year or _try_extract_year(paper.abstract or "")
    if year:
        return f"{author} ({year})"
    return author


# ── Theme extraction ──────────────────────────────────────────────────────────

_THEME_STOP = frozenset({
    'about', 'between', 'and', 'or', 'the', 'a', 'an', 'of', 'in',
    'on', 'for', 'to', 'how', 'what', 'why', 'when', 'where', 'which',
    'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were',
    'have', 'has', 'do', 'does', 'can', 'could', 'would', 'should',
    'will', 'may', 'might', 'with', 'from', 'by', 'at', 'as', 'not',
    'but', 'if', 'so', 'than', 'very', 'much', 'more', 'also', 'any',
    'all', 'each', 'some', 'tell', 'me', 'my', 'your', 'its', 'be',
    'been', 'being', 'there', 'here',
    # extra filler words that produce bad themes
    'research', 'study', 'paper', 'work', 'analysis', 'model',
    'method', 'approach', 'framework', 'novel', 'new', 'based',
    'relate', 'related', 'using', 'used', 'like', 'think', 'know', 'believe',
    'consider', 'make', 'made', 'get', 'got',
})


def _extract_theme(query: str, papers: list["PaperMeta"]) -> str:
    """
    Build a natural-sounding theme phrase from the query.

    Strategy: split the query into phrases of adjacent content words
    (gaps at stop words), then join the first two phrases with "and".
    Example: "Hick's law can be related to thermodynamic constraints?"
             → ["Hick's law", "thermodynamic constraints"]
             → "Hick's law and thermodynamic constraints"
    """
    words = query.split()
    phrases: list[str] = []
    current: list[str] = []

    for w in words:
        w_clean = w.lower().strip('.,;:?!')
        if w_clean in _THEME_STOP or len(w_clean) <= 2:
            if current:
                phrases.append(' '.join(current))
                current = []
        else:
            current.append(w_clean)
    if current:
        phrases.append(' '.join(current))

    if len(phrases) >= 2:
        return f"{phrases[0]} and {phrases[1]}"
    if phrases:
        return phrases[0]

    # Fall back to common title words across papers
    freq: dict[str, int] = {}
    for p in papers[:6]:
        for w in p.title.lower().split():
            w = w.strip('.,;:')
            if len(w) > 4 and w not in _THEME_STOP:
                freq[w] = freq.get(w, 0) + 1
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return ' '.join(w for w, _ in top[:3])

    return "this research area"


# ── Paragraph generators ───────────────────────────────────────────────────────

def _gen_convergence(
    query : str,
    papers: list["PaperMeta"],
    theme : str,
    graph : "HierarchicalResearchGraph",
) -> str:
    p_a, p_b = papers[0], papers[1]
    claim_a  = _extract_main_claim(p_a.abstract, title=p_a.title)
    claim_b  = _extract_main_claim(p_b.abstract, title=p_b.title)
    ref_a    = _paper_ref(p_a)
    ref_b    = _paper_ref(p_b)

    openings = [
        f"The literature on {theme} points toward a coherent picture.",
        f"A convergent body of evidence has emerged around {theme}.",
        f"Research on {theme} has reached substantial consensus.",
        f"Several independent lines of work converge on {theme}.",
    ]
    support_verbs = [
        "demonstrated that", "showed that", "provided evidence that",
        "found that", "established that", "reported that",
    ]
    bridges = [
        f"This view is reinforced by",
        f"Independently,",
        f"Consistent with this,",
        f"This finding is echoed by",
    ]
    closings = [
        f"Together, these works solidify the importance of {theme} in the field.",
        f"The convergence of these independent lines strengthens confidence in this view.",
        f"Taken together, they point to a robust phenomenon warranting further inquiry.",
    ]

    if len(papers) >= 3:
        ref_c   = _paper_ref(papers[2])
        mid     = (f"{random.choice(bridges)} {ref_b}, "
                   f"which {random.choice(support_verbs)} {claim_b}. "
                   f"A third line of evidence comes from {ref_c}, "
                   f"whose results align with this emerging consensus. ")
    else:
        mid = (f"{random.choice(bridges)} {ref_b}, "
               f"which {random.choice(support_verbs)} {claim_b}. ")

    return (
        f"{random.choice(openings)} "
        f"{ref_a} {random.choice(support_verbs)} {claim_a}. "
        f"{mid}"
        f"{random.choice(closings)}"
    )


def _gen_tension(
    query           : str,
    papers          : list["PaperMeta"],
    theme           : str,
    graph           : "HierarchicalResearchGraph",
    contradict_pairs: list[tuple[str, str]],
) -> str:
    if contradict_pairs:
        pid_a, pid_b = contradict_pairs[0]
        p_a = graph.get_paper(pid_a) or papers[0]
        p_b = graph.get_paper(pid_b) or (papers[1] if len(papers) > 1 else papers[0])
    else:
        p_a = papers[0]
        p_b = papers[1] if len(papers) > 1 else papers[0]

    claim_a = _extract_main_claim(p_a.abstract, title=p_a.title)
    claim_b = _extract_main_claim(p_b.abstract, title=p_b.title)
    ref_a   = _paper_ref(p_a)
    ref_b   = _paper_ref(p_b)

    openings = [
        f"The interpretation of {theme} remains actively debated.",
        f"A key tension has emerged in research on {theme}.",
        f"The study of {theme} has generated competing interpretations.",
        f"Research on {theme} reveals a productive scientific disagreement.",
    ]
    contrasts = [
        f"This view has been challenged by {ref_b}, which argued that",
        f"However, {ref_b} presented contrasting evidence, finding that",
        f"{ref_b} offered a different perspective, contending that",
        f"In contrast, {ref_b} reported that",
    ]
    closings = [
        "This disagreement likely reflects methodological differences and merits direct comparison.",
        "Resolving this tension may require studies that directly pit these accounts against each other.",
        "The divergence may stem from differences in measurement, sample, or analytic approach.",
        "This debate has been productive, pushing the field toward greater theoretical precision.",
    ]

    return (
        f"{random.choice(openings)} "
        f"{ref_a} argued that {claim_a}. "
        f"{random.choice(contrasts)} {claim_b}. "
        f"{random.choice(closings)}"
    )


def _gen_evolution(
    query : str,
    papers: list["PaperMeta"],
    theme : str,
    graph : "HierarchicalResearchGraph",
) -> str:
    # Prefer causal DAG topological order (acyclic influence flow)
    ordered: list[tuple["PaperMeta", int]] = []
    G_caus = graph.G_causal
    if G_caus.number_of_edges() > 0:
        paper_ids = {p.paper_id for p in papers}
        sub = G_caus.subgraph([n for n in G_caus.nodes if n in paper_ids]).copy()
        if sub.number_of_edges() > 0:
            import networkx as _nx
            for pid in _nx.topological_sort(sub):
                meta = graph.get_paper(pid)
                if meta:
                    ordered.append((meta, meta.year or 0))

    # Fallback: sort by publication year
    if len(ordered) < 2:
        ordered = sorted([(p, p.year) for p in papers if p.year], key=lambda x: x[1])
    if len(ordered) < 2:
        ordered = [(p, 0) for p in papers[:2]]

    dated = ordered
    p_old, y_old = dated[0]
    p_new, y_new = dated[-1]
    claim_old = _extract_main_claim(p_old.abstract, title=p_old.title)
    claim_new = _extract_main_claim(p_new.abstract, title=p_new.title)
    ref_old   = _paper_ref(p_old)
    ref_new   = _paper_ref(p_new)

    openings = [
        f"Understanding of {theme} has evolved considerably.",
        f"The conceptual landscape of {theme} has undergone substantial revision.",
        f"Research on {theme} has progressed through distinct phases.",
        f"The field's approach to {theme} has matured substantially over time.",
    ]
    transitions = [
        f"This framework was later revised by {ref_new}, who showed that",
        f"More recently, {ref_new} advanced the field by demonstrating that",
        f"A significant shift came with {ref_new}, which established that",
        f"Building on this foundation, {ref_new} showed that",
    ]
    closings = [
        "This trajectory reflects the cumulative nature of progress in this area.",
        "Each generation of studies has refined the conceptual tools available to researchers.",
        "This evolution suggests the field is converging toward a more nuanced understanding.",
    ]

    mid_text = ""
    if len(dated) >= 3:
        p_mid   = dated[len(dated) // 2][0]
        ref_mid = _paper_ref(p_mid)
        mid_text = f"An intermediate step came from {ref_mid}, which refined these ideas further. "

    return (
        f"{random.choice(openings)} "
        f"Early work by {ref_old} established that {claim_old}. "
        f"{mid_text}"
        f"{random.choice(transitions)} {claim_new}. "
        f"{random.choice(closings)}"
    )


def _gen_synthesis(
    query : str,
    papers: list["PaperMeta"],
    theme : str,
    graph : "HierarchicalResearchGraph",
) -> str:
    p_a     = papers[0]
    p_b     = papers[1] if len(papers) > 1 else papers[0]
    claim_a = _extract_main_claim(p_a.abstract, title=p_a.title)
    claim_b = _extract_main_claim(p_b.abstract, title=p_b.title)
    finding = _extract_key_finding(p_a.abstract, title=p_a.title)
    ref_a   = _paper_ref(p_a)
    ref_b   = _paper_ref(p_b)

    openings = [
        f"A fuller picture of {theme} emerges when multiple perspectives are combined.",
        f"The literature on {theme} benefits from synthesis across different approaches.",
        f"Integrating findings from different traditions illuminates {theme}.",
        f"Different research traditions have approached {theme} with complementary insights.",
    ]
    bridges = [
        "When viewed together, these perspectives suggest",
        "A synthesis of these findings implies",
        "Combining these insights, one can argue that",
        "Taken together, the accumulated evidence points to",
    ]

    # Use the last clean readable sentence as a synthesis seed
    good_sents = _readable_sentences(p_a.abstract or "")
    synth = f"the broader importance of {theme}"
    for sent in reversed(good_sents):
        candidate = sent[:90].lower()
        if _claim_looks_clean(candidate):
            synth = candidate
            break

    return (
        f"{random.choice(openings)} "
        f"{ref_a} approached the problem from the angle of {claim_a[:60]}... "
        f"{ref_b}, working within a different tradition, emphasised that {claim_b}. "
        f"{random.choice(bridges)} {synth}. "
        f"Such a synthesis positions {theme} as a genuinely multi-faceted phenomenon."
    )


def _gen_gap(
    query : str,
    papers: list["PaperMeta"],
    theme : str,
    graph : "HierarchicalResearchGraph",
) -> str:
    p_a   = papers[0]
    ref_a = _paper_ref(p_a)
    claim_a = _extract_main_claim(p_a.abstract, title=p_a.title)

    openings = [
        f"Despite progress in {theme}, important questions remain open.",
        f"The literature on {theme} has advanced considerably, yet significant gaps persist.",
        f"While much has been learned about {theme}, several key issues await resolution.",
        f"Research on {theme} has made strides, but critical lacunae remain.",
    ]

    body = f"{ref_a} established that {claim_a}"
    if len(papers) > 1:
        p_b   = papers[1]
        ref_b = _paper_ref(p_b)
        body += f". {ref_b} further demonstrated that {_extract_main_claim(p_b.abstract, title=p_b.title)}"

    gap_ideas = [
        f"the precise mechanistic link between these findings",
        f"the boundary conditions under which these effects reliably hold",
        f"the developmental trajectory and temporal stability of {theme}",
        f"the cross-cultural and cross-population generalizability of these results",
        f"the computational or neural basis underlying {theme}",
        f"how individual differences moderate the observed effects",
    ]
    gap = random.choice(gap_ideas)

    closings = [
        f"Future work targeting {gap} would substantially advance the field.",
        f"Addressing {gap} would bridge a critical gap in current understanding.",
        f"The question of {gap} represents a productive direction for future research.",
        f"Systematic investigation of {gap} could resolve current ambiguities.",
    ]

    return (
        f"{random.choice(openings)} "
        f"{body}. "
        f"However, {gap} remains underexplored in the existing literature. "
        f"{random.choice(closings)}"
    )


# ── Main Arguer class ──────────────────────────────────────────────────────────

_ARG_LABELS = {
    ARG_CONVERGENCE : "Convergence  — multiple papers agree",
    ARG_TENSION     : "Tension      — competing interpretations",
    ARG_EVOLUTION   : "Evolution    — how the field has changed",
    ARG_SYNTHESIS   : "Synthesis    — combining different angles",
    ARG_GAP         : "Gap          — what remains unexplored",
}

_ARG_CONN_TYPE = {
    ARG_CONVERGENCE : "supports",
    ARG_TENSION     : "contradicts",
    ARG_EVOLUTION   : "mixed",
    ARG_SYNTHESIS   : "mixed",
    ARG_GAP         : "mentions",
}

_ARG_EXPLANATION = {
    ARG_CONVERGENCE: (
        "Independent papers converge on shared conclusions, mutually reinforcing each other."
    ),
    ARG_TENSION: (
        "Papers present conflicting evidence or competing theoretical interpretations."
    ),
    ARG_EVOLUTION: (
        "The field's understanding of this topic has developed and shifted over time."
    ),
    ARG_SYNTHESIS: (
        "Insights from different research angles are woven together into a unified view."
    ),
    ARG_GAP: (
        "Established findings point to an underexplored question ripe for future work."
    ),
}


class Arguer:
    """
    Creative argumentation engine — the 'Creative Cortex' of ResearchBuddy.

    Works on top of a QueryResult from the Reasoner, using citation type
    information (supports / contradicts / mentions) stored on G_citation
    edges to generate structured argument paragraphs.

    Learns via StyleProfile (stored in the graph pickle) which argument
    types the user finds correct and useful, progressively improving output.
    """

    # ── Main entry point ──────────────────────────────────────────────────

    def generate(
        self,
        query        : str,
        query_result : "QueryResult",
        graph        : "HierarchicalResearchGraph",
        style_profile: StyleProfile,
        n            : int = 3,
    ) -> list[ArgumentParagraph]:
        """
        Generate n argument paragraphs from the given QueryResult.

        Selection strategy:
          1. Detect which argument types are structurally feasible given the
             available papers and their citation relationships.
          2. Sample from feasible types weighted by the StyleProfile.
          3. Generate one paragraph per selected type.
        """
        papers = [m for m, _, _ in query_result.relevant_papers[:8]]
        if len(papers) < 2:
            return []

        theme = _extract_theme(query, papers)

        available     = self._available_types(papers, graph)
        selected      = style_profile.weighted_sample(available, n=n)

        # Pad if fewer than n types are available
        if len(selected) < n:
            extra = [t for t in _ALL_ARG_TYPES if t not in selected]
            selected += extra[:n - len(selected)]

        paragraphs: list[ArgumentParagraph] = []
        for arg_type in selected[:n]:
            para = self._build(arg_type, query, papers, theme, graph)
            if para is not None:
                paragraphs.append(para)

        return paragraphs

    # ── Feasibility check ─────────────────────────────────────────────────

    def _available_types(
        self,
        papers: list["PaperMeta"],
        graph : "HierarchicalResearchGraph",
    ) -> list[str]:
        """Return argument types that are structurally supported by the graph."""
        available = [ARG_SYNTHESIS, ARG_GAP]     # always possible

        # Convergence: at least 2 papers share supporting edges (or always include)
        available.append(ARG_CONVERGENCE)

        # Tension: at least one contradicts edge between paper pairs
        ids = [p.paper_id for p in papers]
        has_contradict = any(
            (graph.G_citation.has_edge(u, v)
             and graph.G_citation[u][v].get("cit_type") == "contradicts")
            for i, u in enumerate(ids)
            for v in ids[i + 1:]
        ) or any(
            (graph.G_citation.has_edge(v, u)
             and graph.G_citation[v][u].get("cit_type") == "contradicts")
            for i, u in enumerate(ids)
            for v in ids[i + 1:]
        )
        if has_contradict:
            available.append(ARG_TENSION)

        # Evolution: papers span at least 3 years OR causal DAG has ≥2 edges
        years = [p.year for p in papers if p.year]
        year_span_ok = years and (max(years) - min(years) >= 3)
        causal_ok = False
        if graph.G_causal.number_of_edges() >= 2:
            paper_ids_set = {p.paper_id for p in papers}
            causal_sub = graph.G_causal.subgraph(
                [n for n in graph.G_causal.nodes if n in paper_ids_set]
            )
            causal_ok = causal_sub.number_of_edges() >= 2
        if year_span_ok or causal_ok:
            available.append(ARG_EVOLUTION)

        return available

    def _get_contradict_pairs(
        self,
        papers: list["PaperMeta"],
        graph : "HierarchicalResearchGraph",
    ) -> list[tuple[str, str]]:
        """Find paper pairs with contradicts edges in the citation graph."""
        pairs = []
        ids   = [p.paper_id for p in papers]
        for i, pid_a in enumerate(ids):
            for pid_b in ids[i + 1:]:
                for u, v in [(pid_a, pid_b), (pid_b, pid_a)]:
                    if (graph.G_citation.has_edge(u, v)
                            and graph.G_citation[u][v].get("cit_type") == "contradicts"):
                        pairs.append((u, v))
        return pairs

    # ── Paragraph builder ─────────────────────────────────────────────────

    def _build(
        self,
        arg_type: str,
        query   : str,
        papers  : list["PaperMeta"],
        theme   : str,
        graph   : "HierarchicalResearchGraph",
    ) -> Optional[ArgumentParagraph]:
        try:
            if arg_type == ARG_CONVERGENCE:
                text = _gen_convergence(query, papers, theme, graph)
            elif arg_type == ARG_TENSION:
                pairs = self._get_contradict_pairs(papers, graph)
                text  = _gen_tension(query, papers, theme, graph, pairs)
            elif arg_type == ARG_EVOLUTION:
                text = _gen_evolution(query, papers, theme, graph)
            elif arg_type == ARG_SYNTHESIS:
                text = _gen_synthesis(query, papers, theme, graph)
            elif arg_type == ARG_GAP:
                text = _gen_gap(query, papers, theme, graph)
            else:
                return None

            return ArgumentParagraph(
                arg_type        = arg_type,
                arg_type_label  = _ARG_LABELS[arg_type],
                text            = text,
                paper_ids       = [p.paper_id for p in papers[:4]],
                paper_refs      = [_paper_ref(p) for p in papers[:4]],
                connection_type = _ARG_CONN_TYPE[arg_type],
                explanation     = _ARG_EXPLANATION[arg_type],
            )
        except Exception:
            return None
