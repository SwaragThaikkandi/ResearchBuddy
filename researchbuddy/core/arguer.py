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


# ── Text extraction helpers ────────────────────────────────────────────────────

_CLAIM_STARTERS = re.compile(
    r'(?:we|this study|this paper|this work|our|here,?\s+we|results?|findings?|'
    r'data|analysis|evidence|the\s+present)\s+'
    r'(?:show|find|demonstrate|propose|report|present|argue|suggest|reveal|'
    r'found|showed|demonstrated|reported|observed|identified|indicates?|suggests?)\s+'
    r'(?:that\s+)?',
    re.IGNORECASE,
)


def _extract_main_claim(abstract: str, max_len: int = 110) -> str:
    """Extract the main claim sentence from an abstract."""
    if not abstract or len(abstract) < 25:
        return "investigates key aspects of the research question"

    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())

    # Prefer sentences with explicit claim language
    for sent in sentences:
        m = _CLAIM_STARTERS.search(sent)
        if m:
            claim = sent[m.end():].strip()[:max_len]
            if len(claim) > 20:
                claim = claim.rstrip('.,;:')
                if len(claim) < max_len and not claim.endswith('.'):
                    claim += '...'
                return claim.lower()

    # Fall back to second sentence (often the main finding)
    if len(sentences) >= 2:
        s = sentences[1].strip()[:max_len]
        if len(s) > 25:
            return s.lower().rstrip('.,;:')

    return abstract[:max_len].lower().rstrip('.,;:')


def _extract_key_finding(abstract: str, max_len: int = 90) -> str:
    """Extract a key result or finding from an abstract (used in synthesis)."""
    if not abstract:
        return "provides relevant evidence"

    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())

    _RESULT_RE = re.compile(
        r'(?:we\s+(?:found|observed|showed|demonstrated|report)|'
        r'results?\s+(?:show|indicate|suggest|reveal|demonstrate)|'
        r'our\s+(?:results?|findings?|analysis)\s+(?:show|indicate|suggest)|'
        r'here\s+we)',
        re.IGNORECASE,
    )
    for sent in sentences[1:]:
        m = _RESULT_RE.search(sent)
        if m:
            fragment = sent[m.end():].strip()[:max_len]
            if len(fragment) > 20:
                return fragment.lower().rstrip('.,;:') + '...'

    # Fall back to last sentence (often conclusion)
    if sentences:
        last = sentences[-1].strip()[:max_len]
        if len(last) > 20:
            return last.lower().rstrip('.,;:')

    return "contributes important evidence to this area"


def _short_author(paper: "PaperMeta") -> str:
    """Return a short author string, e.g. 'Smith' or 'Smith et al.'."""
    if not paper.authors:
        # Fall back to first word of title
        words = paper.title.split()
        return words[0] if words else "Authors"
    first = paper.authors[0]
    if "," in first:
        last = first.split(",")[0].strip()
    else:
        parts = first.split()
        last  = parts[-1] if parts else first
    return f"{last} et al." if len(paper.authors) > 1 else last


def _paper_ref(paper: "PaperMeta") -> str:
    """Format 'Author et al. (YEAR)' reference string."""
    author = _short_author(paper)
    year   = str(paper.year) if paper.year else "n.d."
    return f"{author} ({year})"


def _extract_theme(query: str, papers: list["PaperMeta"]) -> str:
    """Extract a 2–3 word theme phrase from the query and paper titles."""
    _STOP = {
        'about', 'between', 'and', 'or', 'the', 'a', 'an', 'of', 'in',
        'on', 'for', 'to', 'how', 'what', 'why', 'when', 'where', 'which',
        'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were',
        'have', 'has', 'do', 'does', 'can', 'could', 'would', 'should',
        'will', 'may', 'might', 'with', 'from', 'by', 'at', 'as', 'not',
        'but', 'if', 'so', 'than', 'very', 'much', 'more', 'also', 'any',
        'all', 'each', 'some', 'tell', 'me', 'my', 'your', 'its',
        'research', 'study', 'paper', 'work', 'analysis', 'model',
        'method', 'approach', 'framework', 'novel',
    }
    query_words = [
        w.lower().strip('.,;:?!')
        for w in query.split()
        if len(w) > 3 and w.lower() not in _STOP
    ]
    if len(query_words) >= 2:
        return ' '.join(query_words[:3])

    freq: dict[str, int] = {}
    for p in papers[:6]:
        for w in p.title.lower().split():
            w = w.strip('.,;:')
            if len(w) > 4 and w not in _STOP:
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
    claim_a  = _extract_main_claim(p_a.abstract)
    claim_b  = _extract_main_claim(p_b.abstract)
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

    claim_a = _extract_main_claim(p_a.abstract)
    claim_b = _extract_main_claim(p_b.abstract)
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
    dated = sorted([(p, p.year) for p in papers if p.year], key=lambda x: x[1])
    if len(dated) < 2:
        dated = [(p, 0) for p in papers[:2]]

    p_old, y_old = dated[0]
    p_new, y_new = dated[-1]
    claim_old = _extract_main_claim(p_old.abstract)
    claim_new = _extract_main_claim(p_new.abstract)
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
    claim_a = _extract_main_claim(p_a.abstract)
    claim_b = _extract_main_claim(p_b.abstract)
    finding = _extract_key_finding(p_a.abstract)
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

    # Use the last sentence of the best paper's abstract as a synthesis seed
    best_abs = p_a.abstract or ""
    sents    = re.split(r'(?<=[.!?])\s+', best_abs.strip())
    synth    = sents[-1][:90].lower() if len(sents) > 1 else f"the broader importance of {theme}"

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
    claim_a = _extract_main_claim(p_a.abstract)

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
        body += f". {ref_b} further demonstrated that {_extract_main_claim(p_b.abstract)}"

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

        # Evolution: papers span at least 3 years
        years = [p.year for p in papers if p.year]
        if years and (max(years) - min(years) >= 3):
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
