"""
llm.py  --  Local LLM interface ("The Voice").

Central module for all LLM interactions.  Every other module (arguer, searcher)
calls into this; none talk to Ollama directly.

Design philosophy:
  The causal graph is the BRAIN  --  it provides structure (influence flow,
  citation types, cluster assignments).  The LLM is the VOICE  --  it
  articulates graph facts in natural language.  The LLM operates INSIDE the
  cage of the graph:  it must never contradict what the graph says.

  Every prompt constructed by ``build_paper_context()`` includes explicit
  GRAPH RELATIONSHIPS that the LLM is instructed to follow verbatim.

Fallback:
  When Ollama is not running (or the configured model has not been pulled),
  ``is_available()`` returns False and all callers silently fall back to
  template-based generation.  ResearchBuddy is 100% functional without Ollama.
"""

from __future__ import annotations

import json
import re
import time
import requests
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta


# ── Status dataclass ──────────────────────────────────────────────────────────

@dataclass
class LLMStatus:
    available: bool
    model_name: str
    gpu_name: Optional[str]   = None
    gpu_vram_mb: Optional[int] = None
    error: Optional[str]      = None


# ── Mojibake repair ──────────────────────────────────────────────────────────

_MOJIBAKE_MAP = {
    "\u00e2\u20ac\u2122": "\u2019",     # right single quote  (')
    "\u00e2\u20ac\u0153": "\u201c",     # left double quote   (\u201c)
    "\u00e2\u20ac\u009d": "\u201d",     # right double quote  (\u201d)
    "\u00e2\u20ac\u201c": "\u2014",     # em dash             (\u2014)
    "\u00e2\u20ac\u201d": "\u2013",     # en dash             (\u2013)
    "\u00e2\u20ac\u00a6": "\u2026",     # ellipsis            (\u2026)
    "\u00c2\u00a0": " ",               # non-breaking space artifact
}


def fix_mojibake(text: str) -> str:
    """Repair UTF-8-as-CP1252 mojibake (e.g. \u00e2\u20ac\u2122 -> ')."""
    if not text:
        return ""
    t = text
    # Try encoding round-trip: if text was UTF-8 decoded as CP1252
    try:
        fixed = t.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        if fixed and len(fixed) >= len(t) * 0.7:
            t = fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Apply known substitutions for partial fixes
    for bad, good in _MOJIBAKE_MAP.items():
        t = t.replace(bad, good)
    # Normalise smart quotes to ASCII (arguer templates use ASCII quotes)
    for old, new in [("\u2019", "'"), ("\u2018", "'"),
                     ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2014", "--"), ("\u2013", "-"),
                     ("\u2026", "...")]:
        t = t.replace(old, new)
    # Strip remaining high-byte control chars
    t = re.sub(r"[\x80-\x9f]", "", t)
    return t


# ── GPU detection ────────────────────────────────────────────────────────────

def detect_gpu() -> dict:
    """Report GPU name and VRAM (or ``available=False`` if no CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem // (1024 ** 2)
            return {"available": True, "name": name, "vram_mb": vram}
    except ImportError:
        pass
    return {"available": False, "name": None, "vram_mb": None}


# ── Paper context builder (THE CAGE) ────────────────────────────────────────

def build_paper_context(
    papers: list["PaperMeta"],
    graph: "HierarchicalResearchGraph",
    max_papers: int = 5,
) -> str:
    """
    Build a structured text block encoding graph relationships as explicit facts.

    The LLM system prompt instructs it to follow these facts verbatim.
    This is the "cage" that constrains LLM output to match graph structure.
    """
    import networkx as nx

    papers = papers[:max_papers]
    if not papers:
        return "(no papers available)"

    lines: list[str] = []
    pid_to_idx: dict[str, int] = {}

    # ── Paper descriptions ────────────────────────────────────────────
    for i, p in enumerate(papers, 1):
        pid_to_idx[p.paper_id] = i
        title = fix_mojibake(p.title or "Untitled")
        # Author string
        if p.authors:
            auth = p.authors[0].split(",")[0].split()[-1] if p.authors else "Unknown"
            if len(p.authors) > 1:
                auth += " et al."
        else:
            auth = "Unknown"
        year_str = f" ({p.year})" if p.year else ""

        lines.append(f"Paper {i}: \"{title}\" by {auth}{year_str}")

        # Cleaned abstract snippet
        abstract = fix_mojibake(p.abstract or "")
        if abstract:
            # Strip known metadata noise (light version)
            abstract = re.sub(r"https?://\S+", "", abstract)
            abstract = re.sub(r"doi\s*:\s*\S+", "", abstract, flags=re.I)
            abstract = re.sub(r"\s{2,}", " ", abstract).strip()
            snippet = abstract[:300]
            if len(abstract) > 300:
                snippet = snippet.rsplit(" ", 1)[0] + "..."
            lines.append(f"  Abstract: {snippet}")
        lines.append("")

    # ── Graph relationships ───────────────────────────────────────────
    relations: list[str] = []

    for i, pa in enumerate(papers):
        for j, pb in enumerate(papers):
            if i >= j:
                continue
            idx_a = pid_to_idx[pa.paper_id]
            idx_b = pid_to_idx[pb.paper_id]

            # Causal influence (from G_causal)
            if graph.G_causal.has_edge(pa.paper_id, pb.paper_id):
                conf = graph.G_causal[pa.paper_id][pb.paper_id].get(
                    "causal_confidence", 0.0
                )
                relations.append(
                    f"- Paper {idx_a} -> Paper {idx_b}: INFLUENCED "
                    f"(causal confidence: {conf:.2f})"
                )
            elif graph.G_causal.has_edge(pb.paper_id, pa.paper_id):
                conf = graph.G_causal[pb.paper_id][pa.paper_id].get(
                    "causal_confidence", 0.0
                )
                relations.append(
                    f"- Paper {idx_b} -> Paper {idx_a}: INFLUENCED "
                    f"(causal confidence: {conf:.2f})"
                )

            # Citation type (from G_citation)
            for u, v in [(pa.paper_id, pb.paper_id), (pb.paper_id, pa.paper_id)]:
                if graph.G_citation.has_edge(u, v):
                    cit_type = graph.G_citation[u][v].get("cit_type", "mentions")
                    u_idx = pid_to_idx[u]
                    v_idx = pid_to_idx[v]
                    relations.append(
                        f"- Paper {u_idx} {cit_type.upper()} Paper {v_idx} "
                        f"(citation relationship)"
                    )
                    break  # avoid duplicate

    # Cluster / niche assignments
    p2n = graph.paper_to_niche()
    niche_groups: dict[str, list[int]] = {}
    for p in papers:
        niche = p2n.get(p.paper_id)
        if niche:
            niche_groups.setdefault(niche, []).append(pid_to_idx[p.paper_id])
    for niche, indices in niche_groups.items():
        if len(indices) >= 2:
            idx_str = ", ".join(f"Paper {i}" for i in indices)
            relations.append(f"- {idx_str} are in the same niche: \"{niche}\"")

    if relations:
        lines.append("GRAPH RELATIONSHIPS (these are established facts "
                      "-- do not contradict):")
        lines.extend(relations)
        lines.append("")

    # ── Causal chain (topological sort) ───────────────────────────────
    paper_ids = {p.paper_id for p in papers}
    sub = graph.G_causal.subgraph(
        [n for n in graph.G_causal.nodes if n in paper_ids]
    ).copy()
    if sub.number_of_edges() > 0 and nx.is_directed_acyclic_graph(sub):
        chain = []
        for pid in nx.topological_sort(sub):
            if pid in pid_to_idx:
                p = next((pp for pp in papers if pp.paper_id == pid), None)
                if p:
                    year = f" ({p.year})" if p.year else ""
                    chain.append(f"Paper {pid_to_idx[pid]}{year}")
        if len(chain) >= 2:
            lines.append("CAUSAL CHAIN (intellectual influence flow):")
            lines.append("  " + " -> ".join(chain))
            lines.append("")

    return "\n".join(lines)


# ── Ollama client ────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Lightweight HTTP client for Ollama's REST API.

    Uses ``requests`` (already a project dependency) -- no extra packages.
    """

    def __init__(self, base_url: str, model: str):
        self._base = base_url.rstrip("/")
        self._model = model
        self._status: Optional[LLMStatus] = None

    # ── Health check ─────────────────────────────────────────────────

    def _check_health(self) -> LLMStatus:
        gpu = detect_gpu()
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            # Fuzzy match: "phi3.5" matches "phi3.5:latest"
            found = any(
                self._model in m or m.startswith(self._model)
                for m in models
            )
            if found:
                return LLMStatus(
                    available=True,
                    model_name=self._model,
                    gpu_name=gpu.get("name"),
                    gpu_vram_mb=gpu.get("vram_mb"),
                )
            else:
                avail = ", ".join(models[:5]) if models else "(none)"
                return LLMStatus(
                    available=False,
                    model_name=self._model,
                    error=(
                        f"Model '{self._model}' not found. "
                        f"Run: ollama pull {self._model}\n"
                        f"  Available models: {avail}"
                    ),
                )
        except requests.ConnectionError:
            return LLMStatus(
                available=False,
                model_name=self._model,
                error="Ollama not running. Start with: ollama serve",
            )
        except Exception as e:
            return LLMStatus(
                available=False,
                model_name=self._model,
                error=f"Ollama check failed: {e}",
            )

    def status(self) -> LLMStatus:
        if self._status is None:
            self._status = self._check_health()
        return self._status

    def is_available(self) -> bool:
        return self.status().available

    # ── Text generation ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Generate text via Ollama.  Returns None on any failure.

        Timeout is generous (120s) because small GPUs can be slow.
        Retries up to 2 times on transient errors.
        """
        if not self.is_available():
            return None

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self._base}/api/generate",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                text = r.json().get("response", "").strip()
                if text:
                    return text
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        return None

    # ── JSON generation ──────────────────────────────────────────────

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> Optional[dict]:
        """Generate and parse JSON.  Returns None on parse failure."""
        if not self.is_available():
            return None

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self._base}/api/generate",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                text = r.json().get("response", "").strip()
                if text:
                    return json.loads(text)
            except (json.JSONDecodeError, Exception):
                if attempt < 2:
                    time.sleep(1)
        return None


# ── Singleton accessor ───────────────────────────────────────────────────────

_client: Optional[OllamaClient] = None


def get_llm(model: Optional[str] = None) -> OllamaClient:
    """Get (or create) the singleton OllamaClient."""
    global _client
    if _client is None:
        from researchbuddy.config import LLM_MODEL, LLM_OLLAMA_URL
        _client = OllamaClient(
            base_url=LLM_OLLAMA_URL,
            model=model or LLM_MODEL,
        )
    return _client
