"""
Autotune — ResearchBuddy does research on itself (the Karpathy loop).

Inspired by karpathy/autoresearch: give an agent a real system, ONE ground-
truth metric, and a mutate → measure → keep/discard loop, then let it run
overnight and log every experiment. Here the "model" is ResearchBuddy's own
scoring configuration, and the ground-truth metric is quality_report() —
how well the scorer ranks the papers YOU already rated (AUC / NDCG /
precision on your own judgments; deterministic, fully offline, no API calls).

The loop, exactly as in autoresearch's program.md:
    1. propose a change (perturb one hyperparameter)
    2. run the experiment (re-score the rated set)
    3. keep if the metric improved by more than epsilon, else revert
    4. append one row to experiments.tsv
    5. repeat

Kept-changes persist to ~/.researchbuddy/autotune.json and are re-applied at
startup, so the tool you open tomorrow is measurably better-tuned to you
than the one you closed tonight.

Honesty note: this optimises preference alignment ON your rated set — a
diagnostic, not a held-out benchmark. epsilon keeps trivial overfits out,
and every experiment is in the TSV, so nothing is hidden.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

from researchbuddy.config import DATA_DIR
import researchbuddy.core.graph_model as gm
from researchbuddy.core.graph_model import HierarchicalResearchGraph
from researchbuddy.core import audit

logger = logging.getLogger(__name__)

TUNING_FILE = DATA_DIR / "autotune.json"
LOG_FILE    = DATA_DIR / "autotune_experiments.tsv"
EPSILON     = 0.003          # minimum objective gain worth keeping

# ── Tunable parameters ─────────────────────────────────────────────────────────
# Each spec: bounds, step, how to read/apply, and what must be recomputed.
# Applying is cheap except `rebuild` (edges depend on the threshold) and
# `learn` (weights must be refit under the new regulariser).

PARAM_SPECS: dict[str, dict] = {
    "alpha": {
        "lo": 0.1, "hi": 0.9, "step": 0.1,
        "get": lambda g: float(g.alpha),
        "set": lambda g, v: setattr(g, "alpha", float(v)),
        "rebuild": False, "learn": False,
    },
    "ppr_damping": {
        "lo": 0.70, "hi": 0.95, "step": 0.05,
        "get": lambda g: float(gm.PPR_DAMPING),
        "set": lambda g, v: setattr(gm, "PPR_DAMPING", float(v)),
        "rebuild": False, "learn": False,
    },
    "similarity_threshold": {
        "lo": 0.30, "hi": 0.70, "step": 0.05,
        "get": lambda g: float(gm.SIMILARITY_THRESHOLD),
        "set": lambda g, v: setattr(gm, "SIMILARITY_THRESHOLD", float(v)),
        "rebuild": True, "learn": False,
    },
    "weight_regularization": {
        "lo": 0.01, "hi": 1.0, "step": 2.0,   # multiplicative step
        "get": lambda g: float(gm.WEIGHT_LEARNING_REGULARIZATION),
        "set": lambda g, v: setattr(gm, "WEIGHT_LEARNING_REGULARIZATION",
                                    float(v)),
        "rebuild": False, "learn": True, "multiplicative": True,
    },
    "rating_half_life_days": {
        "lo": 30, "hi": 365, "step": 30,
        "get": lambda g: float(gm.RATING_HALF_LIFE_DAYS),
        "set": lambda g, v: setattr(gm, "RATING_HALF_LIFE_DAYS", float(v)),
        "rebuild": False, "learn": False,
    },
}


def propose(spec: dict, current: float, rng: random.Random) -> float:
    """One neighbouring value: current ± step (or ×/÷ step), clamped."""
    if spec.get("multiplicative"):
        new = current * spec["step"] if rng.random() < 0.5 \
            else current / spec["step"]
    else:
        new = current + spec["step"] * (1 if rng.random() < 0.5 else -1)
    return round(max(spec["lo"], min(spec["hi"], new)), 4)


def _objective(graph: HierarchicalResearchGraph) -> Optional[float]:
    """The single ground-truth number (higher is better): mean of the graph
    scorer's AUC / NDCG / precision on the user's rated set. None = not
    enough ratings yet (the loop refuses to run blind)."""
    q = graph.quality_report()
    if not q.get("ready"):
        return None
    g = q["graph"]
    vals = [v for v in (g.get("auc"), g.get("ndcg_at_k"),
                        g.get("precision_at_k")) if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _apply(graph: HierarchicalResearchGraph, name: str, value: float) -> None:
    spec = PARAM_SPECS[name]
    spec["set"](graph, value)
    graph._invalidate()
    graph._ppr_mass_cache = None
    if spec["rebuild"]:
        graph.rebuild_hierarchy()
    if spec["learn"]:
        graph.learn_signal_weights()


def _log_row(row: list, path: Optional[Path] = None) -> None:
    p = Path(path or LOG_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("timestamp\tparam\told\tnew\tobjective\tbest\tstatus\n",
                     encoding="utf-8")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\t".join(str(x) for x in row) + "\n")


def read_log(path: Optional[Path] = None, last: int = 50) -> list[dict]:
    p = Path(path or LOG_FILE)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:][-last:]]


# ── Persisted tuning ──────────────────────────────────────────────────────────

def load_tuning(path: Optional[Path] = None) -> dict:
    p = Path(path or TUNING_FILE)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_tuning(tuning: dict, path: Optional[Path] = None) -> None:
    p = Path(path or TUNING_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tuning, indent=2), encoding="utf-8")


def apply_saved_tuning(graph: HierarchicalResearchGraph,
                       path: Optional[Path] = None) -> list[str]:
    """Re-apply kept experiment results at startup. Returns applied names."""
    tuning = load_tuning(path)
    applied = []
    needs_rebuild = needs_learn = False
    for name, value in tuning.items():
        spec = PARAM_SPECS.get(name)
        if spec is None:
            continue
        try:
            spec["set"](graph, float(value))
            needs_rebuild |= spec["rebuild"]
            needs_learn |= spec["learn"]
            applied.append(name)
        except (TypeError, ValueError):
            continue
    if applied:
        graph._invalidate()
        graph._ppr_mass_cache = None
        if needs_rebuild:
            graph.rebuild_hierarchy()
        if needs_learn:
            graph.learn_signal_weights()
        logger.info("[autotune] applied saved tuning: %s", ", ".join(applied))
    return applied


# ── The experiment loop ───────────────────────────────────────────────────────

def run_session(
    graph: HierarchicalResearchGraph,
    rounds: int = 20,
    progress: Optional[Callable[..., None]] = None,
    log_path: Optional[Path] = None,
    tuning_path: Optional[Path] = None,
    epsilon: float = EPSILON,
    seed: Optional[int] = None,
) -> dict:
    """
    One autonomous self-tuning session. Returns a report dict; refuses to
    run (ready=False) until the rated set can support the metric.
    """
    say = progress or (lambda *a, **k: None)
    rng = random.Random(seed)

    base = _objective(graph)
    if base is None:
        return {"ready": False,
                "note": "Need >=6 rated papers with >=2 positives and "
                        ">=2 negatives before self-tuning can measure "
                        "anything."}

    say(f"Baseline preference-alignment score: {base:.4f}", 0.02)
    _log_row([time.strftime("%Y-%m-%d %H:%M:%S"), "baseline", "", "",
              base, base, "baseline"], log_path)

    best = base
    kept: dict[str, float] = {}
    names = list(PARAM_SPECS.keys())
    experiments = []

    for i in range(int(rounds)):
        name = names[i % len(names)]
        spec = PARAM_SPECS[name]
        old = spec["get"](graph)
        new = propose(spec, old, rng)
        if new == old:
            continue
        say(f"[{i + 1}/{rounds}] trying {name}: {old} → {new}",
            0.05 + 0.9 * i / max(rounds, 1))
        _apply(graph, name, new)
        score = _objective(graph)

        if score is not None and score > best + epsilon:
            status, best = "keep", score
            kept[name] = new
        else:
            status = "discard"
            _apply(graph, name, old)          # revert, recompute state
        _log_row([time.strftime("%Y-%m-%d %H:%M:%S"), name, old, new,
                  score if score is not None else "", best, status], log_path)
        experiments.append({"param": name, "old": old, "new": new,
                            "objective": score, "status": status})

    if kept:
        tuning = load_tuning(tuning_path)
        tuning.update(kept)
        save_tuning(tuning, tuning_path)
    audit.log_event("autotune", rounds=int(rounds), baseline=base,
                    best=best, kept=list(kept.keys()))
    say(f"Self-tuning done: {base:.4f} → {best:.4f} "
        f"({len(kept)} change(s) kept)", 1.0)
    return {"ready": True, "baseline": base, "best": best,
            "improved": best > base, "kept": kept,
            "experiments": experiments}
