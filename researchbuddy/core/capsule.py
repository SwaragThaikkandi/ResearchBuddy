"""
Graph capsules — the privacy-scrubbed interchange format two ResearchBuddy
graphs use to compare and merge. This is the contract the `social-psyche`
package builds on.

A capsule packages ONE researcher's graph so another can integrate it
*without* learning private things:

  always dropped     : thought/draft nodes (kind != "paper"), filepaths
  opt-in only        : DOIs + titles (share_identifiers), ratings (share_ratings)
  always shared       : embeddings + edge structure + cluster centroids

With identifiers off, a capsule still supports structural alignment (Gromov–
Wasserstein, spectral distance) because those use intra-graph distances only —
so collaborators can measure how compatible their landscapes are and find
analogous regions without ever exchanging reading lists.

On-disk form (`*.rbcapsule`) is a zip:
    meta.json        nodes, edges, clusters, stats, privacy flags
    arrays.npz       node embeddings + cluster centroids (float32)
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from researchbuddy.config import CAPSULE_VERSION, CAPSULE_DIR, CAPSULE_MATCH_COS
from researchbuddy.core.graph_model import HierarchicalResearchGraph, PaperMeta

logger = logging.getLogger(__name__)


class CapsuleError(ValueError):
    """Raised when a capsule is malformed, oversized, or unsupported.

    Capsules can arrive from untrusted network peers (social-psyche), so a
    bad capsule must fail loudly here — never crash deep inside a merge or
    silently corrupt the local graph."""


# Hard ceilings for untrusted input. Generous for real research graphs
# (100k papers x 768-dim float32 ≈ 300 MB), tight enough to stop zip bombs.
MAX_CAPSULE_NODES        = 100_000
MAX_CAPSULE_EDGES        = 2_000_000
MAX_DECOMPRESSED_MEMBER  = 512 * 1024 * 1024   # per zip member, bytes


# ── In-memory capsule ─────────────────────────────────────────────────────────

@dataclass
class GraphCapsule:
    version: int
    created: str
    privacy: dict                       # {share_identifiers, share_ratings}
    nodes: list[dict]                   # [{idx, doi?, title?, year?, rating?}]
    embeddings: np.ndarray              # (N, d) row i == node idx i
    edges_sem: list[tuple]             # [(i, j, w)]
    edges_cit: list[tuple]             # [(i, j, w)]
    centroids: np.ndarray               # (K, d) level-1 cluster centroids
    cluster_sizes: list[int]
    stats: dict

    # convenience views
    def doi_set(self) -> set[str]:
        return {n["doi"].lower() for n in self.nodes if n.get("doi")}

    def n_nodes(self) -> int:
        return len(self.nodes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def graph_doi_set(graph: HierarchicalResearchGraph) -> set[str]:
    """DOIs the graph holds (papers only). Used by PSI in social-psyche —
    computed from the live graph, never written into a shared capsule."""
    return {m.doi.lower() for m in graph.all_papers()
            if m.kind == "paper" and m.doi}


def _doi_to_pid(graph: HierarchicalResearchGraph) -> dict[str, str]:
    return {m.doi.lower(): m.paper_id for m in graph.all_papers()
            if m.kind == "paper" and m.doi}


# ── Export ────────────────────────────────────────────────────────────────────

def export_capsule(
    graph: HierarchicalResearchGraph,
    share_identifiers: bool = False,
    share_ratings: bool = False,
    out_path: Optional[Path] = None,
) -> GraphCapsule:
    """
    Build a (optionally on-disk) capsule from `graph`. Thought/draft nodes are
    always excluded; identifiers and ratings are opt-in.
    """
    papers = [m for m in graph.all_papers()
              if m.kind == "paper" and m.embedding is not None]
    pid_to_idx = {m.paper_id: i for i, m in enumerate(papers)}
    dim = len(papers[0].embedding) if papers else 0

    nodes: list[dict] = []
    embs = np.zeros((len(papers), dim), dtype=np.float32)
    for i, m in enumerate(papers):
        embs_i = np.asarray(m.embedding, dtype=np.float32)
        embs_i = embs_i / (np.linalg.norm(embs_i) or 1.0)
        embs[i] = embs_i
        rec: dict = {"idx": i}
        if share_identifiers:
            rec["doi"] = (m.doi or "").lower()
            rec["title"] = m.title[:250]
            rec["year"] = m.year
            rec["is_peer_reviewed"] = m.is_peer_reviewed
        if share_ratings and m.user_rating is not None:
            rec["rating"] = float(m.user_rating)
        nodes.append(rec)

    def _edges(G) -> list[tuple]:
        out = []
        for u, v, data in G.edges(data=True):
            if u in pid_to_idx and v in pid_to_idx:
                out.append((pid_to_idx[u], pid_to_idx[v],
                            float(data.get("weight", 1.0) or 1.0)))
        return out

    edges_sem = _edges(graph.G_semantic)
    edges_cit = _edges(graph.G_citation)

    # Level-1 (niche) cluster centroids — the coarse shape of the landscape.
    centroids = []
    sizes = []
    for nid, nd in getattr(graph, "_clusters", {}).items():
        if getattr(nd, "level", None) == 1 and getattr(nd, "centroid", None) is not None:
            centroids.append(np.asarray(nd.centroid, dtype=np.float32))
            sizes.append(len(nd.paper_ids))
    cent_arr = (np.vstack(centroids) if centroids
                else np.zeros((0, dim), dtype=np.float32))

    capsule = GraphCapsule(
        version=CAPSULE_VERSION,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        privacy={"share_identifiers": share_identifiers,
                 "share_ratings": share_ratings},
        nodes=nodes,
        embeddings=embs,
        edges_sem=edges_sem,
        edges_cit=edges_cit,
        centroids=cent_arr,
        cluster_sizes=sizes,
        stats={
            "n_papers": len(papers),
            "n_sem_edges": len(edges_sem),
            "n_cit_edges": len(edges_cit),
            "n_clusters": len(sizes),
            "embedding_dim": dim,
        },
    )
    if out_path is not None:
        write_capsule(capsule, out_path)
    return capsule


def dumps_capsule(capsule: GraphCapsule) -> bytes:
    """Serialise a capsule to in-memory `.rbcapsule` bytes (no disk touch).

    This is what the networked transport (social-psyche) sends over an
    encrypted channel — the capsule never lands on a shared medium.
    """
    meta = {
        "version": capsule.version,
        "created": capsule.created,
        "privacy": capsule.privacy,
        "nodes": capsule.nodes,
        "edges_sem": capsule.edges_sem,
        "edges_cit": capsule.edges_cit,
        "cluster_sizes": capsule.cluster_sizes,
        "stats": capsule.stats,
    }
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, embeddings=capsule.embeddings,
                        centroids=capsule.centroids)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False))
        zf.writestr("arrays.npz", npz_buf.getvalue())
    return out.getvalue()


def loads_capsule(data: bytes) -> GraphCapsule:
    """Inverse of dumps_capsule: parse and VALIDATE `.rbcapsule` bytes.

    Treats the input as untrusted (it may come from a network peer): guards
    against zip bombs before decompression and structural attacks after.
    Raises CapsuleError on anything suspicious.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            # Zip-bomb guard: check declared decompressed sizes BEFORE reading.
            for info in zf.infolist():
                if info.file_size > MAX_DECOMPRESSED_MEMBER:
                    raise CapsuleError(
                        f"capsule member {info.filename!r} declares "
                        f"{info.file_size} decompressed bytes (max "
                        f"{MAX_DECOMPRESSED_MEMBER}) — rejecting")
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
            arr = np.load(io.BytesIO(zf.read("arrays.npz")))   # allow_pickle=False
            embeddings = arr["embeddings"]
            centroids = arr["centroids"]
    except CapsuleError:
        raise
    except Exception as e:
        raise CapsuleError(f"not a readable capsule: {e}") from e

    try:
        capsule = GraphCapsule(
            version=int(meta["version"]),
            created=str(meta["created"]),
            privacy=dict(meta["privacy"]),
            nodes=list(meta["nodes"]),
            embeddings=embeddings,
            edges_sem=[tuple(e) for e in meta["edges_sem"]],
            edges_cit=[tuple(e) for e in meta["edges_cit"]],
            centroids=centroids,
            cluster_sizes=list(meta["cluster_sizes"]),
            stats=dict(meta["stats"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CapsuleError(f"capsule metadata malformed: {e}") from e
    validate_capsule(capsule)
    return capsule


def validate_capsule(capsule: GraphCapsule) -> None:
    """Structural validation of a (possibly hostile) capsule.

    Every index used later by merge_capsule is bounds-checked HERE, once,
    so the merge itself can trust the data. Raises CapsuleError.
    """
    if capsule.version > CAPSULE_VERSION:
        raise CapsuleError(
            f"capsule version {capsule.version} is newer than supported "
            f"({CAPSULE_VERSION}) — upgrade ResearchBuddy")

    emb = capsule.embeddings
    if not isinstance(emb, np.ndarray) or emb.ndim != 2:
        raise CapsuleError("embeddings must be a 2-D array")
    n = emb.shape[0]
    if n > MAX_CAPSULE_NODES:
        raise CapsuleError(f"{n} nodes exceeds cap {MAX_CAPSULE_NODES}")
    if len(capsule.nodes) != n:
        raise CapsuleError(
            f"node list ({len(capsule.nodes)}) and embedding rows ({n}) disagree")
    if not np.isfinite(emb).all():
        raise CapsuleError("embeddings contain NaN/Inf")

    seen_idx: set[int] = set()
    for node in capsule.nodes:
        if not isinstance(node, dict):
            raise CapsuleError("node entries must be dicts")
        idx = node.get("idx")
        if not isinstance(idx, int) or not (0 <= idx < n) or idx in seen_idx:
            raise CapsuleError(f"node idx {idx!r} out of range or duplicated")
        seen_idx.add(idx)

    for name, edges in (("edges_sem", capsule.edges_sem),
                        ("edges_cit", capsule.edges_cit)):
        if len(edges) > MAX_CAPSULE_EDGES:
            raise CapsuleError(f"{name}: {len(edges)} edges exceeds cap")
        for e in edges:
            if len(e) != 3:
                raise CapsuleError(f"{name}: edge {e!r} malformed")
            i, j, w = e
            if not (isinstance(i, int) and isinstance(j, int)
                    and 0 <= i < n and 0 <= j < n):
                raise CapsuleError(f"{name}: edge endpoints {i!r},{j!r} invalid")
            if not np.isfinite(float(w)):
                raise CapsuleError(f"{name}: non-finite edge weight")

    cent = capsule.centroids
    if not isinstance(cent, np.ndarray) or cent.ndim != 2:
        raise CapsuleError("centroids must be a 2-D array")
    if n and cent.shape[0] and cent.shape[1] != emb.shape[1]:
        raise CapsuleError("centroid dim disagrees with embedding dim")
    if not np.isfinite(cent).all():
        raise CapsuleError("centroids contain NaN/Inf")


def write_capsule(capsule: GraphCapsule, out_path: Path) -> Path:
    out_path = Path(out_path)
    if out_path.suffix != ".rbcapsule":
        out_path = out_path.with_suffix(".rbcapsule")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(dumps_capsule(capsule))
    return out_path


def load_capsule(path: Path) -> GraphCapsule:
    return loads_capsule(Path(path).read_bytes())


# ── Reliability report ────────────────────────────────────────────────────────

@dataclass
class MergeReport:
    # what happened
    shared_by_doi: int = 0
    shared_by_embedding: int = 0
    imported: int = 0
    novel_regions: int = 0
    # graph-theoretic reliability measures
    jaccard_doi: Optional[float] = None
    spectral_distance: Optional[float] = None
    deltacon_similarity: Optional[float] = None
    degree_ks: Optional[float] = None
    gw_distortion: Optional[float] = None
    modularity_self: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _capsule_nx(capsule: GraphCapsule):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(capsule.n_nodes()))
    for i, j, w in capsule.edges_sem:
        G.add_edge(int(i), int(j), weight=float(w))
    return G


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_capsule(
    graph: HierarchicalResearchGraph,
    capsule: GraphCapsule,
    match_cos: float = CAPSULE_MATCH_COS,
    import_new: bool = True,
    rebuild: bool = False,
) -> MergeReport:
    """
    Integrate a peer's capsule into `graph` and report how reliable the merge
    is, using standard graph-theory measures.

    - Nodes shared by DOI are recognised (not duplicated).
    - With identifiers present, genuinely-new papers are imported as
      abstract-less nodes (the user can later harvest their OA full text).
    - Without identifiers, foreign nodes are matched to local papers by
      embedding nearest-neighbour; unmatched ones are counted as novel
      regions but not imported (no identity to attach).
    """
    from researchbuddy.core import graph_distance as gd

    # Defense in depth: callers should already have validated (loads_capsule
    # does), but a capsule constructed in memory could bypass that path.
    validate_capsule(capsule)

    report = MergeReport()

    my_papers = [m for m in graph.all_papers()
                 if m.kind == "paper" and m.embedding is not None]
    my_dois = graph_doi_set(graph)
    my_embs = (np.vstack([np.asarray(m.embedding, float) /
                          (np.linalg.norm(m.embedding) or 1.0)
                          for m in my_papers])
               if my_papers else np.zeros((0, capsule.embeddings.shape[1])))

    # Embedding-space compatibility: a peer on a different embedding model
    # (e.g. 384-dim MiniLM vs 768-dim nomic) produces vectors that are
    # meaningless in our space. Importing them would poison the local graph;
    # NN-matching against them would be noise. Fall back to metadata-only.
    cap_dim = capsule.embeddings.shape[1] if capsule.n_nodes() else 0
    dims_compatible = (not my_papers) or (cap_dim == my_embs.shape[1])
    if not dims_compatible:
        report.notes.append(
            f"Embedding dim mismatch (peer {cap_dim} vs local "
            f"{my_embs.shape[1]}): peer vectors discarded; identified papers "
            "imported metadata-only (harvest full text to embed locally); "
            "structural matching skipped.")

    has_ids = capsule.privacy.get("share_identifiers", False)
    cap_dois = capsule.doi_set()

    # ── 1. Node reconciliation ────────────────────────────────────────────
    for node in capsule.nodes:
        idx = node["idx"]                      # bounds-checked by validate_capsule
        doi = (node.get("doi") or "").lower()
        if doi:
            if doi in my_dois:
                report.shared_by_doi += 1
                continue
            if import_new and node.get("title"):
                meta = PaperMeta(
                    paper_id=HierarchicalResearchGraph.make_id(
                        node["title"], doi=doi),
                    title=node["title"][:250],
                    abstract="",
                    year=node.get("year"),
                    doi=doi,
                    is_peer_reviewed=node.get("is_peer_reviewed"),
                    source="capsule",
                )
                if dims_compatible:
                    emb = np.asarray(capsule.embeddings[idx], dtype=float)
                    graph.add_paper(meta, emb)
                else:
                    graph.add_paper(meta, None)   # metadata stub; embed later
                report.imported += 1
                continue
        # No usable DOI → structural match by embedding NN.
        if my_embs.shape[0] and dims_compatible:
            v = np.asarray(capsule.embeddings[idx], float)
            v = v / (np.linalg.norm(v) or 1.0)
            best = float(np.max(my_embs @ v))
            if best >= match_cos:
                report.shared_by_embedding += 1
            else:
                report.novel_regions += 1
        else:
            report.novel_regions += 1

    # ── 2. Reliability measures ───────────────────────────────────────────
    report.jaccard_doi = gd.jaccard(my_dois, cap_dois) if (my_dois or cap_dois) else None

    A_mine = gd.adjacency_over_ids(graph.G_semantic,
                                   [m.paper_id for m in my_papers])
    cap_nx = _capsule_nx(capsule)
    A_cap = gd.adjacency_over_ids(cap_nx, list(range(capsule.n_nodes())))
    if A_mine.size and A_cap.size:
        report.spectral_distance = gd.spectral_distance(A_mine, A_cap)

    # DeltaCon needs a shared node ordering → use the DOI intersection.
    shared = sorted(my_dois & cap_dois)
    if len(shared) >= 2:
        d2p = _doi_to_pid(graph)
        cap_doi_to_idx = {(n.get("doi") or "").lower(): n["idx"]
                          for n in capsule.nodes if n.get("doi")}
        mine_ids = [d2p[d] for d in shared]
        cap_ids = [cap_doi_to_idx[d] for d in shared]
        Am = gd.adjacency_over_ids(graph.G_semantic, mine_ids)
        Ac = gd.adjacency_over_ids(cap_nx, cap_ids)
        report.deltacon_similarity = gd.deltacon_similarity(Am, Ac)

    if A_mine.size and A_cap.size:
        report.degree_ks = gd.degree_ks(A_mine, A_cap)

    # Gromov–Wasserstein over cluster centroids (compact, label-free).
    if capsule.centroids.shape[0] >= 2 and my_papers:
        my_centroids = _my_centroids(graph)
        if my_centroids.shape[0] >= 2:
            res = gd.gromov_wasserstein(
                gd.cost_from_embeddings(my_centroids),
                gd.cost_from_embeddings(capsule.centroids))
            if res is not None:
                report.gw_distortion = res["distortion"]
            else:
                report.notes.append(
                    "Gromov–Wasserstein skipped (install researchbuddy[social]).")

    report.modularity_self = gd.modularity(graph.G_semantic)

    if not has_ids:
        report.notes.append(
            "Private capsule (no identifiers): nodes aligned structurally; "
            "novel regions reported but not imported.")

    if rebuild and report.imported:
        graph.rebuild_hierarchy()

    return report


def _my_centroids(graph: HierarchicalResearchGraph) -> np.ndarray:
    cs = [np.asarray(nd.centroid, float)
          for nd in getattr(graph, "_clusters", {}).values()
          if getattr(nd, "level", None) == 1 and getattr(nd, "centroid", None) is not None]
    return np.vstack(cs) if cs else np.zeros((0, 1))
