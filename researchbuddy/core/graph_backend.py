"""
graph_backend.py — Graph storage abstraction layer.

Defines a GraphBackend protocol with two implementations:
  * NetworkXBackend — wraps the existing 4 nx.DiGraph objects (zero behavior change)
  * Neo4jBackend   — stores graph data in Neo4j, translates operations to Cypher

The backend handles nodes, edges, traversals, and persistence.
Scoring math (embeddings, SNF, weight learning) stays in numpy.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Layer names used throughout
LAYER_SEMANTIC = "semantic"
LAYER_CITATION = "citation"
LAYER_COMBINED = "combined"
LAYER_CAUSAL   = "causal"
ALL_LAYERS = (LAYER_SEMANTIC, LAYER_CITATION, LAYER_COMBINED, LAYER_CAUSAL)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage operations."""

    # Node operations
    def add_node(self, node_id: str, layer: str, **attrs: Any) -> None: ...
    def has_node(self, node_id: str, layer: str) -> bool: ...
    def get_node_attr(self, node_id: str, attr: str, layer: str) -> Any: ...
    def set_node_attr(self, node_id: str, attr: str, value: Any, layer: str) -> None: ...
    def node_count(self, layer: str) -> int: ...

    # Edge operations
    def add_edge(self, u: str, v: str, layer: str, **attrs: Any) -> None: ...
    def has_edge(self, u: str, v: str, layer: str) -> bool: ...
    def get_edge_attr(self, u: str, v: str, attr: str, layer: str) -> Any: ...
    def set_edge_attr(self, u: str, v: str, attr: str, value: Any, layer: str) -> None: ...
    def edges_data(self, layer: str) -> Iterator[tuple[str, str, dict]]: ...
    def edge_count(self, layer: str) -> int: ...
    def remove_edge(self, u: str, v: str, layer: str) -> None: ...
    def degree(self, node_id: str, layer: str) -> int: ...

    # Batch operations
    def add_nodes_batch(self, nodes: list[tuple[str, dict]], layer: str) -> None: ...
    def add_edges_batch(self, edges: list[tuple[str, str, dict]], layer: str) -> None: ...
    def clear_layer(self, layer: str) -> None: ...

    # Graph algorithms
    def shortest_path(self, source: str, target: str, layer: str) -> Optional[list[str]]: ...
    def pagerank(self, layer: str, node_filter: Optional[set[str]] = None) -> dict[str, float]: ...
    def is_dag(self, layer: str) -> bool: ...
    def find_cycle(self, layer: str) -> Optional[list[tuple[str, str]]]: ...

    # Subgraph
    def subgraph_edges(self, node_ids: set[str], layer: str) -> Iterator[tuple[str, str, dict]]: ...

    # Export to NetworkX (for visualization and algorithms that need it)
    def to_networkx(self, layer: str) -> nx.DiGraph: ...

    # Persistence
    def sync(self) -> None: ...
    def close(self) -> None: ...

    # Backend identification
    @property
    def backend_name(self) -> str: ...


# ── NetworkX Backend ──────────────────────────────────────────────────────────

class NetworkXBackend:
    """
    Wraps 4 nx.DiGraph objects with the GraphBackend interface.
    Zero overhead — all calls delegate directly to NetworkX.
    """

    def __init__(self) -> None:
        self._graphs: dict[str, nx.DiGraph] = {
            LAYER_SEMANTIC: nx.DiGraph(),
            LAYER_CITATION: nx.DiGraph(),
            LAYER_COMBINED: nx.DiGraph(),
            LAYER_CAUSAL:   nx.DiGraph(),
        }

    @property
    def backend_name(self) -> str:
        return "NetworkX"

    def _g(self, layer: str) -> nx.DiGraph:
        return self._graphs[layer]

    # ── Node operations ──────────────────────────────────────────────────

    def add_node(self, node_id: str, layer: str, **attrs: Any) -> None:
        self._g(layer).add_node(node_id, **attrs)

    def has_node(self, node_id: str, layer: str) -> bool:
        return self._g(layer).has_node(node_id)

    def get_node_attr(self, node_id: str, attr: str, layer: str) -> Any:
        return self._g(layer).nodes[node_id].get(attr)

    def set_node_attr(self, node_id: str, attr: str, value: Any, layer: str) -> None:
        if self._g(layer).has_node(node_id):
            self._g(layer).nodes[node_id][attr] = value

    def node_count(self, layer: str) -> int:
        return self._g(layer).number_of_nodes()

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(self, u: str, v: str, layer: str, **attrs: Any) -> None:
        self._g(layer).add_edge(u, v, **attrs)

    def has_edge(self, u: str, v: str, layer: str) -> bool:
        return self._g(layer).has_edge(u, v)

    def get_edge_attr(self, u: str, v: str, attr: str, layer: str) -> Any:
        if self._g(layer).has_edge(u, v):
            return self._g(layer)[u][v].get(attr)
        return None

    def set_edge_attr(self, u: str, v: str, attr: str, value: Any, layer: str) -> None:
        if self._g(layer).has_edge(u, v):
            self._g(layer)[u][v][attr] = value

    def edges_data(self, layer: str) -> Iterator[tuple[str, str, dict]]:
        yield from self._g(layer).edges(data=True)

    def edge_count(self, layer: str) -> int:
        return self._g(layer).number_of_edges()

    def remove_edge(self, u: str, v: str, layer: str) -> None:
        if self._g(layer).has_edge(u, v):
            self._g(layer).remove_edge(u, v)

    def degree(self, node_id: str, layer: str) -> int:
        G = self._g(layer)
        return G.degree(node_id) if G.has_node(node_id) else 0

    # ── Batch operations ─────────────────────────────────────────────────

    def add_nodes_batch(self, nodes: list[tuple[str, dict]], layer: str) -> None:
        self._g(layer).add_nodes_from(nodes)

    def add_edges_batch(self, edges: list[tuple[str, str, dict]], layer: str) -> None:
        self._g(layer).add_edges_from(edges)

    def clear_layer(self, layer: str) -> None:
        self._graphs[layer] = nx.DiGraph()

    # ── Graph algorithms ─────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str, layer: str) -> Optional[list[str]]:
        G = self._g(layer)
        if not G.has_node(source) or not G.has_node(target):
            return None
        try:
            return nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            return None

    def pagerank(self, layer: str, node_filter: Optional[set[str]] = None) -> dict[str, float]:
        G = self._g(layer)
        if node_filter:
            G = G.subgraph(n for n in G.nodes if n in node_filter)
        if G.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            return {}

    def is_dag(self, layer: str) -> bool:
        G = self._g(layer)
        if G.number_of_edges() == 0:
            return True
        return nx.is_directed_acyclic_graph(G)

    def find_cycle(self, layer: str) -> Optional[list[tuple[str, str]]]:
        try:
            cycle = nx.find_cycle(self._g(layer), orientation="original")
            return [(u, v) for u, v, _ in cycle]
        except nx.NetworkXNoCycle:
            return None

    # ── Subgraph ─────────────────────────────────────────────────────────

    def subgraph_edges(self, node_ids: set[str], layer: str) -> Iterator[tuple[str, str, dict]]:
        G = self._g(layer)
        sub = G.subgraph(n for n in G.nodes if n in node_ids)
        yield from sub.edges(data=True)

    # ── Export ───────────────────────────────────────────────────────────

    def to_networkx(self, layer: str) -> nx.DiGraph:
        return self._g(layer)

    def set_from_networkx(self, layer: str, G: nx.DiGraph) -> None:
        """Replace an entire layer with a pre-built NetworkX DiGraph."""
        self._graphs[layer] = G

    # ── Persistence ──────────────────────────────────────────────────────

    def sync(self) -> None:
        pass  # no-op for in-memory backend

    def close(self) -> None:
        pass  # no-op


# ── Neo4j Backend ─────────────────────────────────────────────────────────────

# Relationship type mapping per layer
_NEO4J_REL_TYPES = {
    LAYER_SEMANTIC: {
        "semantic":    "SEM_SIMILARITY",
        "member":      "SEM_MEMBER",
        "cluster_sim": "SEM_CLUSTER_SIM",
        "shortcut":    "SEM_SHORTCUT",
        "_default":    "SEM_SIMILARITY",
    },
    LAYER_CITATION: {
        "citation":     "CIT_CITATION",
        "bib_coupling": "CIT_BIB_COUPLING",
        "_default":     "CIT_CITATION",
    },
    LAYER_CAUSAL: {
        "_default": "CAUSAL_INFLUENCE",
    },
}

# For the combined layer, we query both SEM_* and CIT_* relationship types
_COMBINED_REL_PATTERN = "SEM_SIMILARITY|SEM_MEMBER|SEM_CLUSTER_SIM|SEM_SHORTCUT|CIT_CITATION|CIT_BIB_COUPLING"


def _rel_type_for(layer: str, etype: str = "") -> str:
    """Map a (layer, etype) pair to a Neo4j relationship type string."""
    layer_map = _NEO4J_REL_TYPES.get(layer, {})
    return layer_map.get(etype, layer_map.get("_default", "RELATED"))


def _rel_types_for_layer(layer: str) -> list[str]:
    """All relationship type strings for a given layer."""
    if layer == LAYER_COMBINED:
        return _COMBINED_REL_PATTERN.split("|")
    layer_map = _NEO4J_REL_TYPES.get(layer, {})
    return [v for k, v in layer_map.items() if k != "_default"]


def _rel_pattern(layer: str) -> str:
    """Cypher relationship type pattern for MATCH clauses."""
    types = _rel_types_for_layer(layer)
    if not types:
        return ""
    return ":" + "|".join(types)


class Neo4jBackend:
    """
    Neo4j-backed graph storage.

    Nodes are stored as :Paper or :Cluster (determined by node_type attr).
    Relationships are prefixed by layer: SEM_*, CIT_*, CAUSAL_*.
    The combined layer is a virtual union of SEM_* and CIT_* relationships.

    Write operations are buffered and flushed on sync() for transaction efficiency.
    Read operations go directly to Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        try:
            import neo4j as neo4j_driver
        except ImportError:
            raise ImportError(
                "neo4j package required for Neo4jBackend. "
                "Install with: pip install researchbuddy[neo4j]"
            )

        self._driver = neo4j_driver.GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info("Neo4j connected: %s (database=%s)", uri, database)

        # Write buffers
        self._node_buffer: list[tuple[str, str, dict]] = []   # (node_id, layer, attrs)
        self._edge_buffer: list[tuple[str, str, str, dict]] = []  # (u, v, layer, attrs)
        self._clear_pending: set[str] = set()

        # NetworkX snapshot cache (for to_networkx)
        self._nx_cache: dict[str, nx.DiGraph] = {}
        self._nx_dirty: set[str] = set(ALL_LAYERS)

        # Initialize schema
        self._ensure_schema()

    @property
    def backend_name(self) -> str:
        return "Neo4j"

    def _ensure_schema(self) -> None:
        """Create indexes and constraints if they don't exist."""
        with self._driver.session(database=self._database) as session:
            # Unique constraints
            session.run(
                "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS "
                "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS "
                "FOR (c:Cluster) REQUIRE c.node_id IS UNIQUE"
            )
            # Indexes for common lookups
            session.run(
                "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)"
            )
            session.run(
                "CREATE INDEX paper_s2_id IF NOT EXISTS FOR (p:Paper) ON (p.s2_id)"
            )
            # Full-text index for search
            try:
                session.run(
                    "CREATE FULLTEXT INDEX paper_text IF NOT EXISTS "
                    "FOR (p:Paper) ON EACH [p.title, p.abstract]"
                )
            except Exception:
                pass  # may already exist or not be supported

    def _run(self, query: str, **params: Any) -> list[dict]:
        """Execute a Cypher query and return list of record dicts."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def _write(self, query: str, **params: Any) -> None:
        """Execute a write Cypher query."""
        with self._driver.session(database=self._database) as session:
            session.run(query, **params)

    def _invalidate_cache(self, layer: str) -> None:
        self._nx_dirty.add(layer)
        if layer in (LAYER_SEMANTIC, LAYER_CITATION):
            self._nx_dirty.add(LAYER_COMBINED)

    # ── Node operations ──────────────────────────────────────────────────

    def add_node(self, node_id: str, layer: str, **attrs: Any) -> None:
        node_type = attrs.get("node_type", "paper")
        label = "Paper" if node_type == "paper" else "Cluster"
        id_field = "paper_id" if label == "Paper" else "node_id"

        props = {k: v for k, v in attrs.items()
                 if v is not None and not isinstance(v, (np.ndarray, np.generic))}
        props[id_field] = node_id

        self._write(
            f"MERGE (n:{label} {{{id_field}: $nid}}) SET n += $props",
            nid=node_id, props=props,
        )
        self._invalidate_cache(layer)

    def has_node(self, node_id: str, layer: str) -> bool:
        result = self._run(
            "MATCH (n) WHERE n.paper_id = $nid OR n.node_id = $nid RETURN count(n) AS c",
            nid=node_id,
        )
        return result[0]["c"] > 0 if result else False

    def get_node_attr(self, node_id: str, attr: str, layer: str) -> Any:
        result = self._run(
            f"MATCH (n) WHERE n.paper_id = $nid OR n.node_id = $nid RETURN n.{attr} AS val",
            nid=node_id,
        )
        return result[0]["val"] if result else None

    def set_node_attr(self, node_id: str, attr: str, value: Any, layer: str) -> None:
        if isinstance(value, (np.ndarray, np.generic)):
            return  # don't store numpy arrays in Neo4j
        self._write(
            f"MATCH (n) WHERE n.paper_id = $nid OR n.node_id = $nid SET n.{attr} = $val",
            nid=node_id, val=value,
        )
        self._invalidate_cache(layer)

    def node_count(self, layer: str) -> int:
        # Node count is global (nodes aren't layer-specific in Neo4j)
        result = self._run("MATCH (n) RETURN count(n) AS c")
        return result[0]["c"] if result else 0

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(self, u: str, v: str, layer: str, **attrs: Any) -> None:
        if layer == LAYER_COMBINED:
            return  # combined is a virtual layer
        etype = attrs.get("etype", "")
        rel_type = _rel_type_for(layer, etype)
        props = {k: v for k, v in attrs.items()
                 if v is not None and not isinstance(v, (np.ndarray, np.generic))}
        self._write(
            f"MATCH (a) WHERE a.paper_id = $u OR a.node_id = $u "
            f"MATCH (b) WHERE b.paper_id = $v OR b.node_id = $v "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
            u=u, v=v, props=props,
        )
        self._invalidate_cache(layer)

    def has_edge(self, u: str, v: str, layer: str) -> bool:
        rel_pat = _rel_pattern(layer)
        result = self._run(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"WHERE (a.paper_id = $u OR a.node_id = $u) "
            f"AND (b.paper_id = $v OR b.node_id = $v) "
            f"RETURN count(r) AS c",
            u=u, v=v,
        )
        return result[0]["c"] > 0 if result else False

    def get_edge_attr(self, u: str, v: str, attr: str, layer: str) -> Any:
        rel_pat = _rel_pattern(layer)
        result = self._run(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"WHERE (a.paper_id = $u OR a.node_id = $u) "
            f"AND (b.paper_id = $v OR b.node_id = $v) "
            f"RETURN r.{attr} AS val LIMIT 1",
            u=u, v=v,
        )
        return result[0]["val"] if result else None

    def set_edge_attr(self, u: str, v: str, attr: str, value: Any, layer: str) -> None:
        if layer == LAYER_COMBINED:
            return
        if isinstance(value, (np.ndarray, np.generic)):
            return
        rel_pat = _rel_pattern(layer)
        self._write(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"WHERE (a.paper_id = $u OR a.node_id = $u) "
            f"AND (b.paper_id = $v OR b.node_id = $v) "
            f"SET r.{attr} = $val",
            u=u, v=v, val=value,
        )
        self._invalidate_cache(layer)

    def edges_data(self, layer: str) -> Iterator[tuple[str, str, dict]]:
        rel_pat = _rel_pattern(layer)
        results = self._run(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"RETURN coalesce(a.paper_id, a.node_id) AS u, "
            f"       coalesce(b.paper_id, b.node_id) AS v, "
            f"       properties(r) AS props"
        )
        for row in results:
            yield row["u"], row["v"], row["props"]

    def edge_count(self, layer: str) -> int:
        rel_pat = _rel_pattern(layer)
        result = self._run(
            f"MATCH ()-[r{rel_pat}]->() RETURN count(r) AS c"
        )
        return result[0]["c"] if result else 0

    def remove_edge(self, u: str, v: str, layer: str) -> None:
        if layer == LAYER_COMBINED:
            return
        rel_pat = _rel_pattern(layer)
        self._write(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"WHERE (a.paper_id = $u OR a.node_id = $u) "
            f"AND (b.paper_id = $v OR b.node_id = $v) "
            f"DELETE r",
            u=u, v=v,
        )
        self._invalidate_cache(layer)

    def degree(self, node_id: str, layer: str) -> int:
        rel_pat = _rel_pattern(layer)
        result = self._run(
            f"MATCH (n)-[r{rel_pat}]-(m) "
            f"WHERE n.paper_id = $nid OR n.node_id = $nid "
            f"RETURN count(r) AS d",
            nid=node_id,
        )
        return result[0]["d"] if result else 0

    # ── Batch operations ─────────────────────────────────────────────────

    def add_nodes_batch(self, nodes: list[tuple[str, dict]], layer: str) -> None:
        if not nodes:
            return
        papers = []
        clusters = []
        for nid, attrs in nodes:
            node_type = attrs.get("node_type", "paper")
            props = {k: v for k, v in attrs.items()
                     if v is not None and not isinstance(v, (np.ndarray, np.generic))}
            if node_type == "paper":
                props["paper_id"] = nid
                papers.append(props)
            else:
                props["node_id"] = nid
                clusters.append(props)

        with self._driver.session(database=self._database) as session:
            if papers:
                session.run(
                    "UNWIND $batch AS p "
                    "MERGE (n:Paper {paper_id: p.paper_id}) SET n += p",
                    batch=papers,
                )
            if clusters:
                session.run(
                    "UNWIND $batch AS c "
                    "MERGE (n:Cluster {node_id: c.node_id}) SET n += c",
                    batch=clusters,
                )
        self._invalidate_cache(layer)

    def add_edges_batch(self, edges: list[tuple[str, str, dict]], layer: str) -> None:
        if not edges or layer == LAYER_COMBINED:
            return

        # Group edges by relationship type for efficient batching
        by_type: dict[str, list[dict]] = {}
        for u, v, attrs in edges:
            etype = attrs.get("etype", "")
            rel_type = _rel_type_for(layer, etype)
            props = {k: val for k, val in attrs.items()
                     if val is not None and not isinstance(val, (np.ndarray, np.generic))}
            props["_u"] = u
            props["_v"] = v
            by_type.setdefault(rel_type, []).append(props)

        with self._driver.session(database=self._database) as session:
            for rel_type, batch in by_type.items():
                session.run(
                    f"UNWIND $batch AS e "
                    f"MATCH (a) WHERE a.paper_id = e._u OR a.node_id = e._u "
                    f"MATCH (b) WHERE b.paper_id = e._v OR b.node_id = e._v "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"SET r += apoc.map.removeKeys(e, ['_u', '_v'])",
                    batch=batch,
                )
        self._invalidate_cache(layer)

    def clear_layer(self, layer: str) -> None:
        if layer == LAYER_COMBINED:
            return  # virtual layer
        rel_types = _rel_types_for_layer(layer)
        with self._driver.session(database=self._database) as session:
            for rt in rel_types:
                session.run(f"MATCH ()-[r:{rt}]->() DELETE r")
            # Remove cluster nodes if clearing semantic layer
            if layer == LAYER_SEMANTIC:
                session.run("MATCH (c:Cluster) DETACH DELETE c")
        self._invalidate_cache(layer)

    # ── Graph algorithms ─────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str, layer: str) -> Optional[list[str]]:
        rel_pat = _rel_pattern(layer)
        result = self._run(
            f"MATCH (a), (b) "
            f"WHERE (a.paper_id = $src OR a.node_id = $src) "
            f"AND (b.paper_id = $tgt OR b.node_id = $tgt) "
            f"MATCH p = shortestPath((a)-[{rel_pat}*..20]->(b)) "
            f"RETURN [n IN nodes(p) | coalesce(n.paper_id, n.node_id)] AS path",
            src=source, tgt=target,
        )
        return result[0]["path"] if result else None

    def pagerank(self, layer: str, node_filter: Optional[set[str]] = None) -> dict[str, float]:
        # Fallback to NetworkX-based PageRank via snapshot
        G = self.to_networkx(layer)
        if node_filter:
            G = G.subgraph(n for n in G.nodes if n in node_filter)
        if G.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            return {}

    def is_dag(self, layer: str) -> bool:
        G = self.to_networkx(layer)
        if G.number_of_edges() == 0:
            return True
        return nx.is_directed_acyclic_graph(G)

    def find_cycle(self, layer: str) -> Optional[list[tuple[str, str]]]:
        G = self.to_networkx(layer)
        try:
            cycle = nx.find_cycle(G, orientation="original")
            return [(u, v) for u, v, _ in cycle]
        except nx.NetworkXNoCycle:
            return None

    # ── Subgraph ─────────────────────────────────────────────────────────

    def subgraph_edges(self, node_ids: set[str], layer: str) -> Iterator[tuple[str, str, dict]]:
        rel_pat = _rel_pattern(layer)
        id_list = list(node_ids)
        results = self._run(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"WHERE (a.paper_id IN $ids OR a.node_id IN $ids) "
            f"AND (b.paper_id IN $ids OR b.node_id IN $ids) "
            f"RETURN coalesce(a.paper_id, a.node_id) AS u, "
            f"       coalesce(b.paper_id, b.node_id) AS v, "
            f"       properties(r) AS props",
            ids=id_list,
        )
        for row in results:
            yield row["u"], row["v"], row["props"]

    # ── Export to NetworkX ────────────────────────────────────────────────

    def to_networkx(self, layer: str) -> nx.DiGraph:
        if layer not in self._nx_dirty and layer in self._nx_cache:
            return self._nx_cache[layer]

        G = nx.DiGraph()

        # Load all nodes
        nodes = self._run(
            "MATCH (n) RETURN labels(n) AS labels, properties(n) AS props"
        )
        for row in nodes:
            props = row["props"]
            nid = props.get("paper_id") or props.get("node_id")
            if nid:
                G.add_node(nid, **props)

        # Load edges for this layer
        rel_pat = _rel_pattern(layer)
        edges = self._run(
            f"MATCH (a)-[r{rel_pat}]->(b) "
            f"RETURN coalesce(a.paper_id, a.node_id) AS u, "
            f"       coalesce(b.paper_id, b.node_id) AS v, "
            f"       properties(r) AS props"
        )
        for row in edges:
            G.add_edge(row["u"], row["v"], **row["props"])

        self._nx_cache[layer] = G
        self._nx_dirty.discard(layer)
        return G

    def set_from_networkx(self, layer: str, G: nx.DiGraph) -> None:
        """Replace an entire layer from a pre-built NetworkX DiGraph."""
        self.clear_layer(layer)

        # Batch insert nodes
        nodes = []
        for nid, attrs in G.nodes(data=True):
            nodes.append((nid, dict(attrs)))
        if nodes:
            self.add_nodes_batch(nodes, layer)

        # Batch insert edges
        edges = []
        for u, v, attrs in G.edges(data=True):
            edges.append((u, v, dict(attrs)))
        if edges:
            self.add_edges_batch(edges, layer)

        self._invalidate_cache(layer)

    # ── Persistence ──────────────────────────────────────────────────────

    def sync(self) -> None:
        pass  # Neo4j writes are immediate (no buffering in this implementation)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── Factory ───────────────────────────────────────────────────────────────────

def create_backend() -> GraphBackend:
    """
    Create the appropriate graph backend based on configuration.
    Returns Neo4jBackend if configured and available, else NetworkXBackend.
    """
    try:
        from researchbuddy.config import (
            NEO4J_ENABLED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
        )
    except ImportError:
        return NetworkXBackend()

    if not NEO4J_ENABLED:
        return NetworkXBackend()

    # Try environment variable override for password
    import os
    password = os.getenv("RESEARCHBUDDY_NEO4J_PASSWORD", NEO4J_PASSWORD)
    uri = os.getenv("RESEARCHBUDDY_NEO4J_URI", NEO4J_URI)

    try:
        backend = Neo4jBackend(uri=uri, user=NEO4J_USER, password=password,
                               database=NEO4J_DATABASE)
        logger.info("Using Neo4j backend (%s)", uri)
        return backend
    except ImportError:
        logger.warning(
            "Neo4j enabled in config but 'neo4j' package not installed. "
            "Install with: pip install researchbuddy[neo4j]  "
            "Falling back to NetworkX (in-memory) backend."
        )
        return NetworkXBackend()
    except Exception as e:
        logger.warning(
            "Neo4j connection failed: %s. "
            "Falling back to NetworkX (in-memory) backend. "
            "For persistent graph storage, install and start Neo4j: "
            "https://neo4j.com/download/",
            e,
        )
        return NetworkXBackend()
