from pathlib import Path

# ── User data directory ────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / ".researchbuddy"
STATE_FILE = DATA_DIR / "research_graph.pkl"
TEMP_DIR   = DATA_DIR / "temp_papers"
HISTORY_DIR = DATA_DIR / "history"
STATE_HISTORY_KEEP = 200    # max timestamped state snapshots retained

# ── PDF output paths (one per network) ────────────────────────────────────────
SEMANTIC_PDF  = DATA_DIR / "network_semantic.pdf"   # NLP / HSWN
CITATION_PDF  = DATA_DIR / "network_citation.pdf"   # directed citation graph
COMBINED_PDF  = DATA_DIR / "network_combined.pdf"   # fused (semantic + citation)

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "nomic-ai/nomic-embed-text-v1.5"   # 768-dim, 8192-token context
EMBEDDING_DIM        = 768   # expected output dim; bump when changing model

# ── Graph / edges ──────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.45    # min cosine sim to draw a semantic edge
DEFAULT_SEED_WEIGHT  = 5.0     # implicit rating for unrated seed PDFs
LEARNING_RATE        = 0.15    # how fast ratings shift edge weights

# ── Adaptive Hierarchy (Hierarchical Small World Network) ─────────────────────
MIN_CLUSTER_SIZE     = 3       # minimum papers per cluster (prevents micro-niches)
MAX_HIERARCHY_LEVELS = 8       # ceiling on auto-detected levels

# ── Fusion (Similarity Network Fusion) ────────────────────────────────────────
FUSION_ALPHA         = 0.6     # weight for semantic stream  (1-alpha = citation)
SNF_KNN              = 10      # k for KNN-kernel in SNF diffusion
SNF_ITER             = 15      # diffusion iterations

# ── Recommendations ────────────────────────────────────────────────────────────
N_RECOMMENDATIONS    = 10
EXPLORATION_RATIO    = 0.25    # fraction of suggestions that are exploratory
MIN_NOVELTY_DISTANCE = 0.30    # min distance from graph for "explore" papers

# ── Learned scoring weights ───────────────────────────────────────────────────
WEIGHT_LEARNING_MIN_RATINGS = 8   # minimum rated papers before learning kicks in
WEIGHT_LEARNING_REGULARIZATION = 0.1  # L2 regularization strength

# ── Temporal decay ────────────────────────────────────────────────────────────
RATING_HALF_LIFE_DAYS  = 90    # rating influence halves every 90 days
RATING_DECAY_FLOOR     = 0.25  # minimum decay factor (never fully forget)

# ── Cold-start thresholds ─────────────────────────────────────────────────────
COLD_START_THRESHOLD   = 15    # papers below this count = cold-start mode

# ── Reasoner ("Prefrontal Cortex") ───────────────────────────────────────────
QUERY_TOP_K          = 10      # papers shown per query response
QUERY_FEEDBACK_WEIGHT= 0.3     # how much a rated query influences context vector

# ── Search ────────────────────────────────────────────────────────────────────
S2_SEARCH_URL        = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_REC_URL           = "https://api.semanticscholar.org/recommendations/v1/papers"
S2_PAPER_URL         = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_SEARCH_URL     = "https://export.arxiv.org/api/query"
OPENALEX_URL         = "https://api.openalex.org/works"   # free, no key needed

# ── CORE full-text ─────────────────────────────────────────────────────────────
# Register for a free key at https://core.ac.uk/services/api
# Set env var CORE_API_KEY=<your_key>  (anonymous access works but is slower)
CORE_API_URL         = "https://api.core.ac.uk/v3"
CORE_FULL_TEXT       = True   # enrich discovered papers with full text via CORE
MAX_SEARCH_RESULTS   = 30
REQUEST_TIMEOUT      = 15
REQUEST_DELAY        = 1.0

# ── Search allocation (peer-reviewed bias) ────────────────────────────────
S2_SEARCH_QUERIES    = 6       # number of S2 text-search queries
S2_SEARCH_LIMIT      = 20      # results per S2 query
ARXIV_SEARCH_QUERIES = 2       # number of ArXiv queries
ARXIV_SEARCH_LIMIT   = 10      # results per ArXiv query

# Reproducibility / reliability
DETERMINISTIC_MODE   = True    # stable query expansion/rerank + deterministic tie-breaks
SEARCH_CACHE_ENABLED = True    # cache LLM search helpers (query expansion + reranking)
SEARCH_CACHE_DIR     = DATA_DIR / "cache"
SEARCH_CACHE_VERSION = 1       # bump to invalidate old cache entries
# ── Arguer ("Creative Cortex") ────────────────────────────────────────────────
ARGUER_STYLE_LR      = 0.20    # EMA learning rate for StyleProfile type-weight updates
ARGUER_TOP_PARAGRAPHS= 3       # default number of argument paragraphs per session

# ── Causal DAG ("Temporal Cortex") ───────────────────────────────────────────
CAUSAL_CONFIDENCE_THRESHOLD = 0.20   # min confidence to include edge in G_causal
CAUSAL_TRANSITIVE_REDUCE    = False  # optional: remove redundant implied edges

# ── Local LLM (Ollama) ───────────────────────────────────────────────────────
LLM_OLLAMA_URL       = "http://localhost:11434"
LLM_MODEL            = "phi3.5"          # Phi-3.5-mini (~2.5GB Q4) for 4GB VRAM
LLM_TEMPERATURE      = 0.7              # creative tasks (argumentation)
LLM_TEMPERATURE_LOW  = 0.3              # structured tasks (JSON, reranking)
LLM_MAX_TOKENS       = 512              # max tokens per generation
LLM_ENABLED          = True             # master switch (--no-llm to disable)
HYDE_ENABLED          = True             # HyDE for search
LLM_QUERY_EXPANSION  = True             # LLM query expansion for search
LLM_RERANK_ENABLED   = True             # LLM reranking of search results

# ── Neo4j (optional — falls back to NetworkX if unavailable) ─────────────────
import os as _os
NEO4J_ENABLED        = _os.getenv("RESEARCHBUDDY_NEO4J_ENABLED", "").lower() in ("1", "true", "yes")
NEO4J_URI            = _os.getenv("RESEARCHBUDDY_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER           = _os.getenv("RESEARCHBUDDY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD       = _os.getenv("RESEARCHBUDDY_NEO4J_PASSWORD", "")
NEO4J_DATABASE       = _os.getenv("RESEARCHBUDDY_NEO4J_DATABASE", "neo4j")
NON_GRAPH_STATE_FILE = DATA_DIR / "non_graph_state.pkl"

# ── Embedding device ─────────────────────────────────────────────────────────
EMBEDDING_DEVICE     = "auto"           # "auto" | "cuda" | "cpu"

# ── Keyword extraction ─────────────────────────────────────────────────────────
TOP_KEYWORDS         = 8

# ── Visualization ─────────────────────────────────────────────────────────────
SAVE_GRAPH_PDF       = True    # disable with --no-plot CLI flag
