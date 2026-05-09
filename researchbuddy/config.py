from pathlib import Path
import os as _os


def _env_bool(name: str, default: bool) -> bool:
    raw = _os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = _os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value

# ── User data directory ────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / ".researchbuddy"
STATE_FILE = DATA_DIR / "research_graph.pkl"
TEMP_DIR   = DATA_DIR / "temp_papers"
HISTORY_DIR = DATA_DIR / "history"
# Most evolution analysis only needs ~25 numeric fields per save, which we
# now write to history/evolution.jsonl (~1 KB per line). We keep only a few
# full pickles for emergency recovery; everything else is reconstructable
# from the JSONL log + the canonical pickle.
STATE_HISTORY_KEEP = _env_int("RESEARCHBUDDY_HISTORY_KEEP", 3, min_value=0)

# ── PDF output paths (one per network) ────────────────────────────────────────
SEMANTIC_PDF  = DATA_DIR / "network_semantic.pdf"   # NLP / HSWN
CITATION_PDF  = DATA_DIR / "network_citation.pdf"   # directed citation graph
COMBINED_PDF  = DATA_DIR / "network_combined.pdf"   # fused (semantic + citation)

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "nomic-ai/nomic-embed-text-v1.5"   # 768-dim, 8192-token context
EMBEDDING_DIM        = 768   # expected output dim; bump when changing model
# Set to 0 (default) to let ResearchBuddy pick a VRAM-appropriate batch size.
# Any positive value overrides the auto-tuner.
EMBEDDING_BATCH_SIZE = _env_int("RESEARCHBUDDY_EMBEDDING_BATCH_SIZE", 0, min_value=0)
# 0 = let the auto-tuner pick (cap based on VRAM); >0 forces a specific cap
EMBEDDING_MAX_SEQ_LENGTH = _env_int("RESEARCHBUDDY_EMBEDDING_MAX_SEQ_LENGTH", 0, min_value=0)
EMBEDDING_CPU_FALLBACK_ON_OOM = _env_bool("RESEARCHBUDDY_EMBEDDING_CPU_FALLBACK_ON_OOM", True)
# Precision: "auto" picks fp16 on GPUs with <12GB VRAM, fp32 otherwise.
# Force with "fp16", "fp32", or "bf16".
EMBEDDING_PRECISION = _os.getenv("RESEARCHBUDDY_EMBEDDING_PRECISION", "auto").strip().lower()
if EMBEDDING_PRECISION not in ("auto", "fp16", "fp32", "bf16"):
    EMBEDDING_PRECISION = "auto"

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
# One-at-a-time by design: after each suggestion the user is asked to rate
# AND (optionally) supply the PDF. Providing the PDF triggers a full GROBID
# extraction so the rated paper enters the corpus as a real graph node with
# section embeddings + parsed references — that's how the graph actually
# grows over time. Override with --n-recommendations N for batch mode.
N_RECOMMENDATIONS    = _env_int("RESEARCHBUDDY_N_RECOMMENDATIONS", 1, min_value=1)
EXPLORATION_RATIO    = 0.25    # fraction of suggestions that are exploratory
MIN_NOVELTY_DISTANCE = 0.30    # min distance from graph for "explore" papers

# ── Learned scoring weights ───────────────────────────────────────────────────
WEIGHT_LEARNING_MIN_RATINGS = 8   # minimum rated papers before learning kicks in
WEIGHT_LEARNING_REGULARIZATION = 0.1  # L2 regularization strength

# Section types that get their own scoring dimension (4 highest-signal ones).
# A user-preference context vector is built per section, and each candidate
# paper gets a signal = cos(user_section_ctx, candidate_section_emb). Weight
# learning then discovers which sections matter most to *this* user.
# Order is the canonical signal order for the extended weight vector.
SCORED_SECTION_TYPES = ["methods", "results", "discussion", "introduction"]

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
# OpenAlex — 250M+ works, no key required, topic-aware, fast. Primary
# replacement for S2 when S2 throttles. Polite pool: provide an email via
# OPENALEX_MAILTO env var to get faster responses (recommended).
OPENALEX_SEARCH_QUERIES = _env_int("RESEARCHBUDDY_OPENALEX_QUERIES", 4, min_value=0)
OPENALEX_SEARCH_LIMIT   = _env_int("RESEARCHBUDDY_OPENALEX_LIMIT", 25, min_value=1)
# CrossRef — 150M works, no key, complementary coverage (publishers'
# perspective vs OpenAlex's aggregator perspective). Returns rich
# bibliographic metadata + abstracts where available.
CROSSREF_SEARCH_QUERIES = _env_int("RESEARCHBUDDY_CROSSREF_QUERIES", 2, min_value=0)
CROSSREF_SEARCH_LIMIT   = _env_int("RESEARCHBUDDY_CROSSREF_LIMIT", 15, min_value=1)
CROSSREF_API_URL        = "https://api.crossref.org/works"

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

# ── GROBID (academic-PDF parser — falls back to pdfplumber if unavailable) ───
# GROBID is an ML library specialised for scientific PDFs. It outperforms
# pdfplumber on title/abstract/section detection, figures, tables, equations,
# and reference parsing. Run it as a Docker container:
#   docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.1
GROBID_ENABLED = _env_bool("RESEARCHBUDDY_GROBID_ENABLED", True)
GROBID_URL     = _os.getenv("RESEARCHBUDDY_GROBID_URL", "http://localhost:8070").rstrip("/")
# Generous default — GROBID's first request triggers ML model loading
# (~30s on CPU); long PDFs add another 30–60s on top. Subsequent requests
# are much faster. Bump this if you're running GROBID on a slow CPU or
# parsing book-length documents.
GROBID_TIMEOUT = _env_int("RESEARCHBUDDY_GROBID_TIMEOUT", 180, min_value=10)
# Send a tiny warmup request after GROBID becomes reachable so the first
# real PDF doesn't pay the model-load cost. Disable if undesired.
GROBID_WARMUP  = _env_bool("RESEARCHBUDDY_GROBID_WARMUP", True)

# ── Neo4j (optional — falls back to NetworkX if unavailable) ─────────────────
NEO4J_ENABLED        = _os.getenv("RESEARCHBUDDY_NEO4J_ENABLED", "").lower() in ("1", "true", "yes")
NEO4J_URI            = _os.getenv("RESEARCHBUDDY_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER           = _os.getenv("RESEARCHBUDDY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD       = _os.getenv("RESEARCHBUDDY_NEO4J_PASSWORD", "")
NEO4J_DATABASE       = _os.getenv("RESEARCHBUDDY_NEO4J_DATABASE", "neo4j")
NON_GRAPH_STATE_FILE = DATA_DIR / "non_graph_state.pkl"

# ── Embedding device ─────────────────────────────────────────────────────────
_emb_device = _os.getenv(
    "RESEARCHBUDDY_EMBEDDING_DEVICE",
    _os.getenv("EMBEDDING_DEVICE", "auto"),
).strip().lower()
EMBEDDING_DEVICE = _emb_device if _emb_device in ("auto", "cuda", "cpu") else "auto"

# ── Keyword extraction ─────────────────────────────────────────────────────────
TOP_KEYWORDS         = 8

# ── Visualization ─────────────────────────────────────────────────────────────
SAVE_GRAPH_PDF       = True    # disable with --no-plot CLI flag
