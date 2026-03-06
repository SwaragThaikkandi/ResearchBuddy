from pathlib import Path

# ── User data directory (persists across installs / reinstalls) ────────────────
DATA_DIR   = Path.home() / ".researchbuddy"
STATE_FILE = DATA_DIR / "research_graph.pkl"
TEMP_DIR   = DATA_DIR / "temp_papers"

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, 384-dim, runs on CPU fine

# ── Graph ─────────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.45    # Min cosine similarity to draw an edge
DEFAULT_SEED_WEIGHT   = 5.0     # Implicit rating for seed PDFs (1-10 scale)
LEARNING_RATE         = 0.15    # How much a new rating shifts edge weights
MAX_EDGE_WEIGHT       = 10.0

# ── Recommendations ────────────────────────────────────────────────────────────
N_RECOMMENDATIONS     = 10      # Papers shown per search session
EXPLORATION_RATIO     = 0.25    # Fraction of suggestions that are exploratory
MIN_NOVELTY_DISTANCE  = 0.30    # Cosine distance for "exploratory" papers

# ── Search ────────────────────────────────────────────────────────────────────
S2_SEARCH_URL         = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_REC_URL            = "https://api.semanticscholar.org/recommendations/v1/papers"
S2_PAPER_URL          = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_SEARCH_URL      = "https://export.arxiv.org/api/query"
MAX_SEARCH_RESULTS    = 30      # Max candidates fetched per query
REQUEST_TIMEOUT       = 15      # Seconds
REQUEST_DELAY         = 1.0     # Polite delay between API calls (seconds)

# ── Keyword extraction ─────────────────────────────────────────────────────────
TOP_KEYWORDS          = 8       # Keywords extracted per context vector
