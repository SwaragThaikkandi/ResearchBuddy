# ResearchBuddy

> **v2.0.0** — Neo4j graph backend · Adaptive learned scoring · Temporal decay · Cold-start intelligence · Full-text embeddings via CORE · Hierarchical Small World Network + Causal DAG + LLM-powered Reasoning & Creative Modes

A **graph-based literature search assistant** that learns your research interests from your own PDFs and actively finds new papers for you — like a smart colleague who reads everything and brings you only what matters.

**No cloud accounts, no subscriptions, no data leaves your machine.** Everything runs locally.

---

## What does it do? (The 30-second version)

1. You give it a folder of PDFs you've already read
2. It reads them, understands what they're about, and maps how they relate to each other
3. It searches academic databases (Semantic Scholar, ArXiv) to find new papers you'd probably like
4. You rate what it finds (1–10), and it gets smarter about your preferences over time
5. It builds a visual map of your entire research landscape

Think of it as a recommendation engine — like Netflix for research papers — but one that actually understands the *content* of what you're reading, not just metadata.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Features](#features)
- [The Intelligence System](#the-intelligence-system)
- [Architecture Deep Dive](#architecture-deep-dive)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Installation

### Basic setup

```bash
# From GitHub (recommended)
pip install git+https://github.com/SwaragThaikkandi/ResearchBuddy.git

# Or clone and install in development mode
git clone https://github.com/SwaragThaikkandi/ResearchBuddy.git
cd ResearchBuddy
pip install -e .
```

**Requirements:** Python 3.9 or newer. All dependencies are installed automatically:

```
sentence-transformers  einops  networkx  pdfplumber  requests  numpy
scikit-learn  scipy  rich  keybert  matplotlib
```

On first run, the embedding model `nomic-embed-text-v1.5` (~274 MB) is downloaded automatically from HuggingFace. This only happens once.

### Optional: Richer paper embeddings (CORE API)

By default, ResearchBuddy understands papers through their abstracts. For much richer understanding, it can fetch the **full text** of papers from [CORE](https://core.ac.uk) — a free aggregator of 220M+ open-access works.

This works out of the box without any setup. For faster access and no rate limits, get a free API key:

1. Register at [core.ac.uk/services/api](https://core.ac.uk/services/api)
2. Set the environment variable:

```bash
export CORE_API_KEY=your_key_here   # Linux/macOS
set CORE_API_KEY=your_key_here      # Windows
```

### Optional: Neo4j graph database

By default, ResearchBuddy stores the research graph in memory using NetworkX (same as always). For **persistent graph storage**, **faster multi-hop traversals**, and the ability to **explore your graph in Neo4j Browser**, you can optionally use Neo4j as the backend:

```bash
# Install the Neo4j driver
pip install researchbuddy[neo4j]

# Option A: Docker (easiest)
docker run -d --name researchbuddy-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/researchbuddy \
  neo4j:5-community

# Option B: Install Neo4j Desktop from https://neo4j.com/download/
```

Then enable it:

```bash
# Set environment variables (or add to your shell profile)
export RESEARCHBUDDY_NEO4J_ENABLED=true
export RESEARCHBUDDY_NEO4J_PASSWORD=researchbuddy

# Start ResearchBuddy — it will auto-migrate your existing graph
researchbuddy
```

When Neo4j is enabled, you get:
- **Persistent storage** — your graph survives without pickle files
- **Neo4j Browser** — open `http://localhost:7474` to visually explore your research graph with Cypher queries
- **Faster traversals** — citation chains and influence paths use native graph algorithms
- **Full-text search** — Cypher-based search across paper titles and abstracts

If Neo4j is unavailable (server down, not installed), ResearchBuddy automatically falls back to NetworkX with a warning message. No data is lost.

The menu header shows which backend is active: `[Neo4j]` or `[NetworkX]`.

### Optional: LLM features (Ollama)

The Reasoning Mode (ask questions about your papers) and Creative Mode (generate argument paragraphs) require a local LLM via [Ollama](https://ollama.ai/). **ResearchBuddy works perfectly without this** — you just won't have those two features.

```bash
# Install Ollama from https://ollama.ai, then:
ollama serve              # start the server
ollama pull phi3.5        # ~2.5 GB, works on 4 GB VRAM
# or: ollama pull mistral  # ~4 GB, better quality on 8+ GB VRAM
```

When the LLM is available, it also enhances search through:
- **HyDE** — generates hypothetical abstracts for better semantic matching
- **Query expansion** — generates alternative search formulations
- **LLM reranking** — reranks results by relevance to your intent

If the LLM is unavailable, ResearchBuddy tells you clearly what's missing:
```
! LLM: UNAVAILABLE -- Connection refused
! Degraded mode: HyDE, query expansion, LLM reranking, and
! argumentation are all disabled. Search quality is reduced.
  To fix: ollama serve && ollama pull phi3.5
```

---

## Quick Start

### Step 1: Import your papers

```bash
# Point it at a folder of PDFs you've read
researchbuddy --pdf ~/papers/my-research-area/

# Or run without installing (from repo root)
python -m researchbuddy --pdf ~/papers/my-research-area/
```

ResearchBuddy will:
- Extract text from each PDF
- Generate embeddings (semantic fingerprints) for each paper
- Build an initial research graph showing how your papers relate
- Fetch full text from CORE for richer understanding (if available)

### Step 2: Search and rate

Choose option **[1] Search for new papers** from the menu. Describe what you're looking for:

```
Describe what you're looking for (research intent), or Enter to skip:
> attention mechanisms in transformer architectures for medical imaging

Optional: extra search keywords (comma-separated), or Enter to skip:
> vision transformer, self-attention, radiology
```

ResearchBuddy searches Semantic Scholar and ArXiv, then ranks results by relevance to your existing research. Rate each suggestion 1–10:

| Rating | Meaning | What happens |
|--------|---------|--------------|
| **8–10** | "This is exactly what I need" | Strong positive signal — pulls future results toward similar papers |
| **5–7** | "Somewhat relevant" | Moderate signal — noted but doesn't strongly shift direction |
| **1–4** | "Not relevant" | Negative signal — pushes future results *away* from this area |
| **0** | Skip | No signal recorded |

### Step 3: Repeat

Each time you rate papers, ResearchBuddy gets smarter. After 8+ ratings, it learns personalized scoring weights from your specific preferences. The system adapts — your research interests from 3 months ago naturally fade in influence while recent ratings drive recommendations.

### Subsequent runs

```bash
researchbuddy          # loads your saved graph automatically
researchbuddy --reset  # start fresh (clears all saved data)
```

---

## How It Works

### For the curious: the simple version

ResearchBuddy converts every paper into a list of numbers (an "embedding") that captures what the paper is about. Papers about similar topics have similar numbers. Your ratings tell it which directions in this number-space you care about, and it searches for new papers in those directions.

It also looks at which papers cite which other papers, and combines that with the content similarity to get a more complete picture. Papers that are both topically similar *and* cited by the same works are ranked higher.

### For the expert: the full picture

ResearchBuddy maintains **four interconnected networks**:

```
Your PDF folder
      |
      v  (nomic-embed-text-v1.5: 768-dim, 8192-token context)
+---------------------------------------------------------------------+
|             Semantic Network  (HSWN -- auto-levelled)               |
|                                                                     |
|  Level 3 (Domain)    [D1]----------[D2]                             |
|                       |             |                               |
|  Level 2 (Area)    [A1]  [A2]    [A3]  [A4]                        |
|                     |    |        |    |                            |
|  Level 1 (Niche) [N1][N2][N3]  [N4][N5][N6]                       |
|                   |   |   |     |   |   |                          |
|  Level 0 (Paper) [p1][p2][p3] [p4][p5][p6]                        |
|                   <-- dense intra-niche edges -->                   |
|                   <-- sparse cross-niche shortcuts -->              |
+---------------------------------------------------------------------+
      +
+---------------------------------------------------------------------+
|             Citation Network  (directed)                            |
|                                                                     |
|  [Paper A] --cites--> [Paper B] --cites--> [Paper C]               |
|      |                                          |                   |
|      +-- bibliographic coupling ----------------+                   |
|      +-- co-citation (both cited by [X]) ---------->                |
+---------------------------------------------------------------------+
      |
      v  (Similarity Network Fusion -- Wang et al. 2014)
+---------------------------------------------------------------------+
|             Combined Network  (fused, multi-modal)                  |
|  SNF iteratively diffuses information between both networks         |
|  amplifying consistent signals, dampening noise                     |
+---------------------------------------------------------------------+
      |
      v  (Temporal ordering + citation direction)
+---------------------------------------------------------------------+
|             Causal DAG  (directed acyclic graph)                    |
|  Edges oriented by publication year + citation direction            |
|  Cycles broken by removing weakest-confidence edge                  |
|  Temporal anomalies flagged automatically                           |
+---------------------------------------------------------------------+
```

After each session, **three separate PDFs** are generated in `~/.researchbuddy/`:

| File | Contents |
|------|----------|
| `network_semantic.pdf` | NLP/HSWN overview with paper layer, niche layer, and level histogram |
| `network_citation.pdf` | Directed citation graph with most-cited ranking |
| `network_combined.pdf` | Fused graph with edge-type breakdown and stats |

---

## Features

### 1. Paper Search (Menu option 1)

Search using natural language queries. ResearchBuddy:
- Fetches candidates from Semantic Scholar (recommendations + text search) and ArXiv
- Optionally generates a HyDE hypothetical abstract via LLM for better matching
- Optionally expands your query with LLM-generated alternatives
- Ranks results using a multi-signal scoring system (see [Intelligence System](#the-intelligence-system))
- Mixes in exploratory papers to prevent filter bubbles

### 2. Graph Statistics (Menu option 2)

See your graph at a glance: paper counts, hierarchy levels, edge statistics, reliability health, and the current scoring mode (cold-start, default, or learned weights).

### 3. Add PDFs (Menu option 3)

Import additional PDF folders at any time. New papers are embedded and integrated into the existing graph.

### 4. Fetch Citations (Menu option 4)

Enriches your graph with citation relationships from multiple sources (CrossRef, OpenAlex, Semantic Scholar). Citation data is cross-validated: edges verified by 2+ independent sources get higher confidence scores.

### 5. Reasoning Mode (Menu option 7) — requires LLM

Ask research questions grounded in your paper collection. The reasoner:
- Finds relevant papers using PageRank + keyword + embedding matching
- Identifies bridge papers connecting different themes
- Maps research lineages (citation chains and semantic paths)
- Detects frontier papers (relevant but underconnected)
- Generates temporal narratives of how the field developed

### 6. Creative Mode (Menu option 8) — requires LLM

Generates structured argument paragraphs that synthesize your literature. Each paragraph includes claims, evidence, and citations. Rate arguments on Correctness and Usefulness to train a style profile that biases future generation toward your preferred argument types.

### 7. Edge Auditing (Menu option 10)

Review low-confidence citation edges, temporal anomalies (e.g., a 2020 paper apparently citing a 2025 paper), and metadata quality issues. Helps identify weak spots in your literature network.

### 8. Quality Report (Menu option 11)

Diagnostic report comparing the graph's scoring against your actual ratings. Shows AUC, Precision@k, NDCG@k, and Spearman correlation — plus how much the graph-based scoring improves over a simple semantic-only baseline.

### 9. Topology Evolution

Track how your research graph grows over time:

```bash
researchbuddy-evolution
# or: python -m researchbuddy.evolution
```

Outputs to `~/.researchbuddy/evolution/`:
- `graph_evolution_metrics.csv` — per-snapshot graph-theory metrics
- `graph_evolution_timeline.png` — timeline plots (growth, density, clustering, modularity)
- `graph_evolution_summary.txt` — textual trend summary

---

## The Intelligence System

This is what makes ResearchBuddy more than a simple keyword search. There are four layers of intelligence that build on each other as you use the tool.

### Layer 1: Cold-Start Mode (0–14 papers)

When you first start, the graph is too small for clustering, citation coupling, or fusion to work reliably. ResearchBuddy knows this and switches to a **simplified scoring mode**:

- **Direct similarity**: how close is this candidate to each of your rated/seed papers?
- **Context vector**: weighted mean of all your papers' embeddings (your "research centroid")
- **Recency**: slight preference for newer papers
- **Expanded search**: more queries sent to S2 and ArXiv to cast a wider net

The menu shows `[COLD-START]` while in this mode. You'll see:
```
Cold-start mode (8/15 papers) -- using simplified scoring.
Add more papers & ratings for full fusion scoring.
```

### Layer 2: Full Multi-Signal Scoring (15+ papers)

Once you have enough papers, the full scoring engine activates with **7 signals**:

| # | Signal | Default Weight | What it captures |
|---|--------|---------------|------------------|
| 1 | Global context similarity | 3.0 | How close to your overall research direction |
| 2 | Niche centroid similarity | 2.0 | How close to specific research niches you care about |
| 3 | Area/domain similarity | 1.0 | Broader thematic relevance |
| 4 | Citation coupling | 2.0 x (1-alpha) | Shares references with papers you like |
| 5 | SNF fused adjacency | 1.5 | Indirect connections via fused network diffusion |
| 6 | Publication quality | 0.5 | Peer-reviewed venues get a gentle boost |
| 7 | Recency | 0.3 | Newer papers get a slight nudge |

### Layer 3: Learned Weights (8+ rated papers)

After you rate enough papers (8+, with at least 2 positive and 2 negative), ResearchBuddy **learns personalized signal weights** from your rating history. It fits a logistic regression on your rated papers to discover which signals matter most to *you*.

For example, if you consistently rate papers highly when they share citations with your existing work (but don't care much about recency), the citation coupling weight increases while recency decreases.

The menu shows `[LEARNED]` when personalized weights are active. Check your current weights in the Stats page:
```
  Scoring: LEARNED weights from your ratings
    context=3.2  niche=2.4  area=0.8  citation=2.9  snf=1.1  pub_qual=0.3  recency=0.2
```

### Layer 4: Temporal Decay

Your research interests shift over time. ResearchBuddy accounts for this with **exponential decay on ratings**:

- Ratings have a half-life of **90 days** — a paper you rated 9 three months ago now contributes as if it were rated ~4.5
- Decay has a floor of **25%** — very old ratings never completely vanish, they just become whispers
- The context vector naturally drifts toward your current interests without you having to re-rate anything

### Adaptive Exploration

Instead of a fixed 25% exploratory ratio, ResearchBuddy adjusts dynamically:

| Situation | Exploration ratio | Why |
|-----------|------------------|-----|
| Small graph (<15 papers) | Up to 40% | You need breadth when starting out |
| Low niche diversity | Up to 40% | Your graph might be too narrow |
| Good coverage across niches | Down to 10% | You've already explored broadly |
| Normal | ~25% | Balanced default |

Exploratory papers are chosen **strategically**, not randomly. ResearchBuddy identifies **coverage gaps** — niches in your graph that exist but have few/no highly-rated papers — and preferentially explores those areas. This means exploration fills blind spots rather than showing you random distant papers.

---

## Architecture Deep Dive

### Adaptive Hierarchy Detection

The number of hierarchy levels is determined automatically — no user input needed.

1. Compute pairwise distances between paper embeddings (PCA-reduced to 64-dim for stability)
2. Build a **Ward-linkage dendrogram** (scipy)
3. Detect "phase transitions" in merge distances using **two complementary signals**:
   - Acceleration peaks (second derivative of merge-distance sequence)
   - Relative jump ratio (threshold at 75th percentile)
4. Each combined peak becomes a hierarchy level cut
5. Fallback: at least one level near k = sqrt(n)
6. Result: 1–8 levels depending on structural breaks in the data

### Similarity Network Fusion (SNF)

The semantic and citation similarity matrices are fused via iterative cross-network diffusion ([Wang et al. 2014](https://doi.org/10.1038/nmeth.2810)):

1. Build KNN-kernel matrices from both networks
2. Iteratively diffuse: `P_semantic = K_semantic @ P_citation @ K_semantic.T` (and vice versa)
3. After convergence: `Fused = alpha * P_semantic + (1-alpha) * P_citation`
4. If citation data is too sparse (< 5% non-zero), falls back to semantic-only

### Citation Reliability

Citation data is fetched from multiple independent sources and cross-validated:

| Source | Strategy | Reliability |
|--------|----------|-------------|
| CrossRef (by DOI) | Reference list from publisher metadata | Highest |
| CrossRef (bibliographic) | Fuzzy title/abstract matching | Medium |
| OpenAlex | DOI or title lookup | Medium-high |
| Semantic Scholar | Paper ID lookup | Medium |

Citation edges are scored by how many independent sources confirm them:
- 3+ sources: full confidence (1.0)
- 2 sources: high confidence (0.8)
- 1 source: moderate confidence (0.55)

### Causal DAG

Edges are oriented by publication year + citation direction, then cycles are broken by removing the weakest-confidence edge in each cycle. Temporal anomalies (a paper appearing to cite a future publication) are flagged automatically and visible in the Edge Audit (option 10).

---

## CLI Reference

```bash
researchbuddy [OPTIONS]
```

### Startup options

| Flag | Description |
|------|-------------|
| `--pdf FOLDER` | Import PDFs from folder on startup |
| `--reset` | Clear all saved data and start fresh |

### Model parameters (override config.py defaults)

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha FLOAT` | 0.6 | Semantic vs citation weight (0 = citation only, 1 = semantic only) |
| `--exploration-ratio FLOAT` | 0.25 | Base fraction of suggestions that are exploratory (adaptive system adjusts this) |
| `--similarity-threshold FLOAT` | 0.45 | Min cosine similarity to draw a semantic edge |
| `--n-recommendations INT` | 10 | Papers shown per search session |
| `--no-plot` | — | Disable PDF graph generation after each session |

### LLM options

| Flag | Description |
|------|-------------|
| `--llm-model NAME` | Ollama model name (default: phi3.5) |
| `--no-llm` | Disable all LLM features entirely |

### Other

| Flag | Description |
|------|-------------|
| `--log-level LEVEL` | DEBUG, INFO, WARNING, ERROR (default: INFO) |

Hierarchy depth is **not** a parameter — it is determined automatically by the data.

---

## Interactive Session

```
========================  ResearchBuddy  ========================
  42 papers  |  12 rated  |  3 levels  |  7 niches  |  3 areas  |
  sem=186 edges  cit=23 edges  [LEARNED]

  [1] Search for new papers
  [2] Show graph statistics
  [3] Add PDF folder
  [4] Fetch citation data (improves fusion quality)
  [5] Resolve Semantic Scholar IDs for seed papers
  [6] Rebuild hierarchy & regenerate all graph PDFs
  [7] Query your research network (Reasoning Mode)
  [8] Creative Mode -- generate & rate argument paragraphs
  [9] LLM status & setup
  [10] Audit graph edges (low-confidence & anomalies)
  [11] Quality & reliability report
  [q] Save & quit
```

Status tags in the menu header:
- `[COLD-START]` — fewer than 15 papers, simplified scoring active
- `[LEARNED]` — personalized signal weights trained from your ratings
- `[NO LLM]` — Ollama is enabled but unreachable; search runs without HyDE/expansion/reranking

---

## Configuration

Persistent defaults live in `researchbuddy/config.py`. All can be overridden via CLI flags for a single session.

### Core parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model (768-dim, 8192-token context) |
| `EMBEDDING_DIM` | 768 | Expected embedding dimension; bump when changing model |
| `FUSION_ALPHA` | 0.6 | Semantic stream weight in SNF (1-alpha = citation weight) |
| `SIMILARITY_THRESHOLD` | 0.45 | Min cosine sim to draw a semantic edge |
| `N_RECOMMENDATIONS` | 10 | Papers shown per search session |
| `EXPLORATION_RATIO` | 0.25 | Base exploration fraction (adaptive system adjusts this) |

### Hierarchy

| Constant | Default | Description |
|----------|---------|-------------|
| `MIN_CLUSTER_SIZE` | 3 | Minimum papers per cluster |
| `MAX_HIERARCHY_LEVELS` | 8 | Upper bound on auto-detected levels |

### Fusion

| Constant | Default | Description |
|----------|---------|-------------|
| `SNF_KNN` | 10 | KNN kernel size for SNF diffusion |
| `SNF_ITER` | 15 | SNF diffusion iterations |

### Intelligence system

| Constant | Default | Description |
|----------|---------|-------------|
| `WEIGHT_LEARNING_MIN_RATINGS` | 8 | Minimum rated papers before weight learning activates |
| `WEIGHT_LEARNING_REGULARIZATION` | 0.1 | L2 regularization for weight learning |
| `RATING_HALF_LIFE_DAYS` | 90 | Rating influence halves every N days |
| `RATING_DECAY_FLOOR` | 0.25 | Minimum decay factor (ratings never fully vanish) |
| `COLD_START_THRESHOLD` | 15 | Papers below this count triggers cold-start mode |

### LLM

| Constant | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `phi3.5` | Ollama model name |
| `LLM_TEMPERATURE` | 0.7 | Temperature for creative tasks |
| `LLM_TEMPERATURE_LOW` | 0.3 | Temperature for structured tasks (JSON output) |
| `HYDE_ENABLED` | `True` | Generate hypothetical abstracts for search |
| `LLM_QUERY_EXPANSION` | `True` | LLM generates alternative search queries |
| `LLM_RERANK_ENABLED` | `True` | LLM reranks search results |

### Search

| Constant | Default | Description |
|----------|---------|-------------|
| `S2_SEARCH_QUERIES` | 6 | Number of Semantic Scholar text-search queries |
| `ARXIV_SEARCH_QUERIES` | 2 | Number of ArXiv queries |
| `CORE_FULL_TEXT` | `True` | Fetch full paper text from CORE |

---

## Testing

The test suite uses mock embeddings and requires no GPU or model downloads:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_graph_model.py -v
```

Tests cover: graph model operations, scoring, embedder utilities, PDF text processing, causal DAG construction, citation network analysis, and the reasoning engine.

---

## File Structure

```
ResearchBuddy/
├── researchbuddy/
│   ├── __init__.py              # version
│   ├── __main__.py              # python -m researchbuddy
│   ├── cli.py                   # interactive CLI (11 menu options)
│   ├── config.py                # all constants and defaults
│   ├── evolution.py             # topology evolution analysis
│   └── core/
│       ├── graph_model.py       # HierarchicalResearchGraph: 4 networks, scoring, learning
│       ├── searcher.py          # Semantic Scholar + ArXiv API search + LLM helpers
│       ├── embedder.py          # sentence-transformers singleton + cosine similarity
│       ├── hierarchy.py         # adaptive Ward-linkage HSWN algorithm
│       ├── fusion.py            # Similarity Network Fusion (Wang et al. 2014)
│       ├── citation_network.py  # bibliographic coupling + co-citation + cross-validation
│       ├── citation_classifier.py # citation edge type classification
│       ├── causal.py            # causal DAG construction + anomaly detection
│       ├── reasoner.py          # LLM-powered research Q&A
│       ├── arguer.py            # creative argument generation + style learning
│       ├── llm.py               # Ollama LLM interface + GPU detection
│       ├── core_fetcher.py      # CORE full-text API + equation stripping
│       ├── pdf_processor.py     # pdfplumber extraction + chunking + metadata
│       ├── state_manager.py     # pickle save/load + PDF import + S2 resolution
│       └── visualizer.py        # 3-PDF matplotlib graph renderer
├── tests/
│   ├── conftest.py              # shared fixtures + mock embedder
│   ├── test_graph_model.py      # graph operations, scoring, weight learning
│   ├── test_embedder.py         # cosine similarity + torchvision guard
│   ├── test_pdf_processor.py    # text cleaning + DOI extraction
│   ├── test_reasoner.py         # keyword scoring + PageRank caching
│   ├── test_citation_network.py # DOI normalization + coupling
│   └── test_causal.py           # edge orientation + cycle breaking
├── pyproject.toml               # build config + dependencies
└── README.md
```

### Persistent data

All user data is stored in `~/.researchbuddy/`:

| Path | Contents |
|------|----------|
| `research_graph.pkl` | Your complete graph state (papers, ratings, networks, learned weights) |
| `history/` | Timestamped graph snapshots (up to 200 kept) |
| `cache/` | LLM helper cache (HyDE, query expansion, reranking results) |
| `cache/core/` | CORE full-text cache (one JSON per paper, fetched once) |
| `network_semantic.pdf` | Semantic network visualization |
| `network_citation.pdf` | Citation network visualization |
| `network_combined.pdf` | Fused network visualization |

---

## Troubleshooting

### "No context vector yet"

You need at least one paper with an embedding. Run option 3 (Add PDF folder) to import papers, or option 1 (Search) and rate at least one result.

### "LLM: UNAVAILABLE"

Ollama isn't running or the model isn't pulled. Fix:
```bash
ollama serve                # start the server
ollama pull phi3.5          # download the model (~2.5 GB)
```
ResearchBuddy works fully without the LLM — you just lose HyDE, query expansion, reranking, and options 7–8.

### Search returns few/no results

- Check your internet connection (search requires API access to S2 and ArXiv)
- If you see "S2 cooldown active", you've hit Semantic Scholar's rate limit. Wait 90 seconds or set a `SEMANTIC_SCHOLAR_API_KEY` environment variable for higher limits
- Try different keywords or a more specific research intent

### "Scoring: COLD-START mode"

This is normal when you have fewer than 15 papers. The system automatically uses simpler scoring that works with sparse data. Keep adding and rating papers — full scoring activates at 15+ papers.

### Existing graph loads with wrong embedding dimension

If you change the embedding model in `config.py`, ResearchBuddy detects the dimension mismatch on startup and re-embeds all papers automatically. This happens once.

### GPU memory issues

The embedding model runs on CPU by default if CUDA isn't available. If you have a GPU but encounter crashes:
```bash
export EMBEDDING_DEVICE=cpu    # force CPU embedding
```

---

## How exploration prevents tunnel vision

ResearchBuddy actively fights the "filter bubble" problem common in recommendation systems:

1. **Adaptive ratio**: 10–40% of suggestions are exploratory depending on your graph's diversity
2. **Gap targeting**: exploratory papers are chosen to fill coverage gaps — niches that exist in your graph but you haven't rated highly in
3. **Clear labeling**: exploratory papers are marked `[EXPLORE]` so you can decide whether to follow up
4. **Diversity-aware**: the system measures how evenly your ratings spread across niches (using entropy) and increases exploration when your attention is too concentrated

This means your research graph grows in breadth *and* depth, surfacing adjacent fields you might otherwise miss.

---

## References

- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*.
- Wang, B., et al. (2014). Similarity network fusion for aggregating data types on a genomic scale. *Nature Methods*.
- Gao, L., et al. (2022). Precise zero-shot dense retrieval without relevance labels (HyDE). *arXiv:2212.10496*.

---

## License

MIT
