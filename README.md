# ResearchBuddy

> **v0.4.0** — Adaptive Hierarchical Small World Network + Causal DAG + LLM-powered Reasoning & Creative Modes

A **graph-based literature search assistant** that learns your research interests from your own PDFs and actively finds new papers for you — like a smart colleague who reads everything and brings you only what matters.

---

## How it works

ResearchBuddy builds **four interconnected networks** from your literature:

```
Your PDF folder
      │
      ▼  (NLP embeddings via sentence-transformers)
┌─────────────────────────────────────────────────────────────────────┐
│             Semantic Network  (HSWN — auto-levelled)               │
│                                                                    │
│  Level 3 (Domain)    [D1]──────────[D2]                            │
│                       │             │                              │
│  Level 2 (Area)    [A1]  [A2]    [A3]  [A4]                        │
│                     │    │        │    │                            │
│  Level 1 (Niche) [N1][N2][N3]  [N4][N5][N6]                       │
│                   │   │   │     │   │   │                          │
│  Level 0 (Paper) [p1][p2][p3] [p4][p5][p6]                        │
│                   ←── dense intra-niche edges ───►                 │
│                   ←── sparse cross-niche shortcuts ─►              │
└─────────────────────────────────────────────────────────────────────┘
      +
┌─────────────────────────────────────────────────────────────────────┐
│             Citation Network  (directed)                           │
│                                                                    │
│  [Paper A] ──cites──► [Paper B] ──cites──► [Paper C]              │
│      │                                          │                  │
│      └── bibliographic coupling ────────────────►┘                 │
│      └── co-citation (both cited by [X]) ────────►                 │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼  (Similarity Network Fusion — Wang et al. 2014)
┌─────────────────────────────────────────────────────────────────────┐
│             Combined Network  (fused, multi-modal)                 │
│  SNF iteratively diffuses information between both networks        │
│  amplifying consistent signals, dampening noise                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼  (Temporal ordering + citation direction)
┌─────────────────────────────────────────────────────────────────────┐
│             Causal DAG  (directed acyclic graph)                   │
│  Edges oriented by publication year + citation direction            │
│  Cycles broken by removing weakest-confidence edge                 │
│  Temporal anomalies flagged automatically                          │
└─────────────────────────────────────────────────────────────────────┘
```

After each session, **three separate PDFs** are generated in `~/.researchbuddy/`:

| File                     | Contents                                       |
|--------------------------|------------------------------------------------|
| `network_semantic.pdf`   | NLP/HSWN overview · paper layer · niche layer · level histogram |
| `network_citation.pdf`   | Directed citation graph · most-cited ranking   |
| `network_combined.pdf`   | Fused graph · edge-type breakdown · stats      |

---

## Features

### Interactive Paper Search (Option 1)

Search for papers using natural language queries. ResearchBuddy combines Semantic Scholar and ArXiv APIs to find candidates, then ranks them using a multi-signal scoring system. Rate each paper 1–10 to teach the graph your preferences.

### Reasoning Mode (Option 7)

Uses a local LLM (via Ollama) to answer research questions grounded in your graph. The reasoner identifies relevant papers using PageRank + keyword matching, then generates answers with citations. Supports follow-up questions in an interactive session.

### Creative Mode (Option 8)

Generates structured argument paragraphs connecting papers in your graph. Each paragraph includes claims, evidence, and citations. Rate the generated arguments to improve future quality.

### Causal DAG & Edge Auditing (Option 10)

View low-confidence citation edges, temporal anomalies (e.g., a 2020 paper citing a 2025 paper), and metadata quality issues. Helps identify weak spots in your literature network.

### Quality & Reliability Report (Option 11)

Generates a diagnostic report covering graph connectivity, metadata completeness, and citation coverage.

### Topology Evolution Tracking

Timestamped graph snapshots stored automatically. Visualize how your research graph grows and changes over time with `researchbuddy-evolution`.

---

## Key algorithms

### Adaptive Hierarchy

The number of hierarchy levels is **determined automatically** by the data — no user input needed.

1. Compute pairwise distances between paper embeddings (PCA-reduced for stability)
2. Build a **Ward-linkage dendrogram** (scipy)
3. Detect "phase transitions" in merge distances using **two complementary signals**:
   - **Acceleration peaks** (second derivative of merge-distance sequence)
   - **Relative jump ratio** (threshold at 75th percentile)
4. Each combined peak becomes a hierarchy level cut
5. Fallback: at least one level near k ≈ √n
6. Result: 1–8 levels depending on structural breaks in the data

### Small-World Structure

Within each detected level:
- **Dense intra-niche edges** (cosine similarity ≥ threshold) connect similar papers
- **Sparse shortcut edges** between niches create small-world navigation paths

### Multi-Signal Prediction

Candidate papers are scored using five complementary signals:

| # | Signal | Weight |
|---|--------|--------|
| 1 | Cosine similarity to **global context vector** (hierarchical: papers + niches + areas) | 3.0 |
| 2 | Similarity to each **niche centroid**, scaled by niche importance | niche\_weight / 10 × 2 |
| 3 | Similarity to each **area / domain centroid**, with level discount (0.8^level) | area\_weight / 20 |
| 4 | **Citation coupling** (bibliographic coupling + co-citation with existing papers) | 2 × (1 − α) |
| 5 | **SNF-fused adjacency** approximation (proximity to top-rated papers in fused space) | 1.5 |

### Similarity Network Fusion (SNF)

The semantic and citation similarity matrices are fused via iterative cross-network diffusion (Wang et al. 2014). If citation data is too sparse (< 5% non-zero), SNF falls back to the semantic matrix directly.

### Causal DAG Construction

Edges are oriented by publication year and citation direction, then cycles are broken by removing the weakest-confidence edge in each cycle. Temporal anomalies (future citations, missing years) are flagged automatically.

---

## Installation

```bash
# From GitHub
pip install git+https://github.com/SwaragThaikkandi/ResearchBuddy.git

# Or clone and install locally
git clone https://github.com/SwaragThaikkandi/ResearchBuddy.git
cd ResearchBuddy
pip install -e .
```

**Requirements:** Python ≥ 3.9. Dependencies are installed automatically:

```
sentence-transformers  networkx  pdfplumber  requests  numpy
scikit-learn  scipy  rich  keybert  matplotlib
```

### Optional: LLM integration (Reasoning & Creative Modes)

Options 7 and 8 require a local LLM via [Ollama](https://ollama.ai/):

```bash
# Install Ollama, then pull a model
ollama pull mistral
# Or any model — configure in option 9 (LLM status & setup)
```

ResearchBuddy works fully without an LLM — options 7–9 simply won't be available.

---

## Quick start

```bash
# First run — import your PDF folder
researchbuddy --pdf /path/to/your/pdf/folder

# Subsequent runs — load saved graph
researchbuddy

# Clear saved state
researchbuddy --reset

# Without installing (run from repo root)
python -m researchbuddy --pdf /path/to/pdf/folder
```

---

## Interactive session

```
══════════════════════════  ResearchBuddy  ══════════════════════════
  42 papers  |  12 rated  |  3 levels  |  7 niches  |  3 areas  |
  sem=186 edges  cit=23 edges

  [1] Search for new papers
  [2] Show graph statistics
  [3] Add PDF folder
  [4] Fetch citation data (improves fusion quality)
  [5] Resolve Semantic Scholar IDs for seed papers
  [6] Rebuild hierarchy & regenerate all graph PDFs
  [7] Query your research network (Reasoning Mode)
  [8] Creative Mode — generate & rate argument paragraphs
  [9] LLM status & setup
  [10] Audit graph edges (low-confidence & anomalies)
  [11] Quality & reliability report
  [q] Save & quit
```

After searching (option 1), you rate each suggested paper 1–10:
- **7–10** → highly relevant, paper added to graph with high weight
- **4–6** → moderate, added with medium weight
- **1–3** → low relevance, used as negative signal

The graph and all three PDF maps are updated after each session.

---

## Topology Evolution

ResearchBuddy stores timestamped graph snapshots in `~/.researchbuddy/history/` on each save. Visualize topology evolution:

```bash
researchbuddy-evolution
# or
python -m researchbuddy.evolution
```

Outputs are written to `~/.researchbuddy/evolution/`:

- `graph_evolution_metrics.csv`  — per-snapshot graph-theory metrics
- `graph_evolution_timeline.png` — timeline plots (growth, density, clustering, modularity, etc.)
- `graph_evolution_summary.txt`  — textual trend summary

Useful flags:

```bash
researchbuddy-evolution --max-snapshots 200
researchbuddy-evolution --no-current
researchbuddy-evolution --out-dir /custom/path
```

---

## CLI parameters

All parameters override `config.py` defaults for the current session:

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf FOLDER` | — | Import PDFs from folder on startup |
| `--reset` | — | Clear saved state and start fresh |
| `--alpha FLOAT` | 0.6 | Semantic vs citation weight (0 = citation only, 1 = semantic only) |
| `--exploration-ratio FLOAT` | 0.25 | Fraction of suggestions that are exploratory |
| `--similarity-threshold FLOAT` | 0.45 | Min cosine similarity to draw a semantic edge |
| `--n-recommendations INT` | 10 | Papers shown per search session |
| `--no-plot` | — | Disable PDF generation after each session |
| `--log-level` | WARNING | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

Hierarchy depth is **not** a CLI parameter — it is determined automatically by the data.

---

## Testing

The test suite uses mock embeddings and requires no GPU or model downloads:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_graph_model.py -v
```

Tests cover: graph model operations, embedder utilities, PDF text processing, causal DAG construction, citation network analysis, and the reasoning engine.

---

## Configuration

Persistent defaults live in `researchbuddy/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MIN_CLUSTER_SIZE` | 3 | Minimum papers per cluster |
| `MAX_HIERARCHY_LEVELS` | 8 | Upper bound on auto-detected levels |
| `FUSION_ALPHA` | 0.6 | Semantic stream weight in SNF |
| `SNF_KNN` | 10 | KNN kernel size for SNF diffusion |
| `SNF_ITER` | 15 | SNF diffusion iterations |
| `SIMILARITY_THRESHOLD` | 0.45 | Min cosine sim for semantic edges |
| `N_RECOMMENDATIONS` | 10 | Papers shown per search |
| `EXPLORATION_RATIO` | 0.25 | Fraction of exploratory suggestions |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |

---

## File structure

```
ResearchBuddy/
├── researchbuddy/
│   ├── __init__.py              # version
│   ├── __main__.py              # python -m researchbuddy
│   ├── cli.py                   # interactive CLI (11 menu options)
│   ├── config.py                # all constants
│   ├── evolution.py             # topology evolution analysis
│   └── core/
│       ├── pdf_processor.py     # pdfplumber → chunks + metadata
│       ├── embedder.py          # sentence-transformers singleton
│       ├── hierarchy.py         # adaptive Ward HSWN algorithm
│       ├── citation_network.py  # bibliographic coupling + co-citation
│       ├── citation_classifier.py # citation edge classification
│       ├── fusion.py            # SNF (Wang et al. 2014)
│       ├── graph_model.py       # HierarchicalResearchGraph (4 networks)
│       ├── causal.py            # causal DAG construction + anomaly detection
│       ├── reasoner.py          # LLM-powered research Q&A
│       ├── arguer.py            # creative argument generation
│       ├── llm.py               # Ollama LLM interface
│       ├── searcher.py          # Semantic Scholar + ArXiv APIs
│       ├── state_manager.py     # pickle save/load + PDF import
│       └── visualizer.py        # 3-PDF matplotlib renderer
├── tests/
│   ├── conftest.py              # shared fixtures + mock embedder
│   ├── test_graph_model.py      # graph operations + scoring
│   ├── test_embedder.py         # cosine similarity + torchvision guard
│   ├── test_pdf_processor.py    # text cleaning + DOI extraction
│   ├── test_reasoner.py         # keyword scoring + PageRank caching
│   ├── test_citation_network.py # DOI normalization + coupling
│   └── test_causal.py           # edge orientation + cycle breaking
├── pyproject.toml
└── README.md
```

**Persistent data** is stored in `~/.researchbuddy/`:
- `research_graph.pkl` — saved graph state
- `history/` — timestamped graph snapshots
- `network_semantic.pdf` — NLP network visualization
- `network_citation.pdf` — citation network visualization
- `network_combined.pdf` — combined network visualization

---

## How exploration avoids research tunnel vision

25% of suggestions (configurable) are **exploratory** — papers with high novelty relative to the entire graph, including cluster centroids at all hierarchy levels. This means:
- You are always exposed to adjacent research areas
- The graph does not collapse into a filter bubble
- Exploratory papers are clearly marked `[EXPLORE]` so you can decide whether to follow up

---

## References

- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*.
- Wang, B., et al. (2014). Similarity network fusion for aggregating data types on a genomic scale. *Nature Methods*.

---

## License

MIT
