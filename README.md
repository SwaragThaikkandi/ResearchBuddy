# ResearchBuddy

> **v0.3.0** — Adaptive Hierarchical Small World Network + three-network architecture + comprehensive multi-signal prediction

A **graph-based literature search assistant** that learns your research interests from your own PDFs and actively finds new papers for you — like a smart colleague who reads everything and brings you only what matters.

---

## How it works

ResearchBuddy builds **three interconnected networks** from your literature:

```
Your PDF folder
      │
      ▼  (NLP embeddings via sentence-transformers)
┌─────────────────────────────────────────────────────────────────┐
│             Semantic Network  (HSWN — auto-levelled)            │
│                                                                 │
│  Level 3 (Domain)    [D1]──────────[D2]                         │
│                       │             │                           │
│  Level 2 (Area)    [A1]  [A2]    [A3]  [A4]                     │
│                     │    │        │    │                         │
│  Level 1 (Niche) [N1][N2][N3]  [N4][N5][N6]                     │
│                   │   │   │     │   │   │                       │
│  Level 0 (Paper) [p1][p2][p3] [p4][p5][p6]                      │
│                   ←── dense intra-niche edges ───►              │
│                   ←── sparse cross-niche shortcuts ─►           │
└─────────────────────────────────────────────────────────────────┘
      +
┌─────────────────────────────────────────────────────────────────┐
│             Citation Network  (directed)                        │
│                                                                 │
│  [Paper A] ──cites──► [Paper B] ──cites──► [Paper C]            │
│      │                                          │               │
│      └── bibliographic coupling ──────────────►┘               │
│      └── co-citation (both cited by [X]) ──────►               │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  (Similarity Network Fusion — Wang et al. 2014)
┌─────────────────────────────────────────────────────────────────┐
│             Combined Network  (fused, multi-modal)              │
│  SNF iteratively diffuses information between both networks     │
│  amplifying consistent signals, dampening noise                 │
└─────────────────────────────────────────────────────────────────┘
```

After each session, **three separate PDFs** are generated in `~/.researchbuddy/`:

| File                     | Contents                                       |
|--------------------------|------------------------------------------------|
| `network_semantic.pdf`   | NLP/HSWN overview · paper layer · niche layer · level histogram |
| `network_citation.pdf`   | Directed citation graph · most-cited ranking   |
| `network_combined.pdf`   | Fused graph · edge-type breakdown · stats      |

---

## Key algorithms

### Adaptive Hierarchy (new in v0.3.0)

The number of hierarchy levels is **determined automatically** by the data — no user input needed.

1. Compute pairwise distances between paper embeddings (PCA-reduced for stability)
2. Build a **Ward-linkage dendrogram** (scipy)
3. Detect "phase transitions" in merge distances using **two complementary signals**:
   - **Acceleration peaks** (second derivative of merge-distance sequence, threshold = mean + 0.8 × std)
   - **Relative jump ratio** (Δdistance / distance, 75th-percentile threshold)
4. Each combined peak becomes a hierarchy level cut
5. Fallback: always ensure at least one k ≈ √n level exists
6. Result: 1–8 levels depending on how many genuine structural breaks exist

This means small corpora naturally collapse to 1–2 levels, large diverse corpora expand to 3–5+ levels automatically.

### Small-World Structure

Within each detected level:
- **Dense intra-niche edges** (cosine similarity ≥ threshold) connect similar papers
- **Sparse shortcut edges** between niches (best paper-pair across niche boundary, if similarity ≥ threshold + 0.15) create small-world navigation paths

### Five-Signal Prediction

Candidate papers are scored using five complementary signals:

| # | Signal | Weight |
|---|--------|--------|
| 1 | Cosine similarity to **global context vector** (hierarchical: papers + niches + areas) | 3.0 |
| 2 | Similarity to each **niche centroid**, scaled by niche importance | niche\_weight / 10 × 2 |
| 3 | Similarity to each **area / domain centroid**, with level discount (0.8^level) | area\_weight / 20 |
| 4 | **Citation coupling** (bibliographic coupling + co-citation with existing papers) | 2 × (1 − α) |
| 5 | **SNF-fused adjacency** approximation (proximity to top-rated papers in fused space) | 1.5 |

All signals are combined as a weighted mean and clipped to [0, 1].

### Similarity Network Fusion (SNF)

The semantic and citation similarity matrices are fused via iterative cross-network diffusion:

1. Build KNN kernel (k=10) for each network
2. Alternately diffuse: `P1 ← K1 @ P2 @ K1ᵀ`, `P2 ← K2 @ P1 @ K2ᵀ`
3. Row-normalise after each step
4. Final: `W_fused = α·P1 + (1-α)·P2`, symmetrised and min-max scaled

If citation data is too sparse (< 5% non-zero compared to semantic), SNF falls back to the semantic matrix directly.

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

**Requirements:** Python ≥ 3.9, and the following packages (installed automatically):

```
sentence-transformers  networkx  pdfplumber  requests  numpy
scikit-learn  scipy  rich  keybert  matplotlib
```

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
  [q] Save & quit
```

After searching, you rate each suggested paper 1–10:
- **7–10** → highly relevant, paper added to graph with high weight
- **4–6** → moderate, added with medium weight
- **1–3** → low relevance, used as negative signal

The graph and all three PDF maps are updated after each session.

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

Hierarchy depth is **not** a CLI parameter — it is determined automatically by the data.

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
│   ├── __init__.py           # version
│   ├── __main__.py           # python -m researchbuddy
│   ├── cli.py                # interactive CLI
│   ├── config.py             # all constants
│   └── core/
│       ├── pdf_processor.py  # pdfplumber → chunks + metadata
│       ├── embedder.py       # sentence-transformers singleton
│       ├── hierarchy.py      # adaptive Ward HSWN algorithm
│       ├── citation_network.py # bibliographic coupling + co-citation
│       ├── fusion.py         # SNF (Wang et al. 2014)
│       ├── graph_model.py    # HierarchicalResearchGraph (3 networks)
│       ├── searcher.py       # Semantic Scholar + ArXiv APIs
│       ├── state_manager.py  # pickle save/load + PDF import
│       └── visualizer.py     # 3-PDF matplotlib renderer
├── pyproject.toml
└── README.md
```

**Persistent data** is stored in `~/.researchbuddy/`:
- `research_graph.pkl` — saved graph state
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
