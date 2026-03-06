# ResearchBuddy

A **graph-based literature search assistant** that learns your research interests from your own PDFs and actively finds new papers for you — like a smart colleague who reads everything and brings you only what matters.

---

## How it works

```
Your PDF folder
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ResearchGraph                            │
│                                                                 │
│  [Paper A] ──0.82── [Paper B] ──0.61── [Paper C]               │
│      │                   │                  │                   │
│  embedding           embedding           embedding              │
│  weight=8            weight=5            weight=2               │
│      └──────── Context Vector ───────────┘                      │
│                  (weighted mean)                                │
└─────────────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    Semantic Scholar API        ArXiv API
    (text search + rec.)      (text search)
              │                     │
              └──────────┬──────────┘
                         ▼
              Ranked candidates (scored by
              cosine similarity to context)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Relevant (75%)       Exploratory (25%)
         High match score     High novelty score
                         │
                    You rate 1-10
                         │
              Edge weights updated →
              Context vector updated →
              Better results next time
```

**Nodes** = paper embeddings (context vectors).
**Edges** = cosine similarity × combined node weights.
**Context vector** = weighted mean of all node embeddings, where weight = user rating (or 5.0 for unrated seed papers).
**Exploratory suggestions** = 25% of results are papers far from your current graph, so you don't miss adjacent fields.

---

## Features

- **Learn from your PDFs** — drop in your existing library and it builds your initial context
- **Graph memory** — each session builds on the last; the graph saves automatically
- **Dual search** — queries both Semantic Scholar and ArXiv using free public APIs (no key needed)
- **Smart recommendations** — Semantic Scholar's recommendation endpoint uses your highest-rated papers as positive examples
- **Exploration mode** — 25% of suggestions intentionally step outside your current context
- **Feedback loop** — rate papers 1-10; ratings shift edge weights and update future searches
- **Negative learning** — low-rated papers (1-3) become negative examples in recommendations
- **Fully local** — no server, no account, no telemetry; graph state lives in `~/.researchbuddy/`

---

## Installation

### Option A — pip install (recommended)

```bash
pip install researchbuddy
```

### Option B — from source

```bash
git clone https://github.com/SwaragThaikkandi/ResearchBuddy.git
cd ResearchBuddy
pip install -e .
```

> **Python 3.9+** required. Works on Windows, macOS, and Linux.
> First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) — one time only.

---

## Quick start

### 1. First session — import your papers

```bash
researchbuddy --pdf /path/to/your/pdf/folder
```

The tool will:
1. Extract text from every `.pdf` in the folder
2. Embed each paper (sentence-transformers)
3. Build the initial graph
4. Launch the interactive menu

### 2. Subsequent sessions

```bash
researchbuddy
```

The graph is loaded from `~/.researchbuddy/research_graph.pkl`.

### 3. Run without installing

```bash
python -m researchbuddy --pdf /path/to/pdfs
```

---

## Interactive menu

```
──────────────────── ResearchBuddy ────────────────────
  12 papers  |  7 rated  |  34 edges  |  context: ready

  [1] Search for new papers
  [2] Show graph statistics
  [3] Add PDF folder
  [4] Resolve Semantic Scholar IDs (improves recommendations)
  [q] Save & quit
```

### Option 1 — Search

- Optionally enter extra keywords
- Fetches ~30-50 candidates from Semantic Scholar + ArXiv
- Ranks them: 75% by relevance, 25% by novelty (marked with `*`)
- Shows title, authors, year, match %, abstract snippet, URL
- Asks if you want to rate

### Rating papers

```
  [1]     Attention Is All You Need
  [2] [*] Self-supervised learning for protein structure prediction  ← EXPLORE
  [3]     BERT: Pre-training of Deep Bidirectional Transformers

Rating [0]: _
```

| Rating | Meaning                | Effect                          |
|--------|------------------------|---------------------------------|
| 8-10   | Highly relevant        | Large positive weight update    |
| 4-7    | Somewhat relevant      | Moderate positive weight update |
| 1-3    | Not helpful            | Negative example in future S2 recs |
| 0      | Skip                   | No effect                       |

### Option 4 — Resolve S2 IDs

Searches Semantic Scholar for each seed paper by title and stores its S2 paper ID. This enables the **recommendations endpoint** which gives much better results than keyword search alone. Run this once after importing PDFs.

---

## Command-line reference

```bash
researchbuddy                    # load saved graph, start session
researchbuddy --pdf <folder>     # import PDFs, then start session
researchbuddy --reset            # delete saved state, start fresh
python -m researchbuddy [flags]  # same as above without installing
```

---

## Configuration

All tunable parameters are in `researchbuddy/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `SIMILARITY_THRESHOLD` | `0.45` | Min cosine similarity to draw a graph edge |
| `DEFAULT_SEED_WEIGHT` | `5.0` | Implicit weight for unrated seed PDFs |
| `LEARNING_RATE` | `0.15` | How fast ratings shift edge weights |
| `N_RECOMMENDATIONS` | `10` | Papers shown per search session |
| `EXPLORATION_RATIO` | `0.25` | Fraction of suggestions that are exploratory |
| `MIN_NOVELTY_DISTANCE` | `0.30` | Min distance from current graph for "explore" papers |
| `MAX_SEARCH_RESULTS` | `30` | Candidates fetched per search query |

---

## Data & privacy

- Graph state: `~/.researchbuddy/research_graph.pkl`
- Temp downloads: `~/.researchbuddy/temp_papers/`
- All data stays on your machine. No accounts, no cloud sync.
- Search queries are sent to Semantic Scholar and ArXiv APIs (public, free, no auth).

---

## Project structure

```
ResearchBuddy/
├── researchbuddy/
│   ├── __init__.py          # Package version
│   ├── __main__.py          # python -m researchbuddy support
│   ├── cli.py               # Interactive CLI (entry point)
│   ├── config.py            # All tunable constants
│   └── core/
│       ├── pdf_processor.py # PDF → text extraction (pdfplumber)
│       ├── embedder.py      # Sentence-transformers singleton
│       ├── graph_model.py   # ResearchGraph + PaperMeta
│       ├── searcher.py      # Semantic Scholar + ArXiv API
│       └── state_manager.py # Pickle save/load, PDF import
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Paper embeddings (`all-MiniLM-L6-v2`) |
| `networkx` | Graph data structure |
| `pdfplumber` | PDF text extraction |
| `requests` | HTTP calls to S2 and ArXiv APIs |
| `numpy` | Vector math |
| `scikit-learn` | Utility ML functions |
| `rich` | Pretty terminal output |
| `keybert` | Keyword extraction for search queries |

---

## License

MIT — see [LICENSE](LICENSE).
