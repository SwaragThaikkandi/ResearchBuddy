# ResearchBuddy

> **NEW** — Privacy-preserving graph sharing & merge ([social-psyche](#social-psyche-privacy-preserving-graph-sharing) capsules + PSI + Gromov–Wasserstein / DeltaCon reliability) · Snowball frontier walk (no more premature saturation) · Open-access full-text harvest (legal OA only, with license provenance) · Citation snowballing with saturation tracking · One-command review pack (BibTeX / RIS / synthesis matrix / themed scaffold / PRISMA flow) · Living-review watch queries · Automatic PRISMA audit trail
>
> **v2.2.0** — Auto-launch Docker services (Neo4j + GROBID) · Readable Neo4j Browser captions + bundled stylesheet · GROBID PDF parsing (figures/tables/equations + local citation network) · Neo4j graph backend · VRAM-aware embedder · Adaptive learned scoring · Temporal decay · Cold-start intelligence · Full-text embeddings via CORE · Hierarchical Small World Network + Causal DAG + LLM-powered Reasoning & Creative Modes

A **graph-based literature search assistant** that learns your research interests from your own PDFs and actively finds new papers for you — like a smart colleague who reads everything and brings you only what matters.

**No cloud accounts, no subscriptions, no data leaves your machine.** Everything runs locally.

## Web UI

```bash
pip install -e ".[ui]"
researchbuddy-ui          # serves http://127.0.0.1:8230 and opens your browser
```

A local web app — no cloud, no account, bound to 127.0.0.1 only. The
frontend is bundled vanilla JS (no npm, no CDN — fully offline). Tabs:

- **Graph** — interactive force-directed map of your research landscape
  (click a node for details; colors: rated / paper / your writing / full text)
- **Library** — **drag-and-drop PDF import** (multi-file, better than the
  CLI's folder prompt): papers get the full GROBID pipeline; or mark uploads
  as *my own draft* to create thought nodes. Server-side folder import too.
  Files persist in `~/.researchbuddy/uploads/`.
- **Discover** — intent + keyword search with **focus mode** (type to pick
  anchor papers from your library), rate results 1–10 inline
- **Snowball** — backward/forward citation expansion with saturation stats,
  a "show top N" control, and **optional PDF attach after rating** (GROBID
  parses it into section embeddings + references — CLI parity, no typing paths)
- **Harvest** — legal open-access full-text autopilot
- **Reasoning** — ask questions about your own collection (relevant papers,
  theme profiles, citation lineages, bridge/frontier papers) and rate the
  answer to tune the graph
- **Review** — one click builds the review **and shows it inline** (themed
  scaffold + PRISMA rendered in the page) plus a **thought map**: theme
  bubbles sized by paper count, green arcs for screened share, amber rings
  marking under-screened gaps, link thickness = theme coupling
- **Evolution** — time-series charts of graph growth, connectivity, and
  structure quality from the snapshot history
- **Watches** — living-review queries
- **Collaborate** — social-psyche: your identity fingerprint, contribution
  ledger (with chain verification), pinned peers, **live secure merge**
  (serve/connect), and signed capsule publication
- **Services** — live status chips for **Neo4j / GROBID / LLM** in the
  header, one-click start/stop via Docker, "Use as backend now" (switches
  the graph onto Neo4j without restarting the UI), Neo4j Browser link

Long operations (search, snowball, harvest, PDF parsing) show a **progress
bar with a plain-language explanation** of the current step underneath —
polled live from the server.

The CLI (`researchbuddy`) remains fully equivalent — the UI is a view over
the same engine and the same local data.

## Why this exists

Knowledge is a common good. This project defends three non-negotiables:

1. **No monopoly on the topology of knowledge.** The *structure* of a field —
   what connects to what, what matters, where the gaps are — must never be
   lockable behind a corporate platform. GPL-3.0-or-later guarantees every
   derivative stays free; the open archive format (menu 20) guarantees even
   *your own data* is never hostage to this tool.
2. **Share verifiable topology, not raw text.** Copyright locks text; it
   cannot lock structure. Capsules carry embeddings + graph structure with
   cryptographic signatures — the next generation inherits verified maps of
   knowledge, not paywalled PDFs.
3. **Federated and survivable.** No server, no registry, no company. Every
   researcher runs their own node; graphs merge peer-to-peer over an
   encrypted channel; published capsules can be mirrored by anyone, anywhere.
   Nothing here can be acquired, shut down, or enclosed.

---

## What does it do? (The 30-second version)

1. You give it a folder of PDFs you've already read
2. It reads them, understands what they're about, and maps how they relate to each other
3. It searches academic databases (OpenAlex, CrossRef, Semantic Scholar, ArXiv) to find new papers you'd probably like — and snowballs through citation networks from your best papers
4. You rate what it finds (1–10), and it gets smarter about your preferences over time
5. It fetches **legally open-access** full texts automatically, so the graph deepens itself (paywalls are skipped, never circumvented; every download records its license)
6. It builds a visual map of your entire research landscape, watches for new papers while you're away, and exports a complete review pack — BibTeX, synthesis matrix, themed review scaffold, PRISMA flow — whenever you're ready to write

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

# With the privacy-preserving graph-merge extra (Gromov–Wasserstein):
pip install -e ".[social]"
```

> **Note:** ResearchBuddy is **not published on PyPI**, so
> `pip install researchbuddy[social]` will fail (or pull an unrelated/old
> package). Install from this repo with `pip install -e ".[social]"`. The
> `[social]` extra (POT) only exists in this source tree.

**Requirements:** Python 3.9 or newer. All dependencies are installed automatically:

```
sentence-transformers  einops  networkx  pdfplumber  requests  numpy
# Optional services (auto-detected): GROBID (Docker) · Neo4j (Docker) · Ollama
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

By default, ResearchBuddy stores the research graph in memory using NetworkX (same as always). For **persistent graph storage**, **faster multi-hop traversals**, and the ability to **explore your graph in Neo4j Browser**, you can optionally use Neo4j as the backend.

**Easiest path (recommended):** just install Docker, then run `researchbuddy`. On startup it will detect that Neo4j isn't running and offer to launch it for you:

```
Neo4j not running. Start it via Docker now? (y/n/never) [y]:
```

Answer `y` and ResearchBuddy will pull the image (one-time, ~600 MB), start the container, wait for it to become healthy, and configure itself to use it. Your answer is remembered — say `never` and you won't be asked again. The same prompt also offers to start GROBID. To skip these prompts on a given run, pass `--no-services`.

**Manual setup (if you don't want Docker auto-launch):**

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

#### Making the Neo4j Browser graph readable

By default Neo4j Browser displays nodes as coloured circles labelled with their internal ID — not very useful. ResearchBuddy bundles a stylesheet that:

- Labels each Paper with `Author Year — short title` (e.g. *Smith 2020 — Causal Graphs*)
- Labels each Cluster with its level + member count (e.g. *L2-cluster · 12 papers*)
- Colour-codes edges by type (citations = teal, semantic = orange, causal = green, …)
- Sizes edges by their weight so important relationships stand out

To use it:

1. From ResearchBuddy's main menu choose **option 12 — "Browse graph in Neo4j"**. ResearchBuddy will open the Browser at a useful starter query and print the path to the bundled `.grass` file.
2. In the Browser, click the cog icon (bottom-left) → **Document Settings**.
3. Drag the printed `.grass` file onto the *Graph Stylesheet* area. Done — every subsequent query renders with the readable styling.

The same menu also prints a few useful starter Cypher queries you can paste in (citation chains, top semantic neighbours of a paper, cluster membership, etc.).


## Visualizing Graph Topology of Research Articles with Neo4j

Your research tool uses **Neo4j** to model articles, authors, citations, and keywords as a graph. This section explains how to explore the **graph topology** – the structure of nodes and edges – and how the static plots below (generated from the Neo4j graph and saved in `plots/`) help you understand research dynamics.

### Example Visualizations

All plots are automatically generated by the tool and saved in the [`plots/`](./plots) folder.

| Figure | What to look for |
|--------|------------------|
| ![Global overview](plots/visualisation%20(15).png) | **Global overview** – nodes (articles) coloured by cluster, edges = citations/co‑authorships. Dense regions = active research themes. |
| ![Labeled graph](plots/visualisation%20(14).png) | **Labeled graph** – shows how individual articles bridge different clusters. |
| ![Hierarchical clusters](plots/visualisation%20(12).png) | **Hierarchical clusters** (L1, L2) – broad domains (L1) and narrower sub‑topics (L2). |
| ![Focused neighbourhood](plots/visualisation%20(11).png) | **Focused neighbourhood** – e.g., how a specific concept connects to others. |
| ![Stable subgraph](plots/visualisation%20(10).png) | **Stable subgraph** – articles that consistently cluster together. |
| ![Metadata overlay](plots/visualisation%20(9).png) | **Overlay of metadata** – journals, authors, or keywords superimposed on the graph. |
| ![Full network](plots/visualisation%20(8).png) | **Full network** – good for identifying peripheral vs. central nodes. |

> **Note:** Any text fragments inside these plots are only **illustrative labels** from the graph – they will reflect your actual article titles, keywords, or cluster names.

### Interpreting the Graph Topology

- **Clusters** → groups of articles that cite each other or share keywords.  
  - *Large, dense clusters* = established research fronts.  
  - *Small, sparse clusters* = emerging or niche topics.

- **Bridges** → articles that connect two different clusters. These often represent interdisciplinary or highly influential work.

- **Degree (number of connections)** →  
  - *High-degree nodes* = hubs (review articles, foundational papers).  
  - *Low-degree nodes* = specialised or recent articles.

- **Centrality** → nodes positioned near the centre of a cluster or between clusters mediate knowledge flow.

### How to Explore the Graph Live (Neo4j)

1. **Start Neo4j** (e.g., `neo4j start` or via Docker).
2. **Open Neo4j Browser** at `http://localhost:7474`.
3. **Run Cypher queries** to inspect the topology.

**Example queries:**

- Find the largest cluster:
  ```cypher
  MATCH (a:Article)
  WITH a.cluster AS cluster, COUNT(*) AS size
  ORDER BY size DESC
  LIMIT 1
  MATCH (a2:Article {cluster: cluster})
  RETURN a2.title, a2.year
See how two clusters are connected:

cypher
MATCH (a:Article)-[:CITES]->(b:Article)
WHERE a.cluster = 'L1-cluster' AND b.cluster = 'L2-cluster'
RETURN a.title, b.title
LIMIT 50
How This Helps Your Research
Discover hidden connections between seemingly unrelated articles.

Identify research gaps – parts of the graph with few edges (under‑cited topics).

Track evolution over time – compare graphs from different years.

Find boundary‑spanning work – articles that combine two different research communities.

All static plots are saved in plots/. For interactive exploration, use the Neo4j Browser or export the graph to Gephi.
  
### Recommended: GROBID for academic-PDF parsing

By default, ResearchBuddy parses PDFs with `pdfplumber`. That works, but `pdfplumber` was not designed for scholarly papers — it struggles with two-column layouts, ligatures, equations, references, and figure/table captions.

[**GROBID**](https://grobid.readthedocs.io/) is a machine-learning library purpose-built for scientific PDFs. When ResearchBuddy detects a running GROBID instance, it switches to it automatically. You get:

- **Section-aware structure** — every section is classified as `introduction`, `related_work`, `methods`, `experiments`, `results`, `discussion`, `conclusion`, `limitations`, etc. Section numbers (`3.2`) are preserved. Each section is also embedded *separately*, so the graph gets a per-section similarity layer (papers' methods <-> other papers' methods, results <-> results, etc.) — visible in Neo4j Browser as `SEC_SIMILARITY` edges with `section` and `weight` properties. The recommender's scoring function gets one **learnable signal per section type** (`methods`, `results`, `discussion`, `introduction`); after enough ratings, weight learning discovers e.g. *"this user really cares about methods similarity, less about results"* and re-weights accordingly.
- **In-text citation contexts** — every `[12]` in the body is mapped to its specific reference, with the *enclosing paragraph and section type* captured. This means we know not just *that* a paper cites Pearl 2009, but that it cites it *in its methods section, in the paragraph about confounder selection*. Used for stronger citation-edge confidence (body-cited > bibliography-only) and richer downstream queries.
- **Reference extraction from the PDF itself** — every cited paper's authors, title, year, and DOI parsed locally; bibliography matched against your library before any external API call, dramatically reducing rate-limit risk.
- **Structured tables** — preserves rows × cells (not just flat cell text), so each table is a real grid you can query.
- **Equations, figure & table captions** parsed separately.
- **Cleaner embeddings** — section-aware chunking instead of dumb word splits.
- **Far better titles** — no more "Microsoft Word - draft.docx" garbage.

Run GROBID as a Docker container. Pick the flavour that matches your hardware:

| Flavour | Image | RAM | Speed | Accuracy | Best for |
|---|---|---|---|---|---|
| **GPU (best)** | `grobid/grobid:0.8.1` + `--gpus all` | ~3 GB VRAM | ~0.3 s / page | Highest (DL) | NVIDIA GPU + nvidia-container-toolkit |
| **Full (CPU)** | `lfoppiano/grobid:0.8.1` | ~1.5 GB | ~3–5 s / page | Highest (DL) | Modern CPU, no GPU |
| **CRF (lite)** | `grobid/grobid:0.8.1-crf` | ~700 MB | ~0.5–1 s / page | Slightly lower | Slow CPU / minimal RAM |

**GPU (recommended if you have an NVIDIA card):**
```bash
# Prereq: install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
docker run -d --gpus all --name grobid -p 8070:8070 \
  --init --ulimit core=0 \
  grobid/grobid:0.8.1
```

The GPU version uses GROBID's DeLFT (BERT-based) models with CUDA acceleration. Header parsing, citation parsing, and segmentation all run roughly **10× faster** than CPU. A 30-page paper goes from ~90 s on CPU → ~10 s on GPU. GROBID coexists fine with ResearchBuddy's embedding model on the same GPU as long as you have ≥ 6 GB VRAM total — the DeLFT models load once and stay resident.

**Full CPU (no GPU available):**
```bash
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1
```

**CRF lite (slow CPU / RAM-constrained):**
```bash
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.1-crf
```

That's it — no further configuration needed. If GROBID is reachable at `http://localhost:8070`, ResearchBuddy will use it.

> **Note:** the auto-launch prompt (option `y` at startup) currently spins up the full CPU image (`lfoppiano/grobid:0.8.1`). For GPU, start the container manually with the command above before running `researchbuddy` — ResearchBuddy will detect the existing container and use it.

**First-PDF cold start:** the first time GROBID handles a PDF after startup, it loads its ML models (~30 seconds on CPU). ResearchBuddy sends a tiny warmup request the moment it detects GROBID is alive, so by the time your real PDFs arrive the models are already loaded.

**Slow CPU?** Bump the per-PDF timeout. ResearchBuddy already retries once at double the budget on a timeout before falling back to pdfplumber:

```bash
# Default is 180s. Doubles to 360s on first timeout. If your PDFs are
# long or your CPU is slow, raise this:
export RESEARCHBUDDY_GROBID_TIMEOUT=300
```

Other knobs:

```bash
# Disable GROBID entirely (force pdfplumber)
export RESEARCHBUDDY_GROBID_ENABLED=false
# Custom URL (e.g. remote GROBID server)
export RESEARCHBUDDY_GROBID_URL=http://my-grobid-host:8070
# Skip the warmup ping (first real PDF will pay the model-load cost)
export RESEARCHBUDDY_GROBID_WARMUP=false
```

If the GROBID service goes down mid-import, ResearchBuddy will silently fall back to `pdfplumber` for the affected papers — your import never fails because of GROBID.

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
- Fetches candidates from OpenAlex, CrossRef, Semantic Scholar (recommendations + text search) and ArXiv
- Optionally generates a HyDE hypothetical abstract via LLM for better matching
- Optionally expands your query with LLM-generated alternatives
- Ranks results using a multi-signal scoring system (see [Intelligence System](#the-intelligence-system))
- Mixes in exploratory papers to prevent filter bubbles

The ranking engine is built on citable retrieval science, not vibes:

- **Personalized PageRank** (Haveliwala 2002) — relevance mass random-walks
  from your rated papers through the fused semantic+citation graph, so a
  paper two hops from your favourites outranks an isolated keyword match.
  Restart-injection is subtracted before scoring, so weight learning is not
  contaminated by label leakage.
- **Reciprocal Rank Fusion** (Cormack et al. 2009) — each search API returns
  a relevance *order*; papers near the top of several independent sources
  carry corroborating evidence, fused calibration-free.
- **MMR diversification** (Carbonell & Goldstein 1998) — the final slate
  maximises relevance *minus* redundancy: ten results that span the topic
  instead of ten near-duplicates of the best hit.
- **Age-normalised impact prior** — log-scaled citations-per-year from
  OpenAlex/CrossRef counts, masked when unknown.
- **Focus mode** — anchor a search on a few papers from *your* library; the
  context vector and the PageRank restart distribution centre on that subset
  (multi-seed exploration with an explicit, inspectable scoring model).

All of these enter the same learned-weight fusion as the existing signals, so
your ratings continue to tune how much each one matters *for you*.

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

### 10. Open-Access Full-Text Harvest (Menu option 15)

The graph feeds itself. For every paper that has only an abstract, the
harvester resolves a **legal** open-access copy — arXiv → Unpaywall →
OpenAlex → Europe PMC, in that order — downloads it, runs it through
GROBID, and upgrades the node with section embeddings and parsed
references. No manual PDF hunting.

**IP-clean by construction:**
- Only services that index author/publisher-sanctioned OA copies are
  queried (Unpaywall explicitly excludes Sci-Hub and similar hosts).
- Paywalled papers are *skipped*, never circumvented.
- Every download gets a `.provenance.json` sidecar recording the provider,
  URL, license (e.g. `cc-by`), version, and retrieval date — you can always
  show where a file came from and under what terms.

PDFs land in `~/.researchbuddy/oa_library/`. Set
`OPENALEX_MAILTO=you@example.com` to unlock Unpaywall (it requires an
email) and faster OpenAlex responses.

### 11. Citation Snowballing (Menu option 16)

The systematic-review expansion loop:
- **Backward**: follow the reference lists of your best-rated papers —
  first from GROBID-parsed references you already own (offline), then via
  OpenAlex `referenced_works`.
- **Forward**: find newer papers citing them (OpenAlex `cites:` filter,
  Semantic Scholar fallback).

Candidates are deduped against your graph, ranked with your learned
scoring weights, and fed into the normal one-at-a-time rating loop.
Each round reports a **saturation ratio** — when another round adds
almost nothing new, your coverage is saturated (the standard stopping
criterion for systematic searches). Citation links are bibliographic
facts; no copyrighted text is involved.

### 12. Review Pack Export (Menu option 17)

One command turns the graph into literature-review deliverables, written
to `~/.researchbuddy/review_packs/pack_<timestamp>/`:

| File | Contents |
|---|---|
| `review.bib` | BibTeX for every included paper |
| `review.ris` | RIS for EndNote / Zotero / Mendeley |
| `synthesis_matrix.csv` | paper × attribute matrix (theme, rating, decision, full-text status...) |
| `review_scaffold.md` | themed review skeleton with citation keys, per-theme synthesis, gap analysis |
| `prisma_flow.md` | PRISMA-2020 flow counts from the audit trail |

Exports contain only bibliographic facts (not copyrightable), your own
ratings/decisions, and *original* synthesis text generated by your local
LLM. Third-party abstracts and full texts are never copied into exports —
papers are referenced via DOI.

### 13. Living Review (Menu option 18)

A review is obsolete the day it's written unless it keeps watching. Save
watch queries; each check asks OpenAlex only for works published since
your last visit, ranks them against your graph, and surfaces the top
hits for rating. Watches persist in `~/.researchbuddy/watches.json`.

### 14. PRISMA Audit Trail (automatic)

Every search, snowball round, screening decision (rating), full-text
retrieval, and watch check is appended to
`~/.researchbuddy/history/prisma_log.jsonl`. The review-pack export folds
this into PRISMA-2020 flow counts — so your literature review is
*reproducible*, not just thorough.

### 15. Graph Capsules — Share & Merge with a Collaborator (Menu option 19)

The interop contract for [**social-psyche**](#social-psyche-privacy-preserving-graph-sharing):
two researchers package their graphs and merge them **without either
revealing what they've been reading**.

- **Export** a privacy-scrubbed `*.rbcapsule`: thought/draft nodes are always
  dropped; DOIs, titles, and ratings are opt-in; embeddings + structure
  always travel.
- **Merge** a collaborator's capsule into yours. New, identified papers are
  imported (then harvestable for OA full text); structurally-matched papers
  are recognised, not duplicated.
- **Reliability report** — every merge reports standard graph-theory
  error/similarity measures so you know how compatible the two landscapes
  are: spectral distance, **DeltaCon** similarity, degree-distribution KS,
  Jaccard overlap, **Gromov–Wasserstein** distortion (label-free alignment),
  and modularity.

The menu offers four modes:

| | Mode |
|---|---|
| [1] | Export my graph → `*.rbcapsule` file |
| [2] | Inspect a capsule file |
| [3] | Merge a capsule **file** into my graph |
| [4] | **Merge LIVE** with a collaborator over a **secure network** — capsules never touch disk (authenticated, forward-secret, end-to-end-encrypted; PSI hides non-shared papers). Needs the optional [social-psyche](#social-psyche-privacy-preserving-graph-sharing) package. |

Gromov–Wasserstein needs the optional extra: install ResearchBuddy editable
with `pip install -e .[social]` (see install note below).

### 16. Open Archive — Zero Lock-In (Menu option 20)

Your working graph lives in a pickle for speed — but a pickle is readable
only by this tool. If your knowledge topology could only be opened by one
program, that program would be the very monopoly this project opposes. So
menu 20 exports **everything** to open, hash-verified formats:

| File | Format | Contents |
|---|---|---|
| `papers.jsonl` | JSON lines | every metadata field of every paper |
| `edges.jsonl` | JSON lines | all graph layers with edge attributes |
| `embeddings.npz` | pickle-free NPZ | paper + per-section embeddings |
| `state.json` | JSON | alpha, learned scoring weights |
| `manifest.json` | JSON | format version + sha256 of every file |

Readable from any language, greppable, diffable, mirrorable.
`import_archive` rebuilds a full working graph from these files alone —
the pickle is a cache, not a prison. Tampering is detected via the manifest
hashes before any import.

Archives are *personal backups* (they contain abstracts and file paths).
To share topology publicly, export a privacy-scrubbed capsule (menu 19) and
sign it with `social-psyche sign` — that's the publishable artifact.

#### Snowball frontier walk (improved)

Citation snowballing (option 16) now **walks outward** instead of
re-harvesting the same shell. Each round remembers the seeds it expanded
(`~/.researchbuddy/snowball_state.json`) and seeds the next round from the
*frontier* — recently-added, underconnected papers — so successive rounds
reach 2-hop, 3-hop neighbourhoods. Saturation is only declared when there's
little new **and** no fresh frontier left; the menu reports how many seeds
remain. (Reset the frontier from the snowball prompt to start over.)

---

## social-psyche: privacy-preserving graph sharing

[**social-psyche**](../social-psyche) is a companion package (separate repo)
that turns the capsule contract above into a two-party collaboration tool:

- **Private Set Intersection (PSI)** — discover which papers you both hold
  without revealing the ones you don't (Diffie–Hellman commutative
  encryption, pure Python).
- **Capsule exchange** — file-based today; authenticated networked handshake
  is the next milestone.
- **Graph-theoretic reliability** — reuses ResearchBuddy's `graph_distance`
  measures to quantify every merge.

ResearchBuddy owns the capsule format + distance measures (the contract);
social-psyche owns the protocol + crypto. See its README for usage.

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
| `EMBEDDING_BATCH_SIZE` | 0 (auto) | 0 = pick from VRAM tier; >0 forces a specific batch size |
| `EMBEDDING_MAX_SEQ_LENGTH` | 0 (auto) | 0 = pick from VRAM tier; >0 caps token length |
| `EMBEDDING_PRECISION` | `auto` | `auto` / `fp16` / `fp32` / `bf16`. Auto picks fp16 on <12 GB VRAM |
| `FUSION_ALPHA` | 0.6 | Semantic stream weight in SNF (1-alpha = citation weight) |
| `SIMILARITY_THRESHOLD` | 0.45 | Min cosine sim to draw a semantic edge |
| `N_RECOMMENDATIONS` | 1 | Papers shown per search session. After each rating you're prompted for the PDF — providing it triggers a full GROBID extraction so the paper enters the corpus as a real graph node (with section embeddings + parsed references), not just an abstract-level stub. Override via `RESEARCHBUDDY_N_RECOMMENDATIONS` or `--n-recommendations` for batch review. |
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
| `OPENALEX_SEARCH_QUERIES` | 4 | Number of OpenAlex queries (250M+ works, no key, topic-aware). Set `OPENALEX_MAILTO=you@example.com` env var for the polite pool / faster routing. |
| `CROSSREF_SEARCH_QUERIES` | 2 | Number of CrossRef queries (publishers' DOI registry). Set `CROSSREF_MAILTO=you@example.com` env var. |
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
│       ├── pdf_processor.py     # PDF extraction (GROBID first, pdfplumber fallback)
│       ├── grobid_client.py     # HTTP client + TEI-XML parser for GROBID
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

### GPU memory (VRAM) tuning

ResearchBuddy auto-detects your GPU's VRAM at first model load and picks safe defaults. You should not need to do anything for most cards. The tiers are:

| VRAM | Tier | Precision | Batch size | Max seq length | Typical card |
|------|------|-----------|------------|----------------|--------------|
| < 5 GB | tiny | fp16 | 2 | 512 | GTX 1650, RTX 3050 4 GB |
| 5–8 GB | small | fp16 | 4 | 1024 | RTX 3050 6 GB, 3060/4060 8 GB |
| 9–15 GB | medium | fp16 | 8 | 2048 | RTX 3060 12 GB, 4070 |
| 16 GB+ | large | fp32 | 8 | unlimited | RTX 4080/4090, A100 |
| (CPU) | cpu | fp32 | 4 | 512 | no GPU |

If you still hit out-of-memory errors during full-text enrichment, force a tighter setting via env vars:

```powershell
# PowerShell — current session
$env:RESEARCHBUDDY_EMBEDDING_PRECISION = "fp16"
$env:RESEARCHBUDDY_EMBEDDING_BATCH_SIZE = "1"
$env:RESEARCHBUDDY_EMBEDDING_MAX_SEQ_LENGTH = "512"
```

```bash
# bash / zsh
export RESEARCHBUDDY_EMBEDDING_PRECISION=fp16
export RESEARCHBUDDY_EMBEDDING_BATCH_SIZE=1
export RESEARCHBUDDY_EMBEDDING_MAX_SEQ_LENGTH=512
```

To force CPU embeddings (slow but never OOMs):
```powershell
$env:RESEARCHBUDDY_EMBEDDING_DEVICE = "cpu"
```

If a CUDA OOM still occurs during encoding, the embedder automatically retries at smaller batch sizes (down to 1) and finally falls back to CPU (set `RESEARCHBUDDY_EMBEDDING_CPU_FALLBACK_ON_OOM=false` to disable).

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

**GNU General Public License v3.0 or later (GPL-3.0-or-later).** Strictly
open source / copyleft: you may use, study, share, and modify ResearchBuddy,
but any distributed derivative must also be released under the GPL with its
source. See [LICENSE](LICENSE) for the full text.

Copyright (C) 2026 Swarag Thaikkandi and ResearchBuddy contributors.

> This program is free software: you can redistribute it and/or modify it
> under the terms of the GNU General Public License as published by the Free
> Software Foundation, either version 3 of the License, or (at your option)
> any later version. This program is distributed WITHOUT ANY WARRANTY; see
> the GNU General Public License for more details.
