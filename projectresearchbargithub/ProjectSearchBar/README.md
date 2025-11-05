ProjectSearchBar

Overview
- Local academic paper search with math-aware tokenization (text + LaTeX) and cosine similarity.
- Backend in Python (vectorization, SQLite index, API server).
- UI in TypeScript. If `pywebview` is installed the app opens in a desktop window; otherwise it opens in your default browser.
  New: the main search UI embeds your `api_gui_modular` chat renderer in a bottom‑left dock (KaTeX-enabled).

What's New (Oct 2025 — Tokenization + Debug)
- Shared tokenizer module: server now uses a unified tokenizer (`ProjectSearchBar/tokenize.py`) for text + LaTeX queries with canonicalization and symbol expansion.
- Unicode/ASCII math expansion: converts pasted symbols and ASCII operators to LaTeX tokens used by the index.
  - Examples: `-> ⇒ \\to`, `=> ⇒ \\implies`, `<-> ⇒ \\leftrightarrow`, `<= ⇒ \\leq`, `>= ⇒ \\geq`, `!= ⇒ \\neq`, `<< ⇒ \\ll`, `>> ⇒ \\gg`.
  - Unicode: `≈ ⇒ \\approx`, `≪ ⇒ \\ll`, `≫ ⇒ \\gg`, `∑ ⇒ \\sum`, `‖/∥ ⇒ (\\Vert, \\|)`, greek letters, and blackboard bold letters.
- Math variant canonicalization: maps `\\varepsilon→\\epsilon`, `\\varphi→\\phi`, `\\vartheta→\\theta`, `\\varsigma→\\sigma` to improve overlap.
- New debug endpoint: `GET /api/debug/tokenize?q=...` returns `{ text_tokens, math_segments, math_tokens, final }` to inspect how your input is tokenized.

Quick Check (Debug Tokenization)
- Run `python3 launch.py` and open: `http://127.0.0.1:8360/api/debug/tokenize?q=\\left\\|g_\\lambda-...`
- Expect to see tokens like `\\lambda, \\to, \\leq, \\ll, \\sum, \\int, \\|` present in `tokens.final` for math-heavy snippets.

UI Skins (classic vs UI2)
- Two UIs are available:
  - `ui1` (classic): `ProjectSearchBar/ui/public`
  - `ui2` (redesigned): `ProjectSearchBar/ui2/public` (default)
- Switch via env var when launching:
  - Use UI2 (default): `PROJECTSEARCHBAR_UI=ui2 python3 launch.py`
  - Use classic UI1: `PROJECTSEARCHBAR_UI=ui1 python3 launch.py`
- The server also falls back to the other UI for missing static files (e.g., shared assets), so you can evolve UI2 without duplicating everything.

Metadata-Enriched Results (Titles/Authors/Subjects/Abstract)
- New endpoint: `GET /api/paper/meta?ids=<comma-separated arXiv ids>`
  - Returns `{ ok: true, meta: { "1012.1190": { title, authors[], subjects[], abstract, doi, source } } }`.
  - Caches responses to `data/meta/<id>.json` and reuses them on subsequent runs (basic TTL by manual refresh).
  - Tries the arXiv Atom API (batch), and gracefully falls back to an offline heuristic from local vectors if network is restricted.
- Search UI behavior:
  - After `/api/search`, the client batches all unique paper IDs and calls `/api/paper/meta` once.
  - Result cards update to show: Title (link), Authors (condensed), Subject chips, Abstract excerpt, + the cosine-score meter.
  - Degrades gracefully to the original chunk preview if metadata isn’t available.
  - Note: The result header always links to the arXiv abstract by ID even before metadata arrives. A small Copy button copies the shown excerpt.

Terminal Toggle + Centered Mode (UI2)
- Default: centered search and results with a longer searchbar; terminal hidden.
- Toggle: Click `Terminal` (to the left of the DB banner in the header) to switch to a split layout with a taller diagnostics terminal.
- The terminal logs DB/banner updates and auto-scrolls; preference persists across refresh.
 
Paper Chat (UI2) + Enlarge
- Results show an `AI` button that opens a right-side chat for that specific paper (ephemeral session).
- The chat panel embeds the modular chat UI and loads paper context automatically (LaTeX preferred, vectors fallback).
- Send messages via the input bar or Ctrl/Cmd+Enter; responses render as styled bubbles with math/code support.
- Enlarge: Use the `Enlarge` button in the chat header to toggle to a larger preset size; click `Restore` to return to default.
  - The layout responds: the left column (search/results) adjusts accordingly.
  - Sizing uses CSS vars (`--chat-width`, `--chat-height`) with default 480px/65vh and large 720px/85vh.
 
Known Notes (Chat Size)
- Drag-to-resize was removed in favor of a simpler Enlarge/Restore button.

Settings (UI2)
- Sections: AI Settings (model + API key with Test + Save), Search Settings (filters, Scoring, BM25 params, SVD re-rank), Index Build (batching, mode, shards).
- Save is at the top-right of the modal (left of Close). Search settings persist to `data/search_settings.json` (survive restarts, safe for pywebview).

HTTP API Summary
- Search and metadata
  - `POST /api/search` → `{ ok, results: [{ paperId, kind, text, score, cosine, bm25 }], scanned }`
  - `GET /api/paper/meta?ids=ID1,ID2,...` → `{ ok, meta: { ID: { title, authors[], subjects[], abstract, doi, source } } }`
  - `GET /api/search/settings` → `{ ok, settings }`
  - `POST /api/search/settings` with `{ settings }` → `{ ok }`
  - `GET /api/debug/tokenize?q=...` → `{ ok, q, tokens: { text_tokens, math_segments, math_tokens, final } }`
- Index build and status
  - `POST /api/index/build` → `{ ok, started, build_id, papers_total? }`
  - `GET /api/index/status` → `{ state, phase?, papers_done?, papers_total?, chunks?, postings?, ... }`
  - `GET /api/diagnose` → `{ ok, db, stats: { tokens, papers, chunks, postings } }`
- LLM chat (ephemeral sessions)
  - `POST /api/chat/start` → `{ ok, session_id }`
  - `POST /api/chat/add_paper` with `{ session_id, paper_id, mode?: 'auto'|'latex'|'vectors' }` → `{ ok, paper_id, mode, chars, preview? }`
  - `POST /api/chat/message` with `{ session_id, text }` → `{ ok, answer, model, used? }`
  - `POST /api/chat/clear` with `{ session_id }` → `{ ok }`
- LLM settings
  - `GET /api/llm/settings` → `{ ok, model, has_key }`
  - `POST /api/llm/settings` with `{ provider:'openai', model, api_key? }` → `{ ok }`
  - `POST /api/llm/test` → `{ ok, model }` or `{ ok:false, error/detail }`

Current Progress (Oct 2025)
- AI Chat dock with OpenAI: Save API key + select model (gpt‑5 / gpt‑5‑mini / gpt‑4o / gpt‑4o‑mini) in Settings. Chat now uses session‑based paper attachments; assistant is instructed to wrap math in `$...$`/`$$...$$`.
- Session‑based paper reading: Add/drag results to attach papers for this session only (no persistence). New `/api/chat/*` endpoints manage ephemeral attachments and messages.
- Search options moved to Settings: Kind filter, per‑paper cap, max results (defaults effectively unlimited in legacy mode).
- Indexing stability: periodic commits, smaller posting batches, bounded token cache, incremental DF flush to prevent mid‑build freezes.
- Performance: composite index on postings, read PRAGMAs, and parallel scoring with `PROJECTSEARCHBAR_WORKERS` (auto by default) for multi‑core speedup.
 

Current Progress (Oct 2025)
- LLM Chat dock: Embedded chat UI with KaTeX typesetting; uses `/api/ask`.
- OpenAI integration: Settings include OpenAI API key + model selector (GPT‑5 / GPT‑5 Mini / GPT‑4o / GPT‑4o Mini). Key persists in `data/llm_settings.json`.
- Search options moved to Settings: Kind filter, per‑paper cap, and max results. Defaults are effectively unlimited unless you set caps.
- Indexing stability: Reduced posting batch sizes, added periodic commits (even in non‑low‑mem) and incremental DF flush to avoid freezes around ~30k papers.

How It Works
- Papers: We use your downloaded arXiv LaTeX archives at `/home/Brandon/arxiv_tex_selected`.
- Vectorization (tools/vectorize.py):
  - Extracts .tex files from each archive, strips comments, splits into paragraphs and math blocks.
  - Tokenizes text and LaTeX math, adds gentle canonicalization and math-aware hints (e.g., `->` → `\\to`).
  - Produces `chunks.jsonl` per paper in `data/vectors/<arxiv_id>/`.
- Indexing (tools/index_merge.py):
  - Reads all `chunks.jsonl` and builds a SQLite inverted index `data/index.sqlite` with tables `token`, `paper`, `chunk`, and `posting`.
  - Precomputes per-chunk norms for TF–IDF cosine search.
- Search (server_app/server.py):
  - Tokenizes your query (text + inline LaTeX), builds a TF–IDF vector, and searches via the inverted index.
  - Computes two scores for every result:
    - TF–IDF cosine (0–1), displayed as 0–100%.
    - BM25 (unbounded). UI shows a relative 0–100% bar and the raw BM25 in parentheses.
  - “Scoring” in Settings selects which metric ranks results; both metrics are always displayed on each result.

Math Overview (Tokenization → Index → Scoring)
- LaTeX‑aware tokenization
  - Inline math found via `$...$`, `$$...$$`, `\(...\)`, `\[...\]`. For math blocks in environments, wrappers like `\begin{equation}` / `\end{equation}` are stripped.
  - Text tokens: lowercased alphanumerics/underscores (broad Latin/Greek/Cyrillic), split on whitespace.
  - Math tokens: LaTeX commands (`\\alpha`, `\\sum`, `\\int`, `\\mathbb`), symbols (`= + - ^ _ { } ( )`), numbers, and identifiers.
  - Gentle normalization on indexed chunks: singularize words, canonicalize common LaTeX variants (e.g., `\\varepsilon`→`\\epsilon`), and add word→LaTeX hints (e.g., “integral” adds `\\int`) so text queries can match math.

- Vector weights (TF–IDF)
  - Term frequency per chunk: `tf' = 1 + ln(tf)`
  - Inverse document frequency: `idf = ln((1+N)/(1+df)) + 1`
  - Chunk vector components: `v[t] = tf' * idf`; per‑chunk norm: `||v|| = sqrt(Σ (tf' * idf)^2)` (precomputed and stored in DB)
  - Query vector: same `tf'` and `idf`, L2‑normalized to unit length.

- Candidate generation
  - From the query tokens, drop very high‑df terms and keep up to K rare (high‑idf) tokens.
  - Fetch chunks that contain any chosen tokens; this bounds scan cost even for long queries.

- Cosine similarity (vector space model)
  - On each candidate: `dot = Σ_over_overlap q[t] * (tf' * idf)` then `cosine = dot / ||chunk||`, range ∈ [0,1].

- BM25 (probabilistic ranking)
  - For each query term t in chunk d: `idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))` (strictly‑positive variant)
  - TF saturation & length normalization: `w = ((k1 + 1) * tf) / (tf + k1 * (1 - b + b * dl/avgdl))`
    - `dl = Σ tf` (chunk length), `avgdl` is the average chunk length across the index
  - `BM25(d) = Σ idf(t) * w`
  - UI shows: a relative bar (0–100% of the top BM25 for the current query) and the raw BM25 value in parentheses.

- Two‑pass re‑rank (BM25 → Cosine)
  - When enabled, sort by BM25, take top‑N and re‑order those by cosine; append the rest unchanged.
  - This uses BM25’s lexical recall to find a relevant neighborhood, then cosine’s normalized geometry to refine the head.

- Optional SVD/LSA (TruncatedSVD)
  - Dimensionality reduction of TF–IDF into k‑dimensional space (randomized solver); compute cosine in latent space for semantic re‑ranking/blending.

Score display & ranking
- Cosine bar: shows 0–100% of a true [0,1] cosine.
- BM25 bar: relative to the max BM25 among results for the current query; raw BM25 is shown next to the bar.
- “Scoring” in Settings only controls the ordering (TF–IDF vs BM25). Both metrics are always displayed on each result.
- Two‑pass option (Settings) enables BM25→cosine re‑rank of the top‑N candidates.

Indexing
- You can start a build from the UI (Settings → Start Index Build) or use the CLI.
- Suggested CLI flow:
  1) Vectorize (parallel):
     `python3 tools/batch_vectorize.py --src /home/Brandon/arxiv_tex_selected --out ./data/vectors --workers auto`
  2) Build index (atomic swap):
     `python3 tools/build_index.py --reset`
     (equivalent to: `python3 tools/index_merge.py --scan ./data/vectors --db ./data/index.sqlite --reset`)
- When building directly to `data/index.sqlite`, ensure the app is not reading the DB; or write to a temp file and swap with `os.replace`.

Sharded build (UI)
- In Settings → Build mode, choose "Sharded" and set Shards/Shard workers. This builds multiple shard DBs in parallel and merges them with an atomic swap.
- Recommended for very large datasets; reduces peak memory and lock contention.

Troubleshooting
- Banner stuck? Check status: `GET /api/index/status`.
- DB stats: `GET /api/diagnose` shows token/paper/chunk counts.
- Searching returns "scanned 0" if no candidate tokens match or if the DB is empty. Confirm nonzero counts in diagnose; if zero, click Build Index or rebuild manually.
- If you ever see "database is locked", re-run Build Index; the app now uses a temp build and atomic swap to prevent this.
- If it stays on “Indexing (writing DB)…” for a long time, it may be processing many papers; watch the `(X/Y)` counter increase. You can limit via CLI with `--limit-papers` for a quick test.
- Machine freezes during Build Index? Try low‑memory mode:
  - CLI: `PROJECTSEARCHBAR_LOW_MEM=1 PROJECTSEARCHBAR_FAST_BUILD=0 PROJECTSEARCHBAR_POST_BATCH=100000 python3 tools/build_index.py --reset`
  - From app: launch with env: `PROJECTSEARCHBAR_LOW_MEM=1 PROJECTSEARCHBAR_FAST_BUILD=0 PROJECTSEARCHBAR_POST_BATCH=100000 python3 launch.py`, then press Build Index.
  - Or use sharded build to limit peak memory: `python3 -m ProjectSearchBar.tools.sharded_index --scan ./data/vectors --out-db ./data/index.sqlite --shards 4 --workers 2 --reset`.
  - New: the indexer now flushes document-frequency (DF) increments periodically to keep memory and transaction size bounded on large builds. You can tune with `PROJECTSEARCHBAR_FLUSH_DF` (default 250000). Also added periodic commits (default every ~50 papers) and smaller posting batches to prevent stalls.

Key Paths
- `data/papers`: optional local paper cache (we default to your existing `/home/Brandon/arxiv_tex_selected`).
- `data/vectors`: per-paper vectorization outputs (chunks + tokens) — stays inside this project.
- `data/index.sqlite`: global inverted index for fast cosine search over chunks.

Quick Start
1) Vectorize papers (reads from `/home/Brandon/arxiv_tex_selected` by default):
   python3 tools/batch_vectorize.py --src /home/Brandon/arxiv_tex_selected --out ./data/vectors --workers auto

2) Build index (writes to `./data/index.sqlite`):
   python3 tools/build_index.py --reset

Sharded build (faster on multi-core)
- Build N shard DBs in parallel, then merge:
  python3 -m ProjectSearchBar.tools.sharded_index \
    --scan /home/Brandon/ProjectSearchBar/data/vectors \
    --out-db /home/Brandon/ProjectSearchBar/data/index.sqlite \
    --shards 4 --workers 4 --reset
- Tip: put `--out-db` on SSD for speed; read vectors from HDD is fine.

3) Launch desktop app (starts API + window):
   python3 launch.py

Ranking and Scores
- Cosine (TF–IDF): L2‑normalized TF–IDF vectors; dot product in [0,1]. The UI shows a 0–100% bar.
- BM25: Probabilistic ranking (idf for rarity; k1 for term‑frequency saturation; b for length normalization). The UI shows:
  - A relative bar: 0–100% of the top BM25 within the current results.
  - The raw BM25 number (shown in parentheses). Values aren’t comparable across different queries.
- Ranking selection:
  - “Scoring” in Settings controls the ordering only (TF–IDF or BM25). Both metrics are still displayed per result.
  - Planned extensions: “Both (blend)” ranking (α·cosine + (1−α)·BM25_norm), two‑pass (BM25 candidates → cosine re‑rank), and tie‑break rules.

Backup & Restore (Code Only)
- Create a timestamped backup archive of the code (exclude data/ to avoid large vectors/DBs):
  tar -czf ProjectSearchBar_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='ProjectSearchBar/data' \
    --exclude='ProjectSearchBar/__pycache__' \
    --exclude='ProjectSearchBar/*/__pycache__' \
    --exclude='ProjectSearchBar/.git' \
    ProjectSearchBar
- Restore: extract the archive over a new working directory and copy your existing `data/` back in if desired.

Chat UI
- The main search page shows an AI Chat dock (bottom‑left) that embeds your modular chat renderer (KaTeX enabled).
- Attach papers to the current chat session by clicking “Add” on any search result or dragging it into the dock. Attachments are in‑memory only and reset on reload/restart.
- Type a question and press “Send” to call `/api/chat/message`; the server builds context from attached papers (LaTeX preferred, vectors fallback) under a token budget, and replies with citations.
- Debugging: click “Debug” in the dock header to toggle full raw‑context display after answers; or click “View” on a paper chip to dump that paper’s raw LaTeX/vector text into the chat as code blocks.
- Configure key and model in Settings → AI. Supported IDs: `gpt-5`, `gpt-5-mini`, `gpt-4o`, `gpt-4o-mini`.
 

Search API (SVD re‑rank optional)
- Endpoint: `POST /api/search`
- Body:
  {
    "query": "...",
    "top_k": 5000,
    "per_paper_k": 1,
    "kind": "both",
    "max_candidates": 5000,
    "svd": { "enabled": true, "topN": 2000, "alpha": 0.5 } // optional
  }
- If SVD assets are not present, the server ignores the `svd` object and returns TF–IDF results.

Paper Reading UX (Session‑based)
- Goal: Attach specific papers and ask questions; get concise, cited answers.
- Add papers: Click “Add” next to a result or drag it into the dock. Each attached paper is shown as a chip (click “×” to remove). Attachments live only for this session (no persistence).
- Full LaTeX preferred: The server extracts and caches LaTeX from `/home/Brandon/arxiv_tex_selected`; if unavailable, it falls back to vectors text (`chunks.jsonl`).
- Token budget: The chat limits total context to a configurable budget (default ~6000 tokens). If attachments exceed the budget, they’re truncated.
- Answer style: Per‑paper citations like `[paperId]`; optional context previews are included after the answer.

Using the Reader
1) Run a search to show candidate papers.
2) Drag & drop the papers you care about into the chat dock (or click Add).
3) Ask your question in the chat (e.g., “summarize the main result”, “how can I apply X?”). The AI answers with citations, reading only the attached papers.

Chat Sessions API
- Ephemeral by design — state is in memory and resets on server restart.
- `POST /api/chat/start` → `{ ok, session_id }`
- `GET /api/chat/state?session_id=...` → `{ ok, session_id, papers: [{ id, mode }] }`
- `GET /api/chat/context?session_id=...&limit_chars?=N` → `{ ok, session_id, papers: [{ paper, mode, chars, truncated, full }] }`
- `POST /api/chat/add_paper` with `{ session_id, paper_id, mode?: 'latex'|'vectors'|'auto' }` → `{ ok, paper_id, mode, chars, preview }`
- `POST /api/chat/remove_paper` with `{ session_id, paper_id }` → `{ ok }`
- `POST /api/chat/clear` with `{ session_id }` → `{ ok }`
- `POST /api/chat/message` with `{ session_id, text, token_budget?: number }` → `{ ok, answer, model, used?: { papers: string[], previews: [...], chars_used_total, chars_total_attached } }`

Notes:
- LaTeX is preferred (cached in `data/ai_cache`); we fall back to vectors text from the index if LaTeX is missing.
- `/api/ask` remains for single‑shot, non‑reading questions; the dock uses `/api/chat/message` for paper‑aware chat.

Notes
- If you update server or tools code, restart the app so the server picks up changes.
- If the chat reports “Offline mode…”, click Settings → Test LLM to diagnose connectivity/auth.
- The chat uses the saved OpenAI API key (Settings → Save Settings). If the key is missing/invalid or the network is blocked, the model falls back to an offline summary, but you can still inspect the full raw context via Debug or the per‑paper View buttons.
- Ensure the selected model is supported by your OpenAI account; otherwise the API will return an error.

Legacy Search Behavior (long prompts)
- The launcher now defaults to the earlier, long‑prompt‑friendly behavior:
  - `PROJECTSEARCHBAR_NO_FILTER=1` (no stopword/DF filtering)
  - `PROJECTSEARCHBAR_NO_TIMEOUT=1` (no SQLite progress timeout)
  - Higher default caps: `MAX_RESULTS=20000`, `MAX_CANDIDATES=100000`, `SCAN_BUDGET=5000000`
- You can override these in your shell before `python3 launch.py` to tighten/relax as you like.

Search Caps & Paging
- UI Settings → Search: Kind, Per‑paper cap, Max results. If not set, launcher defaults may apply.
- Server clamps extreme values using env vars (raise if needed):
  - `PROJECTSEARCHBAR_MAX_RESULTS` (default 20000)
  - `PROJECTSEARCHBAR_MAX_CANDIDATES` (default 100000)
  - `PROJECTSEARCHBAR_SCAN_BUDGET` (combined DF budget for chosen tokens)
- OFFSET support is available internally for future paging (“Load more”).

Settings modal tips
- The Settings dialog now scrolls if content exceeds the viewport. On smaller displays it opens near the top; scroll inside the panel to see all options.

From the app UI
- Click “Settings” in the header, then “Start Index Build” to run an index build in the background.
- While building you’ll see progress in the banner (papers, chunks, postings, and norms).

After Updating Code
- If you update server or tools code, restart the app: close the window and re-run `python3 launch.py` so the server picks up changes.

Port already in use
- If you see `OSError: [Errno 98] Address already in use`, a server is already running.
- The launcher detects an existing server and opens a window to it; for a fresh server, kill the old process (e.g., `pkill -f server_app.server` or `kill $(lsof -ti tcp:8360)`).

Notes
- The downloader in `tools/download.py` can be used to fetch additional math LaTeX tarballs.
- Build Index is available inside the Settings modal (the main header intentionally does not include a Build button).
- If `pywebview` is not installed, the app will open in your default browser.
- Parallel vectorization: The vectorizer uses a process pool (`--workers auto`) to split work across CPU cores; progress shows in the banner when launched from the UI.
 
Fast indexing notes
- During a build the indexer writes to a temp DB and then atomically swaps in the final DB.
- For speed it defers index creation until the end and computes chunk norms in a single pass.
- It also uses speed‑oriented SQLite PRAGMAs on the temp DB (journal OFF, synchronous OFF, large cache); the final DB is swapped in and served in WAL mode.
 - New: Set `PROJECTSEARCHBAR_LOW_MEM=1` to stream norm computation via a temporary table, reducing Python heap usage on very large corpora.
 - New: DF increments are flushed periodically (`PROJECTSEARCHBAR_FLUSH_DF`) to avoid stalls around large token counts.
  - New: Token cache bounding (`PROJECTSEARCHBAR_TOKEN_CACHE_MAX`) to control RAM during ingest.
  - New: Periodic commits in ingest (default every ~20–50 papers) and reduced posting batches to prevent long stalls.

Search performance hints
- Parallel scoring: set `PROJECTSEARCHBAR_WORKERS=auto` (default) or a number to use multiple threads and connections.
- Read PRAGMAs: the server applies larger cache and mmap for faster reads.
- Composite index: `ix_posting_chunk_token` accelerates candidate scoring; created at build end.

Future Upgrades (Math Foundations)
- Ranking improvements:
  - BM25: probabilistic term weighting with TF saturation and length normalization; add alongside TF–IDF and A/B test.
  - LSI (Truncated SVD): project chunks/queries into a k‑dimensional space (k≈128–256) to capture semantics; use as a re‑ranker over TF–IDF candidates.
  - Lightweight learned re‑ranker: ridge/logistic regression on features (BM25/TF–IDF score, math share, token rarity, coverage, recency) trained from a small labeled set.
- Representation/features:
  - Fielded indexing (equations vs paragraphs) with learnable per‑field weights.
  - Better LaTeX canonicalization (macro/variant normalization; operator role features).
- Probability & inference:
  - Prior smoothing for rare tokens (e.g., add‑α on df) to stabilize idf.
  - Query likelihood (Dirichlet smoothing) as a generative secondary score; blend with cosine/BM25.
  - EM for latent “mathness”/field weights to reweight token classes in a principled way.
  - Variational inference (ELBO) for learning token‑class priors/weights from clicks.
- Graphs & clustering:
  - Paper similarity graph + spectral clustering/PageRank for exploration and diversity (MMR across clusters).
- Time‑aware retrieval:
  - Recency decay on scores; learn decay parameter by maximizing likelihood over past selections.
- Optimization & constraints:
  - Constrained scanning: treat `SCAN_BUDGET` as a constraint; choose informative tokens to maximize utility under a DF budget (Lagrangian tuning).
  - Regularization: ridge/elastic‑net for re‑ranker to avoid overfitting small labels.
- Evaluation & tooling:
  - Offline eval harness with P@k/NDCG and bootstrap confidence intervals to validate changes.
  - Feature/score logging for top‑k to explain “why this result won”.
- Approximate NN (advanced):
  - ANN over LSI vectors (e.g., HNSW/IVF‑PQ) to accelerate candidate retrieval at scale.

SVD Integration (Optional)
- Why: Latent semantic indexing captures relationships not visible to surface tokens.
- Build pipeline (tools/svd_build.py to be added):
  1. Stream a sparse term–chunk TF–IDF matrix from `data/index.sqlite`.
  2. Compute randomized TruncatedSVD (k≈200–400); save chunk embeddings under `data/svd/`.
  3. Add a server toggle to use SVD cosine (or hybrid: TF–IDF candidates + SVD re-rank).
- Query flow: tokenize → TF–IDF → project to SVD space → cosine to chunk embeddings → blend or re-rank.

How to enable SVD re‑rank
- Build SVD: `python3 -m ProjectSearchBar.tools.svd_build --k 128 --fit-sample 100000 --batch-size 2000`
  - Writes artifacts to `data/svd/` (components, idf, token_map) and `data/svd/svd.sqlite` (chunk embeddings).
  - Requires: `pip install scikit-learn scipy numpy`.
- In the UI → Settings:
  - Check “Re‑rank with SVD (LSI)”.
  - Set “SVD Top N” (e.g., 2000) and “SVD Blend α” (0..1; 0.5 = blend TF–IDF 50% + SVD 50%).
- Server loads SVD assets lazily if present. If assets are missing or numpy is unavailable, it gracefully skips SVD.

Optional Dependencies (for SVD)
- `scikit-learn` (TruncatedSVD), `scipy` (sparse CSR), `numpy`.
- The core search, indexing, and chat do not require these; they are only needed for SVD/LSI.

AI Reader — Planned Integration
- Goal: Ask natural-language questions; the AI will search top results and read selected chunks to compose an answer with citations.
- UX:
  - Ask box + “Ask AI” button; answer appears at the top with inline citations [paperId] and a sources list.
  - Controls for top-K papers, per-paper chunk cap, and model selection.
- Backend:
  - Retrieval uses existing search; context builder selects and trims chunks under a token budget.
  - Provider-agnostic LLM interface with support for local (ollama/llama.cpp) and OpenAI-compatible APIs.
  - Background job with `/api/ask` and `/api/ask/status` endpoints.
- Data:
  - Start with `chunk.text` (already normalized from LaTeX). Optionally add raw LaTeX extraction and per-paper plaintext cache.
- Safety/limits:
  - Sensible defaults (e.g., top_k=5 papers, ~6–8k token context); graceful fallback to extractive summarization when no LLM is configured.

Current Plan (high-level)
1) Define UX and answer format
2) Abstract LLM provider interface
3) Implement retrieval + context builder
4) Add /api/ask endpoint + job
5) Add UI Ask panel + citations
6) Add initial providers (local/OpenAI)
7) Tune: limits, caching, eval


Roadmap (Scaling Toward 2M Papers)
- Sharded query fan‑out: keep multiple shard DBs and query them in parallel; merge top‑K in memory to avoid single‑DB hotspots.
- Compressed postings: store one postings BLOB per token (varint‑delta docIDs + tf; optional zstd) or a mmap’d postings.bin with offsets in `token`.
- Index pruning: drop extreme high‑df tokens (true stoplist); optionally keep only K rarest tokens per chunk to shrink postings without hurting recall.
- Two‑stage ranking: doc‑level postings for recall; chunk‑level re‑rank on top‑N docs for precision.
- Optional engine integration: Tantivy/OpenSearch for BM25 recall (custom analyzer for LaTeX/math), keep local cosine/SVD re‑rank.

Next Tokenizer Improvements
- Wrapper neutralization: treat `\left/\right` and size macros (`\big`, `\Big`, etc.) as presentation; canonicalize/ignore at both query and index time.
- Full unification: refactor `tools/vectorize.py` to import the shared tokenizer and rebuild the index for perfect parity.
- Benchmarks: small harness with math‑heavy snippet sets; track P/recall/latency and ablations (BM25 vs cosine; with/without expansions).

Legacy Copy
- A legacy copy of the app (pre‑changes) was created at `/home/Brandon/ProjectSearchBar_legacy`. You can run it independently (set a different port with `PROJECTSEARCHBAR_PORT`).
