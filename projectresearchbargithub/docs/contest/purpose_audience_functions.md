Title: ProjectSearchBar — Purpose, Audience, Functions, Pathway to Satisfaction

1) Purpose
- Provide a fast, local-first academic paper search with math‑aware tokenization (text + LaTeX) and an “Agent” that can read Top‑N papers and report back with exact evidence (quotes + [paperId] citations).
- Make it practical to answer targeted research questions from raw LaTeX without sending your paper library to the cloud.

2) Audience
- Researchers, graduate/undergraduate students, and engineers who:
  - Work with math-heavy papers (LaTeX formulas, theorems, proofs).
  - Need fast local search and evidence extraction from PDFs/LaTeX without data-leak risk.
  - Prefer reproducible, offline-friendly workflows.

3) Core Functions
- Local Index + Search
  - Tokenization (text + LaTeX), TF–IDF cosine + BM25 ranking, optional SVD re-rank.
  - Metadata enrichment (title/authors/subjects/abstract) from arXiv (with offline fallback).
- Paper Chat
  - Per-result AI chat for a single paper (no agent controls). Loads LaTeX context automatically.
- Agent Mission (single agent)
  - Reads Top‑N papers in displayed order with a fresh context per paper.
  - Large read budget to capture as much LaTeX as the model allows.
  - Returns FOUND/NOT_FOUND with exact quotes and [paperId] citations; offline excerpts if model not available.
  - Context meter (approx chars/tokens used).
- Settings + Diagnostics
  - LLM model/API key storage, test connectivity.
  - Index build status, DB stats, simple terminal log.

4) Pathway to Satisfaction (Primary Use Case)
- Step 1: User enters a math-heavy query (can include $\LaTeX$) and clicks Search.
- Step 2: Top results appear (ranked by cosine/BM25); user verifies metadata/abstracts.
- Step 3: User clicks Agent, types an objective (e.g., “Find theorem statements proving X; include section numbers”), sets Top‑N (default 10), and clicks Run Agent.
- Step 4: Agent processes papers sequentially (fresh session each). For each paper, it reports FOUND/NOT_FOUND with quotes and citations.
- Step 5: User reads the streamed evidence in the chat and can follow up with clarifying questions or re-run with a refined objective.
- Result: A concise, evidence-backed answer to the user’s question sourced from local LaTeX.

