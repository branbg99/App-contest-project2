Title: ProjectSearchBar — Proposed Technical Aspects + Architecture

1) Components
- Client (UI2): vanilla TypeScript/DOM, KaTeX rendering for math, minimal CSS.
- Server (Python): `http.server` handler + SQLite; endpoints for search, metadata, chat, agent settings.
- Data:
  - Local LaTeX archives/folders: `ProjectSearchBar/data/arxiv_tex_selected/`.
  - Index DB: `ProjectSearchBar/data/index.sqlite` (token, paper, chunk, posting).
  - Cached AI: `ProjectSearchBar/data/ai_cache/` (concatenated LaTeX text per paper).

2) Key Libraries / Services
- Python stdlib: `sqlite3`, `http.server`, `threading`, `tarfile`.
- UI: KaTeX for math, native DOM APIs.
- LLM (optional): OpenAI Chat Completions (`gpt-4o` / `gpt-4o-mini`) via HTTPS; offline excerpts fallback when unreachable.

3) APIs (selected)
- Search: `POST /api/search` with `{ query, scoring?, bm25?, svd? }` → ranked results.
- Metadata: `GET /api/paper/meta?ids=...` → titles/authors/subjects/abstract.
- LLM settings: `GET/POST /api/llm/settings`, `POST /api/llm/test`.
- Chat (single-paper): `POST /api/chat/start | add_paper | message | clear`, `GET /api/chat/context`.
- Agent settings (persona/defaults): `GET/POST /api/agent/settings` (persisted to `data/agent_settings.json`).

4) Agent Flow (Single Agent)
- User enters objective → UI reads Top‑N paper IDs from current results order.
- For each paper (sequential):
  1) Fresh chat session, attach only this paper.
  2) Ask with system persona; large token budget to read as much LaTeX as supported.
  3) If LLM answers, stream FOUND/NOT_FOUND + quotes; else run offline excerpt selector.
  4) Update context meter, progress, and timing.

5) Architecture Diagram (Mermaid)
```mermaid
flowchart LR
  subgraph Client [UI2]
    Q[Search Bar]
    R[Results List]
    C[Paper Chat]
    A[Agent Mission]
  end

  subgraph Server [Python HTTP API]
    H[Request Handler]
    IDX[(SQLite Index)]
    FS[(arXiv LaTeX Store)]
    AI[data/ai_cache]
  end

  Q -- POST /api/search --> H
  H -- read index --> IDX
  H -- results --> R

  R -- click AI --> C
  C -- POST /api/chat/start|add_paper|message --> H
  H -- read LaTeX --> FS
  H -- cache text --> AI

  A -- GET /api/agent/settings --> H
  A -- sequential per-paper chat --> H
  H -- (optional) HTTPS to OpenAI -->|LLM|
  H -- fallback/verify via LaTeX --> FS
```

6) Scaling Considerations (summary)
- Shard index for 100k+ papers; two-stage retrieval (paper triage → chunk precision).
- Keep top‑M idf tokens per chunk; target 6–12 chunks/paper to bound postings.
- Optional JL/SVD triage for candidates; ANN for fast top‑K.

