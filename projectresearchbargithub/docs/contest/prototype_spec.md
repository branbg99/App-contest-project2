Title: ProjectSearchBar — Prototype Spec (for Figma)

Goal: Provide a high-fidelity clickthrough prototype illustrating the core user flows and the Agent Mission.

Suggested Pages / Frames
1) Landing / Search
   - Header: title, Settings, AI status chip, DB banner.
   - Search bar: input + Search + Agent button.
   - Sort bar: cosine / BM25 / reset.
   - Results list: result card (title, authors, subjects, abstract excerpt, score meters) + AI button (opens paper chat).

2) Paper Chat (per-result)
   - Right panel with: paper title, Attach chips, chat body (KaTeX), input + Send.
   - No agent controls in this view.

3) Agent Mission Panel
   - Right panel labeled “Agent”.
   - Controls: Objective (multiline), Top‑N (numeric), Run Agent, Cancel.
   - Mission bar: “Reading X/N: [paperId] • Context ~Nk chars (~Mk tokens)”.
   - Findings stream: FOUND/NOT_FOUND cards with quotes and [paperId] citations.

4) Settings (AI)
   - Model selector, API key field, Test button, status text.

User Flows to Demonstrate
F1) Search → Results → Open Paper Chat → Ask a question.
F2) Search → Agent → Enter objective → Run Agent → Watch sequential findings.
F3) Settings → Test AI connectivity (OK and fail states).

Design Notes
- Keep Chat vs Agent visually distinct (e.g., subtle color on Agent header).
- Use chips for subjects, slim score bars, muted abstract text.
- KaTeX rendering hints: show an example math bubble.

Hand-off Checklist for Judges (Prototype)
- Share link with “Anyone with the link can view”.
- Include frames: Landing + Results, Paper Chat, Agent Panel, Settings.
- Connect interactions: Search button to Results; AI button to Paper Chat; Agent button to Agent Panel; Run Agent to example findings.

