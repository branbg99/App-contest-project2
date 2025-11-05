Title: ProjectSearchBar — Security Considerations

1) Data Scope & Storage
- Local-first design: papers, index, and caches live under `ProjectSearchBar/data/` on the user’s machine.
- No external user data (no PII, no credit cards). Paper content is academic LaTeX.
- LLM key stored locally in `data/llm_settings.json` (can be rotated/cleared via Settings).

2) Network Exposure
- Server binds to `127.0.0.1` by default; no external ports exposed.
- Outbound traffic only to `api.openai.com` when LLM is used; offline excerpts otherwise.
- TLS: Python `ssl` defaults when calling OpenAI; no custom certs are shipped.

3) Private Keys & Secrets
- OpenAI API key is client-provided and stored locally in plain JSON for simplicity.
  - Mitigation: warn users to keep local machine secure; recommend per‑app API key with limited scope and easy rotation.
  - Option: move to environment variable or OS keychain if stricter storage is required.

4) Input Handling
- Search queries and chat text are plain strings used in SQL via parameterized queries; no SQL string concatenation for user data.
- LaTeX is read from local files/archives; `%` comments stripped, but content is never executed — only tokenized and sent to model.
- UI renders math via KaTeX (client-side sanitization) and does not execute user-provided scripts.

5) Threat Model & Risks
- Local compromise: if device is compromised, local papers/index and `llm_settings.json` could be read.
  - Mitigation: store data in user-owned path; allow key clear; recommend disk encryption and normal OS hardening.
- Prompt injection via model output: agent is instructed to be extractive and cite with quotes; verify quotes against local source to reduce hallucination risk.
- Denial of service: extremely large prompts/timeouts could freeze UI; mitigated by timeouts, sequential agent, and offline fallback.

6) Privacy Posture
- No telemetry, analytics, or external data uploads besides explicit LLM requests.
- Offline-compatible; users can run with no API key to keep everything local (reduced functionality).

7) Operational Controls
- Environment knobs: `PROJECTSEARCHBAR_HOST/PORT`, `PROJECTSEARCHBAR_LLM_TIMEOUT`, offline mode flag, etc.
- Rate and retry: sequential agent by default; optional backoff on 429/timeouts.

8) Future Hardening (optional)
- Keychain/credential store integration.
- Sharded index with per-shard Bloom filters to bound query time.
- Sectioned reading with verified quotes (string/fuzzy match) before display.

