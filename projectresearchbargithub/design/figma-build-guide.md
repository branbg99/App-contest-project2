ProjectSearchBar — Figma Rebuilt UI (Guide)

Overview
- Target frame: 1440×900 desktop, 8px grid.
- Fonts: Inter (UI), Roboto Mono (mono). Substitute SF Pro if preferred on macOS.
- Import tokens: Use Tokens Studio → Import → JSON → `design/figma-tokens.json`.

1) Create Color/Text/Effect Styles
- After importing tokens, create Figma Styles from tokens (Tokens Studio: Export as Styles) so fills/strokes/shadows are reusable.
- Map names:
  - Colors: Color/BG, Color/BG Alt, Color/Panel, Color/Panel Alt, Color/Text, Color/Muted, Color/Border, Color/Link, Color/Accent, Color/Accent Text, Color/Accent Border.
  - Effects: Shadow/Strong. Gradient: Gradient/Accent→Link.
  - Text: Body/14, Small/12, Heading/14, Mono/12.

2) Components (with variants)

Button (component)
- Auto Layout H: padding 10×14, gap 6. Radius 10.
- Variants:
  - kind: Primary | Secondary
  - state: Default | Hover | Pressed | Disabled | Loading
- Primary: Fill Accent, Stroke Accent Border, Text Accent Text.
- Secondary: Fill Panel Alt, Stroke Border, Text Text.
- Loading: add centered spinner (18×18) overlay; other text at 0% opacity.

Search Bar (component)
- Container: Auto Layout H, padding 14×14, gap 10, radius 12, Fill Panel, Stroke Border, Effect Shadow/Strong (focus variant only).
- Slots: Text input (radius 10, Fill Panel Alt, Stroke Border, padding 14×16), Actions group (Buttons/Chips).
- Variants: focused: true|false (toggle stroke Accent + shadow).

Chip (component)
- Auto Layout H, padding 2×8, gap 6, radius Pill, Fill Panel, Stroke Border, Text Small/12.

Sort Button (component)
- Auto Layout H, padding 2×8, radius Pill.
- Variants: active: true|false, kind: primary|secondary.
- Active: Fill Accent, Stroke Accent Border, Text Accent Text. Inactive: Fill Panel Alt (or Panel), Stroke Border, Text Text.

Score Bar (component)
- Frame 120×6, radius Pill; Stroke Border, Fill Panel.
- Inner fill: 100% width rectangle using Gradient/Accent→Link; make instances with width 24/48/72/96/120 for 20/40/60/80/100.

Result Card (component)
- Container: Auto Layout V, padding 12, gap 8, radius 10, Fill BG Alt, Stroke Border.
- Header row: Title (Text link color), Subject chips (Chip instances), Score group (Score Bar + numeric text in Muted).
- Body slot: paragraph preview text (Body/14).

Header (component)
- Frame: Height ~44, Fill Panel, bottom Stroke Border.
- Left: App title (Heading/14, Link color). Right: meta banners (small dashed boxes using Panel + Border).

Terminal Panel (component)
- Container: radius 12, Fill Panel, Stroke Border, Effect Shadow/Strong.
- Head: title (Small/12 Muted), mini buttons.
- Body: scroll area with lines using Mono/12 (Text), subtle background overlay.

Paper Chat Panel (component)
- Head (title + actions), Attach row (Chip instances), Body (placeholder frame for iframe), Foot (input + Button).

Chat Message (component)
- Variants: role: user|assistant.
- Layout: Avatar (36×36 with role color) + Bubble (Auto Layout, padding 12×16, radius 8).
- Colors: User bubble BG `#3c3836`; Assistant bubble BG Panel.

3) Screens (frames)

Search — Centered
- Header (component). Search Bar. Status text (Small/12 Muted). Sort bar (Sort Buttons). List of Result Card instances.
- Center the main column: place inside a 1400px-width container; set frame Clip contents off.

Search — Split
- Two-column layout: Left = Search + Results; Right = Terminal Panel. Keep gaps 14px.

Paper Chat
- Same grid as Split; right column replaced by Paper Chat Panel instance.

LLM Chat (Full page)
- Stack of Chat Message instances; add a typing indicator card (3 dots) using Muted color.

4) Prototype wiring
- Start point: Search — Centered.
- Toggle: a small button switches Centered <→ Split (swap screens or variants).
- Result title click: Navigate to Paper Chat.
- Chat “Send” button: simple auto-scroll interaction (Mock: Navigate to same frame with another message visible).
- Use Instant transitions; avoid Smart Animate unless animating simple position/opacity.

5) Measurements and spacing
- Gaps: 8–14px (use 8/10/12/14 tokens for consistency).
- Radii: 10–12px; pills for chips/sort buttons/score track.
- Shadows: only on interactive containers (search focus, panels).

6) Notes on fidelity
- Match link color for titles (#83a598). Keep muted copy (#bdae93) for meta.
- Score gradient uses Accent→Link; keep bar height 6px.
- Use Text styles consistently (Body/14, Small/12, Heading/14).

7) Optional: Variables
- Create Figma Variables set `Theme/Dark` and bind fills/strokes/radii to tokens so future theming is trivial. Map tokens from `figma-tokens.json` to variables with the same names.

8) Export
- Create a cover page frame (key screens) and a Share link with “Can view/prototype”.

Appendix — Component naming (recommended)
- Button, SearchBar, Chip, SortButton, ScoreBar, ResultCard, Header, TerminalPanel, PaperChatPanel, ChatMessage.

