from __future__ import annotations

from pathlib import Path
import os


"""
Configuration for ProjectSearchBar. Adds support for selecting between two UI
skins (ui1 and ui2). Default is ui2. You can override via env var:
  PROJECTSEARCHBAR_UI=ui1  -> use classic UI
  PROJECTSEARCHBAR_UI=ui2  -> use redesigned UI (default)
"""

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# UI root (UI2 only)
UI2_DIR = BASE_DIR / "ui2" / "public"
UI_PUBLIC = UI2_DIR
# No fallback to classic UI; we no longer serve UI1
UI_PUBLIC_FALLBACK = None

# Ensure data folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "vectors").mkdir(parents=True, exist_ok=True)

# External papers source (already downloaded)
# Default now points inside the project data folder.
# Override via env var PROJECTSEARCHBAR_PAPERS if desired.
PAPERS_SRC = Path(os.environ.get("PROJECTSEARCHBAR_PAPERS", str(DATA_DIR / "arxiv_tex_selected")))

# Local outputs
VECTORS_DIR = DATA_DIR / "vectors"
DB_PATH = DATA_DIR / "index.sqlite"

# Server settings
HOST = os.environ.get("PROJECTSEARCHBAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROJECTSEARCHBAR_PORT", "8360") or 8360)
