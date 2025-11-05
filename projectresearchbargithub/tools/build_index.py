#!/usr/bin/env python3
from __future__ import annotations

"""
Convenience wrapper to build the SQLite index from the project's vectors folder.

Uses:
- Scan: ProjectSearchBar/data/vectors
- Output DB: ProjectSearchBar/data/index.sqlite

Usage:
  python3 tools/build_index.py --reset
  python3 tools/build_index.py --limit-papers 1000
"""

import argparse
from pathlib import Path
import sys

import os
import sys
from pathlib import Path as _Path

# Ensure package import works when run as a script
_HERE = _Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ProjectSearchBar import config
from ProjectSearchBar.tools.index_merge import merge_index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build global SQLite index from project vectors")
    ap.add_argument("--reset", action='store_true', help="Drop and recreate schema before ingest")
    ap.add_argument("--limit-papers", type=int, default=None, help="Limit number of papers during ingest (debug)")
    args = ap.parse_args(argv)

    scan = config.VECTORS_DIR
    db = config.DB_PATH
    scan.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build_index] scan={scan}")
    print(f"[build_index] db  ={db}")
    # Build to a temp file, then atomically swap in to avoid partial reads.
    temp = db.parent / 'index.build.sqlite'
    try:
        if temp.exists():
            temp.unlink()
    except Exception:
        pass
    # Speed-oriented build: index_merge sets fast PRAGMAs on temp DB.
    merge_index(scan, temp, reset=True, limit_papers=args.limit_papers)
    print(f"[build_index] Swapping {temp} -> {db}")
    try:
        os.replace(str(temp), str(db))
    except Exception:
        temp.replace(db)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
