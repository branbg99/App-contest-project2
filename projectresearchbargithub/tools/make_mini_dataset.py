#!/usr/bin/env python3
from __future__ import annotations

"""
Create a small, GitHub-friendly dataset under data/mini (or a custom path) by
sampling a subset of papers from the existing vectors directory and rebuilding
an index for just those files.

Usage examples:
  python3 -m ProjectSearchBar.tools.make_mini_dataset --papers 50 --out data/mini
  PROJECTSEARCHBAR_DATA_DIR=./data/mini python3 -m ProjectSearchBar.tools.make_mini_dataset --papers 100
  # Then run the app against the mini dataset:
  PROJECTSEARCHBAR_DATA_DIR=./data/mini python3 launch.py

Notes:
- Keeps SVD off to minimize size. Focuses on TF–IDF index only.
- Selects the first N papers by default; you can randomize.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ProjectSearchBar import config
from ProjectSearchBar.tools.index_merge import merge_index


def list_papers(vectors_dir: Path) -> list[Path]:
    out = []
    if not vectors_dir.exists():
        return out
    for p in vectors_dir.iterdir():
        if p.is_dir():
            # Require a chunks.jsonl to treat as a paper folder
            cj = p / "chunks.jsonl"
            if cj.exists():
                out.append(p)
    return sorted(out)


def count_chunks(chunks_path: Path, max_read: int | None = None) -> int:
    n = 0
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for n, _ in enumerate(f, start=1):
                if max_read and n >= max_read:
                    break
    except FileNotFoundError:
        return 0
    return n


def copy_subset(src_vectors: Path, dst_vectors: Path, max_chunks: int | None) -> None:
    dst_vectors.mkdir(parents=True, exist_ok=True)
    # The caller already selected and iterates papers; this helper truncates chunks if requested.
    pass


def build_mini(scan_dir: Path, out_db: Path, limit_papers: int | None) -> None:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    temp = out_db.parent / "index.build.sqlite"
    if temp.exists():
        try:
            temp.unlink()
        except Exception:
            pass
    merge_index(scan_dir, temp, reset=True, limit_papers=limit_papers)
    try:
        os.replace(str(temp), str(out_db))
    except Exception:
        Path(temp).replace(out_db)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create a small dataset under data/mini")
    ap.add_argument("--out", type=str, default=str(config.BASE_DIR / "data" / "mini"), help="Output mini dataset directory (default: data/mini)")
    ap.add_argument("--papers", type=int, default=50, help="Number of papers to include")
    ap.add_argument("--max-chunks", type=int, default=200, help="Max chunks per paper (truncate if larger)")
    ap.add_argument("--random", action="store_true", help="Randomize paper selection (default: take first N)")
    ap.add_argument("--src", type=str, default=str(config.VECTORS_DIR), help="Source vectors directory (default: data/vectors from current config)")
    args = ap.parse_args(argv)

    src_vectors = Path(args.src).resolve()
    out_dir = Path(args.out).resolve()
    out_vectors = out_dir / "vectors"
    out_db = out_dir / "index.sqlite"

    papers = list_papers(src_vectors)
    if not papers:
        print(f"[mini] No papers found in {src_vectors}. Build vectors first.")
        return 2

    if args.random:
        random.shuffle(papers)

    selected = []
    for p in papers:
        if len(selected) >= args.papers:
            break
        selected.append(p)

    print(f"[mini] Source vectors: {src_vectors}")
    print(f"[mini] Output dir   : {out_dir}")
    print(f"[mini] Choosing     : {len(selected)} papers (requested {args.papers})")

    # Copy paper folders, truncating chunks.jsonl to --max-chunks
    out_vectors.mkdir(parents=True, exist_ok=True)
    for sp in selected:
        dp = out_vectors / sp.name
        dp.mkdir(parents=True, exist_ok=True)
        # Copy minimal files
        cj_src = sp / "chunks.jsonl"
        cj_dst = dp / "chunks.jsonl"
        tok_src = sp / "tokens.jsonl"
        tok_dst = dp / "tokens.jsonl"

        # Truncate chunks.jsonl lines
        if cj_src.exists():
            with cj_src.open("r", encoding="utf-8") as fsrc, cj_dst.open("w", encoding="utf-8") as fdst:
                for i, line in enumerate(fsrc, start=1):
                    fdst.write(line)
                    if args.max_chunks and i >= args.max_chunks:
                        break
        # Tokens file may be large; only copy if reasonably small
        try:
            if tok_src.exists() and tok_src.stat().st_size < 10_000_000:  # 10MB guard
                shutil.copy2(tok_src, tok_dst)
        except Exception:
            pass

    # Build the mini index from the copied vectors
    print("[mini] Building mini index...")
    build_mini(out_vectors, out_db, limit_papers=None)
    print(f"[mini] Done. Mini dataset at: {out_dir}")
    print("[mini] To run against this dataset:")
    print(f"       PROJECTSEARCHBAR_DATA_DIR={out_dir} python3 launch.py")

    # Write a small metadata file for reference
    meta = {
        "papers": len(selected),
        "max_chunks_per_paper": args.max_chunks,
        "source_vectors": str(src_vectors),
    }
    (out_dir / "README.mini.md").write_text(
        """Mini Dataset for ProjectSearchBar\n\n"""
        + json.dumps(meta, indent=2)
        + "\n\nGenerated with: tools/make_mini_dataset.py\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

