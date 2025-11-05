#!/usr/bin/env python3
from __future__ import annotations

"""
Batch vectorize a directory tree of arXiv sources (.tar.gz/.tgz/.gz) into per-paper
output folders under a vectors directory inside the project.

Usage:
  python3 tools/batch_vectorize.py \
    --src /home/Brandon/arxiv_tex_selected \
    --out ./data/vectors \
    --limit 0 \
    --workers auto

Notes:
- Parallelizes across CPU cores using a process pool.
- Skips already-processed papers (existing chunks.jsonl).
- If env PROJECTSEARCHBAR_STATUS is set, writes progress JSON for the UI.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple
import os
import concurrent.futures as cf
import time
import json


def iter_archives(src: Path) -> Iterable[Path]:
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith('.tar.gz') or name.endswith('.tgz') or name.endswith('.gz'):
            yield p


def strip_suffixes(name: str) -> str:
    for suf in ('.tar.gz', '.tgz', '.gz'):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _write_status(state: dict):
    status_path = os.environ.get('PROJECTSEARCHBAR_STATUS')
    if not status_path:
        return
    try:
        obj = {**state, 'ts': time.time()}
        build_id = os.environ.get('PROJECTSEARCHBAR_BUILD_ID')
        if build_id:
            obj['build_id'] = build_id
        sp = Path(status_path)
        tmp = sp.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj), encoding='utf-8')
        try:
            os.replace(str(tmp), str(sp))
        except Exception:
            tmp.replace(sp)
    except Exception:
        pass


def _process_one(args: Tuple[Path, Path, str]) -> Tuple[str, str]:
    src_file, out_root, pid = args
    try:
        from ProjectSearchBar.tools import vectorize as av
        out_dir = out_root / pid
        av.process_archive(src_file, out_dir)
        return ('ok', pid)
    except Exception as e:
        return (f'fail:{e}', pid)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='Max files to process (0=all)')
    ap.add_argument('--workers', type=str, default='auto', help='Workers: number or "auto" for CPU count (>=2). 0/1 disables parallelism.')
    args = ap.parse_args(argv)

    # Lazy import to avoid conflicts
    from ProjectSearchBar.tools import vectorize as av

    args.out.mkdir(parents=True, exist_ok=True)
    # Build task list
    tasks: list[Tuple[Path, Path, str]] = []
    total_found = 0
    for f in iter_archives(args.src):
        pid = strip_suffixes(f.name)
        total_found += 1
        out_dir = args.out / pid
        if (out_dir / 'chunks.jsonl').exists():
            continue
        tasks.append((f, args.out, pid))
        if args.limit and len(tasks) >= args.limit:
            break

    total_to_process = len(tasks)
    _write_status({'ok': True, 'state': 'vectorizing', 'total_files': total_to_process, 'done_files': 0, 'skipped': total_found - total_to_process})

    # Determine workers
    if args.workers == 'auto':
        cpu = os.cpu_count() or 2
        workers = max(2, cpu)
    else:
        try:
            workers = int(args.workers)
        except Exception:
            workers = 1
        workers = max(1, workers)

    if workers <= 1:
        # Serial fallback
        count = 0
        failures = 0
        for src_file, out_root, pid in tasks:
            out_dir = out_root / pid
            try:
                print(f"[PROC] {pid}")
                av.process_archive(src_file, out_dir)
                count += 1
            except Exception as e:
                failures += 1
                print(f"[FAIL] {pid}: {e}", file=sys.stderr)
            _write_status({'ok': True, 'state': 'vectorizing', 'total_files': total_to_process, 'done_files': count, 'skipped': total_found - total_to_process})
        print(f"[DONE] processed={count} failures={failures}")
        return

    print(f"[POOL] workers={workers} tasks={total_to_process} (skipped existing={total_found - total_to_process})")
    done = 0
    failures = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for status, pid in ex.map(_process_one, tasks, chunksize=1):
            done += 1
            if status != 'ok':
                failures += 1
                print(f"[FAIL] {pid}: {status}", file=sys.stderr)
            if done % 10 == 0 or done == total_to_process:
                print(f"[POOL] {done}/{total_to_process} done (failures={failures})")
            _write_status({'ok': True, 'state': 'vectorizing', 'total_files': total_to_process, 'done_files': done, 'skipped': total_found - total_to_process})
    print(f"[DONE] processed={done} failures={failures}")


if __name__ == '__main__':
    main()
