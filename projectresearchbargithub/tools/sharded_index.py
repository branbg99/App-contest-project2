#!/usr/bin/env python3
from __future__ import annotations

"""
Sharded index builder: builds multiple shard SQLite indexes in parallel,
then merges them into a single final DB.

Usage (example):
  python3 -m ProjectSearchBar.tools.sharded_index \
    --scan /home/Brandon/ProjectSearchBar/data/vectors \
    --out-db /home/Brandon/ProjectSearchBar/data/index.sqlite \
    --shards 4 --workers 4 --reset

Notes:
- Each shard is built by symlinking a subset of paper folders into a temp scan dir
  and invoking the existing index_merge on it (fast build).
- Merge step reads shard DBs, consolidates tokens/papers/chunks/postings, recomputes df and norms,
  creates final indexes, then atomically swaps the result into --out-db.
"""

import argparse
import concurrent.futures as cf
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def list_papers(scan: Path) -> list[Path]:
    return [p for p in sorted(scan.iterdir()) if p.is_dir()]


def chunked(seq: list, n: int) -> list[list]:
    k = max(1, n)
    out: list[list] = [[] for _ in range(k)]
    for i, item in enumerate(seq):
        out[i % k].append(item)
    return out


def make_symlink_scan_dir(papers: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in papers:
        link = dest / p.name
        try:
            if link.exists() or link.is_symlink():
                continue
            os.symlink(str(p), str(link))
        except FileExistsError:
            pass


def build_shard(scan_dir: Path, db_path: Path, reset: bool = True) -> None:
    # Delegate to index_merge via module call
    from ProjectSearchBar.tools.index_merge import merge_index
    os.environ.setdefault('PROJECTSEARCHBAR_FAST_BUILD', '1')
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass
    merge_index(scan_dir, db_path, reset=reset, limit_papers=None)


def fast_pragmas(con: sqlite3.Connection) -> None:
    try:
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute("PRAGMA mmap_size = 536870912")
        con.execute("PRAGMA page_size = 32768")
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA cache_size=-200000")
        con.execute("PRAGMA busy_timeout=20000")
    except Exception:
        pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS token (
  id INTEGER PRIMARY KEY,
  term TEXT UNIQUE NOT NULL,
  df INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS paper (
  id INTEGER PRIMARY KEY,
  arxiv_id TEXT UNIQUE,
  title TEXT,
  year INT,
  path TEXT
);
CREATE TABLE IF NOT EXISTS chunk (
  id INTEGER PRIMARY KEY,
  paper_id INT NOT NULL,
  kind TEXT,
  text TEXT,
  norm REAL,
  FOREIGN KEY(paper_id) REFERENCES paper(id)
);
CREATE TABLE IF NOT EXISTS posting (
  token_id INT NOT NULL,
  chunk_id INT NOT NULL,
  tf REAL NOT NULL,
  FOREIGN KEY(token_id) REFERENCES token(id),
  FOREIGN KEY(chunk_id) REFERENCES chunk(id)
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_token_term ON token(term);
CREATE INDEX IF NOT EXISTS ix_posting_token ON posting(token_id);
CREATE INDEX IF NOT EXISTS ix_posting_chunk ON posting(chunk_id);
CREATE INDEX IF NOT EXISTS ix_chunk_paper ON chunk(paper_id);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)


def create_indexes(con: sqlite3.Connection) -> None:
    con.executescript(INDEX_SQL)


def merge_shards(shard_dbs: list[Path], out_db: Path) -> None:
    tmp = out_db.with_suffix('.build.sqlite')
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    con_out = sqlite3.connect(str(tmp), timeout=30.0, check_same_thread=False)
    fast_pragmas(con_out)
    ensure_schema(con_out)

    # Token term -> id in final
    term_to_id: dict[str, int] = {}
    # Paper arxiv_id -> id in final
    paper_to_id: dict[str, int] = {}

    def upsert_token(term: str) -> int:
        if term in term_to_id:
            return term_to_id[term]
        cur = con_out.execute("INSERT OR IGNORE INTO token(term, df) VALUES(?, 0)", (term,))
        tid = con_out.execute("SELECT id FROM token WHERE term=?", (term,)).fetchone()[0]
        term_to_id[term] = tid
        return tid

    def upsert_paper(aid: str, path: str | None = None) -> int:
        if aid in paper_to_id:
            return paper_to_id[aid]
        con_out.execute("INSERT OR IGNORE INTO paper(arxiv_id, path) VALUES(?, ?)", (aid, path or ''))
        pid = con_out.execute("SELECT id FROM paper WHERE arxiv_id=?", (aid,)).fetchone()[0]
        paper_to_id[aid] = pid
        return pid

    postings_batch: list[tuple[int, int, float]] = []
    BATCH = 100000

    for sdb in shard_dbs:
        con_s = sqlite3.connect(str(sdb), timeout=30.0, check_same_thread=False)
        # Map shard token_id -> term
        s_token: dict[int, str] = {tid: term for (tid, term) in con_s.execute("SELECT id, term FROM token")}
        # Map shard paper_id -> final paper_id
        s_paper_to_out: dict[int, int] = {}
        for pid, aid, path in con_s.execute("SELECT id, arxiv_id, path FROM paper"):
            s_paper_to_out[int(pid)] = upsert_paper(aid or str(pid), path)
        # Insert chunks for this shard and record mapping shard chunk_id -> out chunk_id
        s_chunk_to_out: dict[int, int] = {}
        cur = con_s.execute("SELECT id, paper_id, kind, text, norm FROM chunk")
        rows = cur.fetchmany(10000)
        while rows:
            with con_out:
                for cid, spid, kind, text, norm in rows:
                    out_pid = s_paper_to_out.get(int(spid))
                    con_out.execute(
                        "INSERT INTO chunk(paper_id, kind, text, norm) VALUES(?, ?, ?, ?)",
                        (out_pid, kind, text, None),  # recalc norms later
                    )
                    out_cid = con_out.execute("SELECT last_insert_rowid()").fetchone()[0]
                    s_chunk_to_out[int(cid)] = int(out_cid)
            rows = cur.fetchmany(10000)

        # Stream shard postings and copy into final with remapped ids
        curp = con_s.execute("SELECT token_id, chunk_id, tf FROM posting")
        rows = curp.fetchmany(200000)
        while rows:
            for tid, cid, tf in rows:
                term = s_token.get(int(tid))
                if term is None:
                    continue
                out_tid = upsert_token(term)
                out_cid = s_chunk_to_out.get(int(cid))
                if out_cid is None:
                    continue
                postings_batch.append((out_tid, out_cid, float(tf)))
                if len(postings_batch) >= BATCH:
                    with con_out:
                        con_out.executemany(
                            "INSERT INTO posting(token_id, chunk_id, tf) VALUES(?, ?, ?)", postings_batch
                        )
                    postings_batch.clear()
            rows = curp.fetchmany(200000)
        if postings_batch:
            with con_out:
                con_out.executemany(
                    "INSERT INTO posting(token_id, chunk_id, tf) VALUES(?, ?, ?)", postings_batch
                )
            postings_batch.clear()
        con_s.close()

    # Compute df using a temp table
    with con_out:
        con_out.execute("DROP TABLE IF EXISTS df_tmp")
        con_out.execute("CREATE TEMP TABLE df_tmp(token_id INT, cnt INT)")
        con_out.execute(
            "INSERT INTO df_tmp SELECT token_id, COUNT(DISTINCT chunk_id) FROM posting GROUP BY token_id"
        )
        con_out.execute(
            "UPDATE token SET df = COALESCE((SELECT cnt FROM df_tmp WHERE df_tmp.token_id = token.id), 0)"
        )

    # Compute norms: single-pass over postings
    print("[sharded_merge] Computing norms...")
    idf_cache: dict[int, float] = {}
    total_chunks = con_out.execute("SELECT COUNT(1) FROM chunk").fetchone()[0]
    for tid, df in con_out.execute("SELECT id, df FROM token"):
        idf = math.log((1.0 + total_chunks) / (1.0 + float(df))) + 1.0
        idf_cache[int(tid)] = idf

    sums = defaultdict(float)
    cur = con_out.execute("SELECT chunk_id, token_id, tf FROM posting")
    rows = cur.fetchmany(1000000)
    logf = math.log
    while rows:
        for cid, tid, tf in rows:
            tfw = 1.0 + logf(float(tf))
            idf = idf_cache.get(int(tid), 1.0)
            sums[int(cid)] += (tfw * idf) ** 2
        rows = cur.fetchmany(1000000)
    with con_out:
        con_out.executemany(
            "UPDATE chunk SET norm=? WHERE id=?",
            [(math.sqrt(v) if v > 0 else 1.0, cid) for cid, v in sums.items()],
        )

    # Create indexes
    print("[sharded_merge] Creating indexes...")
    create_indexes(con_out)
    con_out.close()

    # Atomic swap
    print(f"[sharded_merge] Swapping {tmp} -> {out_db}")
    try:
        os.replace(str(tmp), str(out_db))
    except Exception:
        Path(tmp).replace(out_db)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sharded build + merge index from vectors")
    ap.add_argument("--scan", type=Path, required=True, help="Vectors folder containing per-paper subfolders")
    ap.add_argument("--out-db", type=Path, required=True, help="Final SQLite database path")
    ap.add_argument("--shards", type=int, default=4, help="Number of shards to build in parallel")
    ap.add_argument("--workers", type=int, default=4, help="Max parallel shard builders")
    ap.add_argument("--reset", action='store_true', help="Reset shard DBs when building")
    args = ap.parse_args(argv)

    papers = list_papers(args.scan)
    if not papers:
        print(f"No paper folders found under: {args.scan}", file=sys.stderr)
        return 2
    shards = chunked(papers, max(1, args.shards))
    with tempfile.TemporaryDirectory(prefix='psb_sharded_') as tmpd:
        tmpdir = Path(tmpd)
        shard_scan_dirs: list[Path] = []
        shard_db_paths: list[Path] = []
        for i, spapers in enumerate(shards):
            sdir = tmpdir / f'scan_shard_{i+1}'
            make_symlink_scan_dir(spapers, sdir)
            shard_scan_dirs.append(sdir)
            shard_db_paths.append(tmpdir / f'shard_{i+1}.sqlite')
        print(f"[sharded] Building {len(shards)} shard DBs in parallel...")
        with cf.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(build_shard, sdir, sdb, True) for sdir, sdb in zip(shard_scan_dirs, shard_db_paths)]
            for f in cf.as_completed(futs):
                exc = f.exception()
                if exc:
                    print("[sharded] shard build error:", exc, file=sys.stderr)
                    raise exc
        print("[sharded] Merging shards into final DB...")
        args.out_db.parent.mkdir(parents=True, exist_ok=True)
        merge_shards(shard_db_paths, args.out_db)
    print("[sharded] Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

