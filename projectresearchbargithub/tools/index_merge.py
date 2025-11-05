#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path


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
CREATE INDEX IF NOT EXISTS ix_posting_chunk_token ON posting(chunk_id, token_id);
CREATE INDEX IF NOT EXISTS ix_chunk_paper ON chunk(paper_id);
"""


def ensure_schema(con: sqlite3.Connection):
    con.executescript(SCHEMA_SQL)


def create_indexes(con: sqlite3.Connection):
    con.executescript(INDEX_SQL)


def reset_schema(con: sqlite3.Connection):
    con.executescript(
        """
        DROP TABLE IF EXISTS posting;
        DROP TABLE IF EXISTS chunk;
        DROP TABLE IF EXISTS paper;
        DROP TABLE IF EXISTS token;
        """
    )
    ensure_schema(con)


TOKEN_CACHE_MAX = int(os.environ.get('PROJECTSEARCHBAR_TOKEN_CACHE_MAX', '500000') or '500000')

def upsert_token_ids(con: sqlite3.Connection, terms: list[str], cache: dict[str, int]) -> dict[str, int]:
    """Resolve term->token_id with minimal queries.

    Optimizations:
    - Only SELECT ids for "unseen" terms (already-cached terms skip lookup).
    - Batched INSERT OR IGNORE for unseen terms once per chunk.
    - Chunk IN(...) selects to respect SQLite variable limits.
    """
    out: dict[str, int] = {}
    if not terms:
        return out

    # Unique ordering of requested terms to keep batching stable
    uniq_terms: list[str] = list(dict.fromkeys(terms))

    # Filter to unseen terms not in cache yet
    unseen = [t for t in uniq_terms if t not in cache]
    if unseen:
        con.executemany(
            "INSERT OR IGNORE INTO token(term, df) VALUES(?, 0)",
            ((t,) for t in unseen)
        )
        # Fetch ids only for unseen terms
        CHUNK = 800  # SQLite bound variables safety margin (< 999)
        for i in range(0, len(unseen), CHUNK):
            batch = unseen[i:i + CHUNK]
            placeholders = ",".join(["?"] * len(batch))
            rows = con.execute(
                f"SELECT term, id FROM token WHERE term IN ({placeholders})",
                batch,
            ).fetchall()
            for term, tid in rows:
                cache[term] = int(tid)
        # Bound cache memory: if it grows too large, shrink but keep current terms
        if TOKEN_CACHE_MAX and len(cache) > TOKEN_CACHE_MAX:
            try:
                keep: dict[str,int] = {t: cache[t] for t in uniq_terms if t in cache}
                cache.clear()
                cache.update(keep)
            except Exception:
                # As a last resort, avoid clearing mid-batch to prevent KeyError
                pass

    # Build output mapping from the cache (no DB hits)
    for t in terms:
        tid = cache.get(t)
        if tid is not None:
            out[t] = tid
    return out


def merge_index(scan_path: Path, db_path: Path, reset: bool = False, limit_papers: int | None = None, incremental: bool = False):
    con = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    # Speed-oriented PRAGMAs for temp build
    fast = os.environ.get('PROJECTSEARCHBAR_FAST_BUILD', '1') == '1'
    low_mem = os.environ.get('PROJECTSEARCHBAR_LOW_MEM', '0') == '1'
    try:
        # In low-mem mode prefer conservative settings
        if low_mem:
            con.execute("PRAGMA temp_store = DEFAULT")
            con.execute("PRAGMA page_size = 32768")
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA cache_size=-50000")  # ~50MB
        else:
            con.execute("PRAGMA temp_store = MEMORY")
            con.execute("PRAGMA mmap_size = 536870912")  # 512MB
            con.execute("PRAGMA page_size = 32768")
            con.execute("PRAGMA locking_mode = EXCLUSIVE")
            if fast:
                con.execute("PRAGMA journal_mode=OFF")
                con.execute("PRAGMA synchronous=OFF")
                con.execute("PRAGMA cache_size=-200000")  # ~200MB
            else:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=15000")
    except Exception:
        pass
    if reset and not incremental:
        print("[index] Resetting schema")
        reset_schema(con)
    else:
        ensure_schema(con)

    token_cache: dict[str, int] = {}
    paper_count = 0
    chunk_count = 0
    posting_count = 0

    # Track document-frequency increments. This can get very large for big corpora,
    # so we periodically flush to the DB to avoid unbounded growth and huge commits.
    df_increments: Counter[int] = Counter()
    DF_FLUSH_THRESHOLD = int(os.environ.get('PROJECTSEARCHBAR_FLUSH_DF', '250000'))
    def flush_df_increments():
        nonlocal df_increments
        if not df_increments:
            return
        try:
            with con:
                con.executemany(
                    "UPDATE token SET df = df + ? WHERE id = ?",
                    ((int(inc), int(tid)) for tid, inc in df_increments.items())
                )
        finally:
            df_increments.clear()

    # Pre-filter to processable papers (those with chunks.jsonl present)
    all_dirs = [p for p in sorted(scan_path.iterdir()) if p.is_dir()]
    papers = [p for p in all_dirs if (p / 'chunks.jsonl').exists()]
    if limit_papers:
        papers = papers[:limit_papers]

    # Optional progress reporting
    status_path = os.environ.get('PROJECTSEARCHBAR_STATUS')
    build_id = os.environ.get('PROJECTSEARCHBAR_BUILD_ID')
    def write_status(obj: dict):
        if not status_path:
            return
        try:
            obj2 = {**obj, 'ts': time.time()}
            if build_id:
                obj2['build_id'] = build_id
            sp = Path(status_path)
            tmp = sp.with_suffix('.tmp')
            tmp.write_text(json.dumps(obj2), encoding='utf-8')
            try:
                os.replace(str(tmp), str(sp))
            except Exception:
                tmp.replace(sp)
        except Exception:
            pass

    total_papers = len(papers)
    done_papers = 0
    # UI-friendly display counter to avoid large visible jumps between polls
    UI_STEP = int(os.environ.get('PROJECTSEARCHBAR_DISPLAY_STEP', '500'))
    display_done = 0
    def emit_indexing_status(extra: dict | None = None):
        nonlocal display_done
        # advance display_done gradually toward done_papers
        if done_papers > display_done:
            display_done = min(done_papers, display_done + UI_STEP)
        base = {
            'ok': True,
            'state': 'indexing',
            'papers_total': total_papers,
            'papers_done': done_papers,
            'display_papers_done': display_done,
            'chunks': chunk_count,
            'postings': posting_count,
        }
        if extra:
            base.update(extra)
        write_status(base)

    emit_indexing_status({})

    # Larger batch lowers executemany overhead; keep memory reasonable
    # Reduce default batch sizes further to avoid long insert stalls at scale
    default_batch = '50000' if low_mem else '100000'
    BATCH_POSTINGS = int(os.environ.get('PROJECTSEARCHBAR_POST_BATCH', default_batch))
    # Optionally commit periodically to cap transaction size (helps RAM/disk pressure)
    commit_every_env = os.environ.get('PROJECTSEARCHBAR_COMMIT_PAPERS', '')
    if commit_every_env.strip():
        try:
            COMMIT_EVERY = max(1, int(commit_every_env))
        except Exception:
            COMMIT_EVERY = 0
    else:
        # Default to periodic commits in low-mem mode
        COMMIT_EVERY = 200 if low_mem else 0
    postings_batch: list[tuple[int, int, float]] = []  # (token_id, chunk_id, tf)

    # Use explicit periodic commits even in non-low-mem mode to avoid huge transactions
    cur = con.cursor()
    # Begin an explicit transaction
    try:
        con.execute('BEGIN')
    except Exception:
        pass
    for p in papers:
            # Keep UI counters monotonic: report completed papers, not the current in-progress one
            emit_indexing_status({'paper_current': p.name})
            # In incremental mode, skip papers already present in DB
            if incremental:
                row = cur.execute("SELECT id FROM paper WHERE arxiv_id=?", (p.name,)).fetchone()
                if row:
                    done_papers += 1
                    emit_indexing_status({})
                    continue
            cur.execute("INSERT OR IGNORE INTO paper(arxiv_id, path) VALUES(?, ?)", (p.name, str(p)))
            row = cur.execute("SELECT id FROM paper WHERE arxiv_id=?", (p.name,)).fetchone()
            if not row:
                # Could not create/find paper id; skip
                done_papers += 1
                emit_indexing_status({})
                continue
            paper_id = row[0]
            paper_count += 1

            with (p / 'chunks.jsonl').open('r', encoding='utf-8', errors='ignore') as fch:
                local_chunks = 0
                for line in fch:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    kind = rec.get('kind')
                    text = rec.get('text', '')
                    toks = rec.get('tokens') or []
                    tf_counter: Counter[str] = Counter(toks)
                    if not tf_counter:
                        continue
                    token_ids = upsert_token_ids(con, list(tf_counter.keys()), token_cache)
                    cur.execute("INSERT INTO chunk(paper_id, kind, text, norm) VALUES(?, ?, ?, NULL)", (paper_id, kind, text))
                    chunk_id = cur.lastrowid
                    chunk_count += 1
                    local_chunks += 1
                    for term, cnt in tf_counter.items():
                        tid = token_ids[term]
                        postings_batch.append((tid, int(chunk_id), float(cnt)))
                        posting_count += 1
                        df_increments[tid] += 1
                    if len(postings_batch) >= BATCH_POSTINGS:
                        con.executemany("INSERT INTO posting(token_id, chunk_id, tf) VALUES(?, ?, ?)", postings_batch)
                        postings_batch.clear()
                    # Periodically flush DF increments to bound memory/transaction size
                    if len(df_increments) >= DF_FLUSH_THRESHOLD:
                        flush_df_increments()
                    # Periodic progress update within a paper
                    if (local_chunks % 2000) == 0:
                        emit_indexing_status({'paper_current': p.name})
            # Completed this paper
            done_papers += 1
            emit_indexing_status({})
            # Periodic commit to limit giant transactions
            if (COMMIT_EVERY and (done_papers % COMMIT_EVERY == 0)) or (not COMMIT_EVERY and done_papers % 20 == 0):
                if postings_batch:
                    con.executemany("INSERT INTO posting(token_id, chunk_id, tf) VALUES(?, ?, ?)", postings_batch)
                    postings_batch.clear()
                # Also flush DF increments on periodic commits
                flush_df_increments()
                try:
                    con.commit()
                    # Start a new transaction for subsequent work
                    con.execute('BEGIN')
                except Exception:
                    pass
    # Progress update per paper
    if postings_batch:
        con.executemany("INSERT INTO posting(token_id, chunk_id, tf) VALUES(?, ?, ?)", postings_batch)
        postings_batch.clear()
    # Final DF flush for any remaining increments before norms
    flush_df_increments()
    try:
        con.commit()
    except Exception:
        pass
    emit_indexing_status({})

    # df_increments has been flushed incrementally; nothing (or very little) remains.
    if df_increments:
        print(f"[index] Applying remaining DF increments for {len(df_increments)} tokens...")
        with con:
            con.executemany(
                "UPDATE token SET df = df + ? WHERE id = ?",
                ((int(inc), int(tid)) for tid, inc in df_increments.items())
            )

    print("[index] Computing chunk norms...")
    # Advertise total postings for progress
    # Norms phase; set display_papers_done = papers_done (ingest finished)
    write_status({'ok': True, 'state': 'indexing', 'phase': 'norms', 'postings_total': posting_count, 'postings_seen': 0, 'papers_total': total_papers, 'papers_done': done_papers, 'display_papers_done': done_papers})
    total_chunks = con.execute("SELECT COUNT(1) FROM chunk").fetchone()[0]

    # Two strategies: default fast in-memory accumulation, or low-mem streaming into a temp table
    if not low_mem:
        idf_cache: dict[int, float] = {}
        for tid, df in con.execute("SELECT id, df FROM token"):
            idf = math.log((1.0 + total_chunks) / (1.0 + float(df))) + 1.0
            idf_cache[tid] = idf
        from collections import defaultdict
        sums: dict[int, float] = defaultdict(float)
        print("[index] Summing norms in one pass over postings...")
        cur = con.execute("SELECT chunk_id, token_id, tf FROM posting")
        rows = cur.fetchmany(1000000)
        total_seen = 0
        logf = math.log
        while rows:
            for cid, tid, tf in rows:
                tfw = 1.0 + logf(float(tf))
                idf = idf_cache.get(int(tid), 1.0)
                sums[int(cid)] += (tfw * idf) ** 2
            total_seen += len(rows)
            if total_seen % 5000000 == 0:
                write_status({'ok': True, 'state': 'indexing', 'phase': 'norms', 'postings_seen': total_seen, 'postings_total': posting_count, 'papers_total': total_papers, 'papers_done': done_papers, 'display_papers_done': done_papers})
            rows = cur.fetchmany(1000000)
        norms = [(math.sqrt(v) if v > 0 else 1.0, cid) for cid, v in sums.items()]
        print(f"[index] Writing {len(norms)} norms...")
        with con:
            con.executemany("UPDATE chunk SET norm=? WHERE id=?", norms)
    else:
        # Low-memory mode: spill partial sums to a temp table to avoid a giant Python dict
        print("[index] Low-memory norms: streaming into temp table...")
        with con:
            con.execute("DROP TABLE IF EXISTS norm_tmp")
            con.execute("CREATE TEMP TABLE norm_tmp (chunk_id INT PRIMARY KEY, s REAL NOT NULL DEFAULT 0.0)")
        # Lazy idf cache by token id
        idf_cache_lite: dict[int, float] = {}
        def idf_for(tid: int) -> float:
            v = idf_cache_lite.get(tid)
            if v is not None:
                return v
            row = con.execute("SELECT df FROM token WHERE id=?", (int(tid),)).fetchone()
            df = float(row[0]) if row and row[0] is not None else 0.0
            v = math.log((1.0 + total_chunks) / (1.0 + df)) + 1.0
            idf_cache_lite[tid] = v
            return v
        cur = con.execute("SELECT chunk_id, token_id, tf FROM posting")
        rows = cur.fetchmany(200000)
        total_seen = 0
        logf = math.log
        SPILL = 500000  # spill to DB when batch has this many distinct chunks
        from collections import defaultdict
        batch_sums: dict[int, float] = defaultdict(float)
        while rows:
            for cid, tid, tf in rows:
                tfw = 1.0 + logf(float(tf))
                idf = idf_for(int(tid))
                batch_sums[int(cid)] += (tfw * idf) ** 2
            total_seen += len(rows)
            if len(batch_sums) >= SPILL:
                with con:
                    con.executemany(
                        "INSERT INTO norm_tmp(chunk_id, s) VALUES(?, ?) ON CONFLICT(chunk_id) DO UPDATE SET s = s + excluded.s",
                        list(batch_sums.items()),
                    )
                batch_sums.clear()
            if total_seen % 2000000 == 0:
                write_status({'ok': True, 'state': 'indexing', 'phase': 'norms', 'postings_seen': total_seen, 'postings_total': posting_count, 'papers_total': total_papers, 'papers_done': done_papers, 'display_papers_done': done_papers})
            rows = cur.fetchmany(200000)
        if batch_sums:
            with con:
                con.executemany(
                    "INSERT INTO norm_tmp(chunk_id, s) VALUES(?, ?) ON CONFLICT(chunk_id) DO UPDATE SET s = s + excluded.s",
                    list(batch_sums.items()),
                )
            batch_sums.clear()
        # Write final norms back to chunk in manageable batches
        print("[index] Finalizing norms from temp table...")
        # Iterate temp table in chunks to cap memory
        last_id = 0
        B = 50000
        while True:
            rows = con.execute(
                "SELECT chunk_id, s FROM norm_tmp WHERE chunk_id > ? ORDER BY chunk_id LIMIT ?",
                (last_id, B),
            ).fetchall()
            if not rows:
                break
            with con:
                con.executemany(
                    "UPDATE chunk SET norm=? WHERE id=?",
                    [((math.sqrt(s) if s > 0 else 1.0), cid) for (cid, s) in rows],
                )
            last_id = rows[-1][0]
        with con:
            con.execute("DROP TABLE IF EXISTS norm_tmp")

    print("[index] Done.")
    print(f"  Papers:  {paper_count}")
    print(f"  Chunks:  {chunk_count}")
    print(f"  Tokens:  {con.execute('SELECT COUNT(1) FROM token').fetchone()[0]}")
    print(f"  Postings:{posting_count}")
    print(f"  DB path: {db_path}")
    # Build indexes at the end for speed
    print("[index] Creating indexes...")
    try:
        create_indexes(con)
    except Exception as e:
        print("[index] Index creation error:", e, file=sys.stderr)

    write_status({'ok': True, 'state': 'done', 'papers_total': len(papers), 'papers_done': len(papers), 'display_papers_done': len(papers), 'chunks': chunk_count, 'postings': posting_count})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge per-paper vectors into a global SQLite index")
    ap.add_argument("--scan", type=Path, required=True, help="Folder containing paper subfolders")
    ap.add_argument("--db", type=Path, required=True, help="SQLite database output path")
    ap.add_argument("--reset", action='store_true', help="Drop and recreate schema before ingest")
    ap.add_argument("--limit-papers", type=int, default=None, help="Limit number of papers during ingest (debug)")
    ap.add_argument("--incremental", action='store_true', help="Append only new papers; recompute norms in-place")
    ap.add_argument("--allowlist", type=Path, default=None, help="Optional file with paper folder names to include (one per line)")
    args = ap.parse_args(argv)

    if not args.scan.exists():
        print(f"Scan path not found: {args.scan}", file=sys.stderr)
        return 2
    args.db.parent.mkdir(parents=True, exist_ok=True)
    # If allowlist is provided, create a temporary scan dir with only allowed papers
    if args.allowlist and args.allowlist.exists():
        try:
            lines = args.allowlist.read_text(encoding='utf-8', errors='ignore').splitlines()
            allow = {ln.strip() for ln in lines if ln.strip()}
        except Exception:
            allow = set()
        import tempfile, os as _os
        with tempfile.TemporaryDirectory(prefix='psb_allow_') as tmpd:
            tdir = Path(tmpd)
            for p in sorted(args.scan.iterdir()):
                if p.is_dir() and p.name in allow:
                    try:
                        _os.symlink(str(p), str(tdir / p.name))
                    except FileExistsError:
                        pass
            # In incremental mode, force reset=False to preserve existing data
            reset_flag = bool(args.reset) and (not bool(args.incremental))
            merge_index(tdir, args.db, reset=reset_flag, limit_papers=args.limit_papers, incremental=bool(args.incremental))
    else:
        # In incremental mode, force reset=False to preserve existing data
        reset_flag = bool(args.reset) and (not bool(args.incremental))
        merge_index(args.scan, args.db, reset=reset_flag, limit_papers=args.limit_papers, incremental=bool(args.incremental))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
