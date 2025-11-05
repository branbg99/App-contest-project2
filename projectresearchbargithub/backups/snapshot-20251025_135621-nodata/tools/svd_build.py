#!/usr/bin/env python3
from __future__ import annotations

"""
Build truncated SVD (LSI) over TF–IDF(chunk, token) using the existing index.

Artifacts written to data/svd/:
- model.json: metadata (k, n_terms, fit_sample, token_map sizes)
- idf.npy: float32 idf per term column
- components.npy: float32 shape (k, n_terms)
- svd.sqlite: table chunk_svd(id INTEGER PRIMARY KEY, v BLOB) storing float32[k]

Usage (example):
  python3 -m ProjectSearchBar.tools.svd_build \
    --k 128 --fit-sample 100000 --batch-size 2000

Notes:
- Requires scikit-learn and numpy. If unavailable, script will explain and exit.
- Operates in batches; does not materialize full matrices.
"""

import argparse
import json
import math
import os
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from scipy.sparse import csr_matrix  # type: ignore
except Exception as e:
    TruncatedSVD = None  # type: ignore
    csr_matrix = None  # type: ignore

# Local imports
from ProjectSearchBar import config


def load_terms(con: sqlite3.Connection) -> Tuple[np.ndarray, Dict[int, int]]:
    rows = con.execute("SELECT id, df FROM token").fetchall()
    if not rows:
        raise RuntimeError("No tokens in index; build the index first.")
    token_ids = [int(r[0]) for r in rows]
    dfs = [int(r[1] or 0) for r in rows]
    # Build continuous column index mapping for tokens
    id_to_col: Dict[int, int] = {tid: i for i, tid in enumerate(token_ids)}
    # Compute idf using chunk count (N)
    N = int(con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1)
    idf = np.zeros(len(token_ids), dtype=np.float32)
    for i, df in enumerate(dfs):
        idf[i] = math.log((1.0 + N) / (1.0 + float(df))) + 1.0
    return idf, id_to_col


def sample_chunk_ids(con: sqlite3.Connection, fit_sample: int) -> List[int]:
    total = int(con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 0)
    ids = [int(r[0]) for r in con.execute('SELECT id FROM chunk').fetchall()]
    if fit_sample <= 0 or fit_sample >= len(ids):
        return ids
    random.shuffle(ids)
    return ids[:fit_sample]


def batch_csr_for_chunks(con: sqlite3.Connection, chunk_ids: List[int], id_to_col: Dict[int, int], idf: np.ndarray) -> csr_matrix:
    # Build CSR from postings for the given chunk_ids
    # SELECT token_id, tf for these chunks
    data: List[float] = []
    rows: List[int] = []
    cols: List[int] = []
    # To avoid too many variables, fetch per chunk
    for ri, cid in enumerate(chunk_ids):
        cur = con.execute('SELECT token_id, tf FROM posting WHERE chunk_id=?', (int(cid),))
        for tid, tf in cur.fetchall():
            col = id_to_col.get(int(tid))
            if col is None:
                continue
            tfw = 1.0 + math.log(float(tf))
            val = tfw * float(idf[col])
            rows.append(ri)
            cols.append(col)
            data.append(val)
    if csr_matrix is None:
        raise RuntimeError("scipy.sparse is required for SVD build")
    mat = csr_matrix((np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(len(chunk_ids), len(idf)))
    # L2-normalize rows to standardize scale before SVD
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    inv = 1.0 / norms
    mat = mat.multiply(inv[:, None])
    return mat


def ensure_out_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def save_blob_vec(cur: sqlite3.Cursor, cid: int, vec: np.ndarray) -> None:
    cur.execute('INSERT OR REPLACE INTO chunk_svd(id, v) VALUES(?, ?)', (int(cid), memoryview(vec.astype(np.float32).tobytes())))


def build_main(k: int, fit_sample: int, batch_size: int, out_dir: Path) -> int:
    if TruncatedSVD is None or csr_matrix is None:
        print("ERROR: scikit-learn and scipy are required. Install: pip install scikit-learn scipy")
        return 2
    db = config.DB_PATH
    if not db.exists():
        print(f"Index DB not found: {db}. Build the index first.")
        return 2
    ensure_out_dirs(out_dir)
    con = sqlite3.connect(str(db))
    print("[svd] Loading terms and idf...")
    idf, id_to_col = load_terms(con)
    n_terms = idf.shape[0]
    print(f"[svd] Terms: {n_terms}")

    # Sample for fit
    print(f"[svd] Sampling chunks for fit (fit_sample={fit_sample})...")
    fit_ids = sample_chunk_ids(con, fit_sample)
    if not fit_ids:
        print("[svd] No chunks to fit on.")
        return 2
    # Build fit matrix in batches
    print("[svd] Building fit matrix (batched)...")
    fit_mat_batches: List[csr_matrix] = []
    for i in range(0, len(fit_ids), batch_size):
        sub = fit_ids[i:i+batch_size]
        mat = batch_csr_for_chunks(con, sub, id_to_col, idf)
        fit_mat_batches.append(mat)
        print(f"  - batch {i//batch_size+1}/{(len(fit_ids)+batch_size-1)//batch_size} rows={mat.shape[0]}")
    # stack rows
    from scipy.sparse import vstack  # type: ignore
    fit_mat = vstack(fit_mat_batches)
    print(f"[svd] Fit matrix: {fit_mat.shape}")
    print(f"[svd] Fitting TruncatedSVD(k={k})...")
    svd = TruncatedSVD(n_components=int(k), algorithm='randomized', random_state=42)
    svd.fit(fit_mat)
    # Save model artifacts
    import numpy.lib.format as npformat  # type: ignore
    (out_dir / 'idf.npy').write_bytes(idf.tobytes())
    np.save(out_dir / 'idf.npy', idf.astype(np.float32))
    np.save(out_dir / 'components.npy', svd.components_.astype(np.float32))
    # Token map as {token_id: col}
    token_map_path = out_dir / 'token_map.json'
    with token_map_path.open('w', encoding='utf-8') as f:
        json.dump({str(tid): int(col) for tid, col in id_to_col.items()}, f)
    with (out_dir / 'model.json').open('w', encoding='utf-8') as f:
        json.dump({'k': int(k), 'n_terms': int(n_terms), 'fit_rows': int(fit_mat.shape[0])}, f)

    # Prepare embeddings store
    svd_db = sqlite3.connect(str(out_dir / 'svd.sqlite'))
    svd_db.execute('CREATE TABLE IF NOT EXISTS chunk_svd (id INTEGER PRIMARY KEY, v BLOB NOT NULL)')
    svd_db.commit()
    # Transform all chunks in batches
    all_ids = [int(r[0]) for r in con.execute('SELECT id FROM chunk ORDER BY id')]
    print(f"[svd] Transforming {len(all_ids)} chunks to k={k} embeddings...")
    cur = svd_db.cursor()
    for i in range(0, len(all_ids), batch_size):
        sub = all_ids[i:i+batch_size]
        mat = batch_csr_for_chunks(con, sub, id_to_col, idf)
        emb = svd.transform(mat).astype(np.float32)
        # L2 norm
        norms = np.linalg.norm(emb, axis=1)
        norms[norms == 0] = 1.0
        emb = (emb.T / norms).T
        for row_idx, cid in enumerate(sub):
            save_blob_vec(cur, int(cid), emb[row_idx])
        svd_db.commit()
        print(f"  - embedded chunks {i+1}..{i+len(sub)}")
    svd_db.close()
    con.close()
    print("[svd] Done.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build TruncatedSVD (LSI) embeddings for chunks')
    ap.add_argument('--k', type=int, default=128, help='SVD dimensions (default 128)')
    ap.add_argument('--fit-sample', type=int, default=100000, help='Rows to sample for fitting SVD')
    ap.add_argument('--batch-size', type=int, default=2000, help='Batch size for building/transforms')
    ap.add_argument('--out', type=Path, default=config.DATA_DIR / 'svd', help='Output directory')
    args = ap.parse_args(argv)
    return build_main(args.k, args.fit_sample, args.batch_size, args.out)


if __name__ == '__main__':
    raise SystemExit(main())

