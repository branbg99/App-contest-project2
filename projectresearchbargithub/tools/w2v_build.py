#!/usr/bin/env python3
from __future__ import annotations

"""
Train a Word2Vec model on tokenized chunks from the existing SQLite index.

Outputs to data/w2v/:
- model.kv: Gensim KeyedVectors in native format
- model.json: metadata (vector_size, window, min_count, sg)

Usage examples:
  python3 -m ProjectSearchBar.tools.w2v_build --vector-size 200 --window 8 --min-count 3 --sg 1

Notes:
- Requires gensim and numpy. If unavailable, the script will explain and exit.
- Tokens are produced by the shared tokenizer on each chunk's text to align with search.
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List

from ProjectSearchBar import config
from ProjectSearchBar import tokenize as tok


def iter_chunk_tokens(con: sqlite3.Connection, limit: int | None = None) -> Iterable[List[str]]:
    cur = con.execute('SELECT id, kind, text FROM chunk ORDER BY id')
    count = 0
    for cid, kind, text in cur.fetchall():
        s = text or ''
        # Tokenize consistently with server query tokenization
        if str(kind or '').lower() == 'equation':
            tokens = tok.tokenize_math(tok.strip_math_delims(s))
        else:
            tokens = tok.tokenize_text(s)
        tokens = [t for t in tokens if t]
        if tokens:
            yield tokens
            count += 1
            if limit is not None and count >= limit:
                return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Train Word2Vec on indexed chunks')
    ap.add_argument('--vector-size', type=int, default=200, help='Embedding size (default 200)')
    ap.add_argument('--window', type=int, default=8, help='Context window (default 8)')
    ap.add_argument('--min-count', type=int, default=3, help='Min token count (default 3)')
    ap.add_argument('--sg', type=int, default=1, help='1=skip-gram, 0=CBOW (default 1)')
    ap.add_argument('--epochs', type=int, default=5, help='Training epochs (default 5)')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of chunks to train on (debug)')
    ap.add_argument('--out', type=Path, default=config.DATA_DIR / 'w2v', help='Output directory')
    args = ap.parse_args(argv)

    try:
        import gensim  # type: ignore
        from gensim.models import Word2Vec  # type: ignore
    except Exception:
        print('ERROR: gensim is required. Install: pip install gensim')
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = config.DB_PATH
    if not db_path.exists():
        print(f'Index DB not found: {db_path}. Build the index first.')
        return 2

    con = sqlite3.connect(str(db_path))
    sentences = iter_chunk_tokens(con, args.limit)

    print('[w2v] Building vocabulary...')
    model = Word2Vec(
        vector_size=int(args.vector_size),
        window=int(args.window),
        min_count=int(args.min_count),
        sg=int(args.sg),
        workers=max(1, (os.cpu_count() or 2) - 1),
    )
    s_list = list(sentences)
    if not s_list:
        print('[w2v] No sentences found. Is the index populated?')
        return 2
    model.build_vocab(s_list)
    print(f"[w2v] Vocab size: {len(model.wv)}")
    print('[w2v] Training...')
    model.train(s_list, total_examples=len(s_list), epochs=int(args.epochs))

    kv_path = out_dir / 'model.kv'
    print(f'[w2v] Saving KeyedVectors to {kv_path}')
    model.wv.save(str(kv_path))
    with (out_dir / 'model.json').open('w', encoding='utf-8') as f:
        json.dump({
            'vector_size': int(args.vector_size),
            'window': int(args.window),
            'min_count': int(args.min_count),
            'sg': int(args.sg),
            'epochs': int(args.epochs),
            'sentences': int(len(s_list)),
        }, f)
    print('[w2v] Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

