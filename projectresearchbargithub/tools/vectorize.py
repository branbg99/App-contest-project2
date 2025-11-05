#!/usr/bin/env python3
from __future__ import annotations

"""
Math-aware vectorization for arXiv LaTeX sources.

Outputs under an --out directory per paper:
- chunks.jsonl: chunk records with kind, text, tokens
- vocab.json, df.json (mostly useful for debugging)
- vectors.jsonl: sparse TF–IDF vectors (not required by SQLite indexer)
"""

import argparse
import json
import math
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MATH_ENV_NAMES = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "alignat", "alignat*",
}


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    safe_makedirs(extract_dir)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, mode="r:*") as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tf, path=str(extract_dir))
        return
    out_name = archive_path.name
    if out_name.endswith('.gz'):
        out_name = out_name[:-3]
    out_path = extract_dir / out_name
    try:
        with gzip.open(archive_path, 'rb') as gzf:
            data = gzf.read()
    except Exception:
        data = archive_path.read_bytes()
    with open(out_path, 'wb') as out:
        out.write(data)
    header = data[:512].decode('utf-8', errors='ignore')
    if ('\\documentclass' in header or '\\begin{document}' in header) and out_path.suffix != '.tex':
        (extract_dir / 'main.tex').write_bytes(data)
    elif header.startswith('%PDF') and out_path.suffix != '.pdf':
        (extract_dir / 'paper.pdf').write_bytes(data)


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")


def strip_comments(tex: str) -> str:
    out_lines = []
    for line in tex.splitlines():
        i = 0
        result = []
        while i < len(line):
            ch = line[i]
            if ch == '%':
                if i > 0 and line[i-1] == '\\':
                    result.append('%'); i += 1; continue
                else:
                    break
            result.append(ch)
            i += 1
        out_lines.append(''.join(result))
    return '\n'.join(out_lines)


_MATH_PATTERN = re.compile(
    r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[\s\S]*?\$",
    re.MULTILINE,
)
_BEGIN_ENV = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_END_ENV = re.compile(r"\\end\{([A-Za-z*]+)\}")


def extract_math_blocks(tex: str) -> Tuple[str, List[str]]:
    math_blocks: List[str] = []
    def repl(m: re.Match) -> str:
        math = m.group(0)
        idx = len(math_blocks)
        math_blocks.append(math)
        return f" [MATH_{idx}] "
    tex2 = _MATH_PATTERN.sub(repl, tex)
    lines = tex2.splitlines()
    out_lines = []
    stack = []
    for line in lines:
        m = _BEGIN_ENV.search(line)
        if m and m.group(1) in MATH_ENV_NAMES:
            stack.append((m.group(1), [line])); continue
        if stack:
            stack[-1][1].append(line)
            m2 = _END_ENV.search(line)
            if m2 and m2.group(1) == stack[-1][0]:
                env_name, collected = stack.pop()
                block = '\n'.join(collected)
                idx = len(math_blocks)
                math_blocks.append(block)
                out_lines.append(f" [MATH_{idx}] ")
            continue
        out_lines.append(line)
    for _, collected in stack:
        out_lines.extend(collected)
    return '\n'.join(out_lines), math_blocks


_CMD_WITH_ARG = re.compile(r"\\([A-Za-z@]+)\\s*\\*?\\s*\\{([^}]*)\\}")
_CMD_ONLY = re.compile(r"\\([A-Za-z@]+)")


def latex_to_text_preserve_args(tex: str) -> str:
    DROP_CMDS_KEEP_NOTHING = {
        'cite', 'citep', 'citet', 'ref', 'eqref', 'label', 'footnote', 'url',
        'includegraphics', 'thanks', 'footnotemark', 'footnotetext', 'bibliography',
        'bibliographystyle', 'usepackage', 'documentclass', 'newcommand',
        'renewcommand', 'DeclareMathOperator', 'input', 'include', 'tableofcontents',
    }
    def keep_arg(m: re.Match) -> str:
        cmd = m.group(1)
        arg = m.group(2)
        if cmd in DROP_CMDS_KEEP_NOTHING:
            return ' '
        return ' ' + arg + ' '
    s = _CMD_WITH_ARG.sub(keep_arg, tex)
    s = _CMD_ONLY.sub(' ', s)
    s = s.replace('{', ' ').replace('}', ' ')
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize_text(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^0-9A-Za-z_\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF]+", " ", s)
    return [t for t in s.split() if t]


def _singularize(tok: str) -> str:
    t = tok.lower()
    if len(t) < 4:
        return t
    if t.endswith('ies') and len(t) > 4:
        return t[:-3] + 'y'
    if t.endswith('sses') or t.endswith('zzes'):
        return t[:-2]
    if t.endswith('es') and (t[-3] in 'sxz' or t[-4:-2] in ('ch', 'sh')):
        return t[:-2]
    if t.endswith('s') and not t.endswith('ss'):
        return t[:-1]
    return t


_CANON_MATH = {
    r'\\varepsilon': r'\\epsilon',
    r'\\varphi': r'\\phi',
    r'\\vartheta': r'\\theta',
    r'\\varsigma': r'\\sigma',
}

_WORD_TO_LATEX = {
    'forall': [r'\\forall'],
    'thereexists': [r'\\exists'], 'exists': [r'\\exists'],
    'implies': [r'\\implies'], 'iff': [r'\\iff'],
    'sum': [r'\\sum'], 'summation': [r'\\sum'], 'product': [r'\\prod'], 'coproduct': [r'\\coprod'],
    'integral': [r'\\int'], 'doubleintegral': [r'\\iint'], 'tripleintegral': [r'\\iiint'],
    'gradient': [r'\\nabla'], 'grad': [r'\\nabla'], 'partial': [r'\\partial'],
    'union': [r'\\cup'], 'intersection': [r'\\cap'],
    'subset': [r'\\subset'], 'subseteq': [r'\\subseteq'], 'supset': [r'\\supset'], 'supseteq': [r'\\supseteq'],
    'infinity': [r'\\infty'], 'emptyset': [r'\\emptyset'], 'nabla': [r'\\nabla'], 'in': [r'\\in'], 'notin': [r'\\notin'],
    'alpha': [r'\\alpha'], 'beta': [r'\\beta'], 'gamma': [r'\\gamma'], 'delta': [r'\\delta'],
    'epsilon': [r'\\epsilon'], 'zeta': [r'\\zeta'], 'eta': [r'\\eta'], 'theta': [r'\\theta'],
    'iota': [r'\\iota'], 'kappa': [r'\\kappa'], 'lambda': [r'\\lambda'], 'mu': [r'\\mu'],
    'nu': [r'\\nu'], 'xi': [r'\\xi'], 'pi': [r'\\pi'], 'rho': [r'\\rho'],
    'sigma': [r'\\sigma'], 'tau': [r'\\tau'], 'upsilon': [r'\\upsilon'], 'phi': [r'\\phi'],
    'chi': [r'\\chi'], 'psi': [r'\\psi'], 'omega': [r'\\omega'],
}


def _expand_and_normalize_tokens(tokens: List[str], raw_text: str | None = None) -> List[str]:
    out: List[str] = []
    seen_once: set[str] = set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith('math_') and t[5:].isdigit():
            i += 1; continue
        if t.startswith('\\') and t in _CANON_MATH:
            t = _CANON_MATH[t]
        out.append(t)
        i += 1
    extras: set[str] = set()
    for t in out:
        if not t.startswith('\\'):
            s = _singularize(t)
            if s and s != t:
                extras.add(s)
        for la in _WORD_TO_LATEX.get(t, []):
            extras.add(la)
    for a, b, la in [('-', '>', r'\\to'), ('=', '>', r'\\implies'), ('<', '-', r'\\leftarrow'),
                     ('<', '>', r'\\leftrightarrow'), ('<', '=', r'\\leq'), ('>', '=', r'\\geq'), ('!', '=', r'\\neq')]:
        for idx in range(len(out) - 1):
            if out[idx] == a and out[idx + 1] == b:
                extras.add(la)
    if raw_text:
        rt = raw_text.lower()
        if 'for all' in rt:
            extras.add(r'\\forall')
        if 'there exists' in rt:
            extras.add(r'\\exists')
        if 'if and only if' in rt:
            extras.add(r'\\iff')
    for la in sorted(extras):
        if la not in seen_once:
            out.append(la)
            seen_once.add(la)
    return out


def tokenize_math(latex_math: str) -> List[str]:
    s = latex_math
    if s.startswith('$$') and s.endswith('$$'):
        s = s[2:-2]
    elif s.startswith('$') and s.endswith('$'):
        s = s[1:-1]
    elif s.startswith('\\[') and s.endswith('\\]'):
        s = s[2:-2]
    elif s.startswith('\\(') and s.endswith('\\)'):
        s = s[2:-2]
    s = re.sub(r"\\begin\{[^}]+\}", " ", s)
    s = re.sub(r"\\end\{[^}]+\}", " ", s)

    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            if j == i + 1:
                tokens.append('\\' + s[j:j+1])
                i = j + 1
            else:
                tokens.append('\\' + s[i+1:j].lower())
                i = j
            continue
        if ch in '^_{}[]()=+-*/:,.;<>|!':
            tokens.append(ch); i += 1; continue
        if ch.isdigit():
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j]); i = j; continue
        if ch.isalpha():
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "'"):
                j += 1
            tokens.append(s[i:j].lower()); i = j; continue
        i += 1
    return tokens


def chunk_latex(tex: str, math_blocks: List[str]) -> List[dict]:
    chunks: List[dict] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", tex) if p.strip()]
    cid = 0
    for p in paras:
        clean = latex_to_text_preserve_args(p)
        if clean:
            tokens = tokenize_text(clean)
            tokens = _expand_and_normalize_tokens(tokens, raw_text=clean)
            chunks.append({'id': f'chunk_{cid}', 'kind': 'paragraph', 'text': clean, 'math': [], 'tokens': tokens})
            cid += 1
    for m in math_blocks:
        tokens = tokenize_math(m)
        tokens = _expand_and_normalize_tokens(tokens)
        if tokens:
            chunks.append({'id': f'chunk_{cid}', 'kind': 'equation', 'text': m, 'math': tokens, 'tokens': tokens})
            cid += 1
    return chunks


def build_vocab_and_tfidf(chunks: List[dict]) -> tuple[dict[str, int], dict[int, int], List[dict[int, float]]]:
    vocab: Dict[str, int] = {}
    df: Dict[int, int] = defaultdict(int)
    doc_tf: List[Counter] = []
    for ch in chunks:
        tf = Counter()
        seen_in_doc = set()
        for tok in ch['tokens']:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            idx = vocab[tok]
            tf[idx] += 1
            seen_in_doc.add(idx)
        for idx in seen_in_doc:
            df[idx] += 1
        doc_tf.append(tf)
    n_docs = max(1, len(chunks))
    vectors: List[dict[int, float]] = []
    for tf in doc_tf:
        tf_log = {i: 1.0 + math.log(c) for i, c in tf.items()}
        vec = {}
        for i, t in tf_log.items():
            idf = math.log((1 + n_docs) / (1 + df.get(i, 0))) + 1.0
            vec[i] = t * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vec = {i: v / norm for i, v in vec.items()}
        vectors.append(vec)
    return vocab, df, vectors


def find_tex_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*.tex') if p.is_file()]


def choose_main_tex(tex_files: List[Path]) -> List[Path]:
    scored: List[tuple[int, int, Path]] = []
    for p in tex_files:
        try:
            s = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            s = p.read_text(errors='ignore')
        score = 0
        if '\\begin{document}' in s:
            score += 10
        score += min(5, s.count('\\section'))
        scored.append((score, len(s), p))
    scored.sort(reverse=True)
    ordered = [p for _, __, p in scored]
    return ordered[:5] if ordered else []


def process_archive(archive_path: Path, out_dir: Path) -> None:
    safe_makedirs(out_dir)
    with tempfile.TemporaryDirectory(prefix='arxiv_extract_') as tmpd:
        tmpdir = Path(tmpd)
        extract_archive(archive_path, tmpdir)
        tex_files = find_tex_files(tmpdir)
        if not tex_files:
            print(f"[WARN] No .tex files found in {archive_path.name}; skipping.")
            return
        main_files = choose_main_tex(tex_files)
        combined = []
        for p in main_files if main_files else tex_files:
            combined.append(read_text_file(p))
        tex = "\n\n".join(combined)
        tex = strip_comments(tex)
        tex_no_math, math_blocks = extract_math_blocks(tex)
        chunks = chunk_latex(tex_no_math, math_blocks)
        vocab, df, vectors = build_vocab_and_tfidf(chunks)

        (out_dir / 'chunks.jsonl').write_text('\n'.join(json.dumps(ch) for ch in chunks), encoding='utf-8')
        json.dump(vocab, (out_dir / 'vocab.json').open('w', encoding='utf-8'), ensure_ascii=False)
        json.dump({int(k): v for k, v in df.items()}, (out_dir / 'df.json').open('w', encoding='utf-8'), ensure_ascii=False)
        with (out_dir / 'vectors.jsonl').open('w', encoding='utf-8') as f:
            for vec in vectors:
                f.write(json.dumps({str(i): v for i, v in vec.items()}))
                f.write('\n')
        print(f"[OK] Wrote vectors for {len(chunks)} chunks to {out_dir}")


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vectorize arXiv LaTeX sources with math-aware tokenization")
    ap.add_argument('archive', type=Path, help='Path to .tar.gz of arXiv source')
    ap.add_argument('--out', type=Path, required=True, help='Output directory')
    args = ap.parse_args(argv)

    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")
    process_archive(args.archive, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())

