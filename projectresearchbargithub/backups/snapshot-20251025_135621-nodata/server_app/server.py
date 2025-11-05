#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

from ProjectSearchBar import config
from ProjectSearchBar import tokenize as tok
import time
import stat
import urllib.request
import urllib.error
import ssl
from typing import Set

# Basic English stopwords to de-noise long prompts
STOPWORDS: Set[str] = set(
    s.lower() for s in (
        'the','and','for','with','that','this','from','into','onto','over','under','between','among','about','above','below',
        'is','are','was','were','be','been','being','of','in','to','by','on','as','at','or','an','a','it','its','their','there',
        'we','you','he','she','they','them','his','her','our','ours','your','yours','my','mine','me','i','us','one','two','three',
        'using','use','used','via','such','these','those','which','who','whom','whose','both','either','neither','each','every',
        'can','could','may','might','must','shall','should','will','would','do','does','did','done','have','has','had','having',
        'also','thus','hence','therefore','etc','eg','ie','vs','et','al'
    )
)

# Unicode → LaTeX token hints for PDF-copied math
UNICODE_TO_LATEX: dict[str, str | tuple[str, ...]] = {
    '±': r'\\pm',
    '∈': r'\\in',
    '∉': r'\\notin',
    '⊆': r'\\subseteq',
    '⊂': r'\\subset',
    '⊇': r'\\supseteq',
    '⊃': r'\\supset',
    '≤': r'\\leq',
    '≥': r'\\geq',
    '≠': r'\\neq',
    '→': r'\\to',
    '←': r'\\leftarrow',
    '↔': r'\\leftrightarrow',
    '⇒': r'\\implies',
    '⇔': r'\\iff',
    '⋅': r'\\cdot',
    '·': r'\\cdot',
    '∗': r'\\ast',
    '‖': r'\\Vert',
    '⟨': r'\\langle',
    '⟩': r'\\rangle',
    '⊥': r'\\perp',
    '∇': r'\\nabla',
    '∂': r'\\partial',
    '∪': r'\\cup',
    '∩': r'\\cap',
    '∞': r'\\infty',
    '∅': r'\\emptyset',
}

# Greek Unicode → LaTeX command
GREEK_TO_LATEX: dict[str, str] = {
    'α': r'\\alpha','β': r'\\beta','γ': r'\\gamma','δ': r'\\delta','ε': r'\\epsilon','ζ': r'\\zeta','η': r'\\eta','θ': r'\\theta','ι': r'\\iota','κ': r'\\kappa','λ': r'\\lambda','μ': r'\\mu','ν': r'\\nu','ξ': r'\\xi','ο': 'o','π': r'\\pi','ρ': r'\\rho','σ': r'\\sigma','τ': r'\\tau','υ': r'\\upsilon','φ': r'\\phi','χ': r'\\chi','ψ': r'\\psi','ω': r'\\omega',
    'Α': r'\\Alpha','Β': r'\\Beta','Γ': r'\\Gamma','Δ': r'\\Delta','Ε': r'\\Epsilon','Ζ': r'\\Zeta','Η': r'\\Eta','Θ': r'\\Theta','Ι': r'\\Iota','Κ': r'\\Kappa','Λ': r'\\Lambda','Μ': r'\\Mu','Ν': r'\\Nu','Ξ': r'\\Xi','Ο': 'O','Π': r'\\Pi','Ρ': r'\\Rho','Σ': r'\\Sigma','Τ': r'\\Tau','Υ': r'\\Upsilon','Φ': r'\\Phi','Χ': r'\\Chi','Ψ': r'\\Psi','Ω': r'\\Omega',
}

# Blackboard bold common sets
BLACKBOARD_TO_LATEX: dict[str, tuple[str, str]] = {
    'ℝ': (r'\\mathbb', 'r'),
    'ℂ': (r'\\mathbb', 'c'),
    'ℤ': (r'\\mathbb', 'z'),
    'ℚ': (r'\\mathbb', 'q'),
    'ℕ': (r'\\mathbb', 'n'),
    'ℙ': (r'\\mathbb', 'p'),
}


def _singularize(tok: str) -> str:
    t = (tok or '').lower()
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


def tokenize_text(s: str) -> list[str]:
    return tok.tokenize_text(s)


def strip_math_delims(s: str) -> str:
    return tok.strip_math_delims(s)


def tokenize_math(latex: str) -> list[str]:
    return tok.tokenize_math(latex)


def tokenize_query(q: str) -> list[str]:
    return tok.tokenize_query(q)


class SearchRequestHandler(SimpleHTTPRequestHandler):
    _db: sqlite3.Connection | None = None
    _db_lock = threading.Lock()
    _db_sig = None  # (mtime_ns, size)
    _build_active = False
    _build_lock = threading.Lock()
    _build_id: str | None = None
    # Ephemeral chat sessions: in-memory only, cleared on server restart
    # sessions[session_id] = {
    #   'created': ts,
    #   'papers': [ {'id': 'arxiv_id', 'mode': 'latex'|'vectors'} ],
    #   'default_token_budget': int,
    # }
    _chat_sessions: dict[str, dict] = {}
    _chat_lock = threading.Lock()
    # Optional SVD re-ranker (lazy-loaded)
    _svd_loaded = False
    _svd_components = None  # numpy.ndarray[k, n_terms]
    _svd_token_map: dict[int,int] | None = None
    _svd_k: int | None = None
    _svd_db: sqlite3.Connection | None = None  # data/svd/svd.sqlite

    @classmethod
    def db(cls) -> sqlite3.Connection:
        if cls._db is None:
            with cls._db_lock:
                if cls._db is None:
                    # Open read-optimized connection with busy timeout and WAL
                    cls._db = sqlite3.connect(
                        str(config.DB_PATH), timeout=10.0, check_same_thread=False
                    )
                    try:
                        cls._db.execute('PRAGMA journal_mode=WAL')
                        cls._db.execute('PRAGMA synchronous=NORMAL')
                        cls._db.execute('PRAGMA busy_timeout=10000')
                        # Boost read performance
                        cls._db.execute('PRAGMA temp_store=MEMORY')
                        try:
                            # ~100MB cache; negative = KB units
                            cls._db.execute('PRAGMA cache_size=-100000')
                        except Exception:
                            pass
                        try:
                            # 1 GiB mmap if available
                            cls._db.execute('PRAGMA mmap_size=1073741824')
                        except Exception:
                            pass
                    except Exception:
                        pass
        return cls._db

    @classmethod
    def _svd_try_load(cls) -> None:
        if cls._svd_loaded:
            return
        try:
            import json as _json
            import numpy as _np  # type: ignore
            svd_root = config.DATA_DIR / 'svd'
            model = svd_root / 'model.json'
            comp = svd_root / 'components.npy'
            tmap = svd_root / 'token_map.json'
            svd_sql = svd_root / 'svd.sqlite'
            if not (model.exists() and comp.exists() and tmap.exists() and svd_sql.exists()):
                cls._svd_loaded = True
                return
            meta = _json.loads(model.read_text(encoding='utf-8'))
            raw = _json.loads(tmap.read_text(encoding='utf-8'))
            token_map = {int(k): int(v) for (k, v) in raw.items()}
            comps = _np.load(comp)
            cls._svd_components = comps
            cls._svd_token_map = token_map
            cls._svd_k = int(meta.get('k') or comps.shape[0])
            cls._svd_db = sqlite3.connect(str(svd_sql), timeout=10.0, check_same_thread=False)
        except Exception:
            pass
        finally:
            cls._svd_loaded = True

    @classmethod
    def close_db(cls) -> None:
        with cls._db_lock:
            try:
                if cls._db is not None:
                    try:
                        cls._db.close()
                    except Exception:
                        pass
                    finally:
                        cls._db = None
            except Exception:
                cls._db = None

    @classmethod
    def _db_signature(cls):
        try:
            st = os.stat(config.DB_PATH)
            return (st.st_mtime_ns, st.st_size)
        except Exception:
            return None

    @classmethod
    def refresh_db_if_changed(cls) -> None:
        sig = cls._db_signature()
        if sig != cls._db_sig:
            # Underlying DB file changed (or was created). Reopen to pick up new data.
            cls.close_db()
            cls._db_sig = sig

    def translate_path(self, path):
        # Serve static files from active UI root; on miss, try fallback UI.
        # For unknown /api/* paths, return index.html as a harmless fallback
        if path.startswith('/api/'):
            return str(config.UI_PUBLIC / 'index.html')
        raw = path.split('?', 1)[0].split('#', 1)[0]
        raw = os.path.normpath(unquote(raw))
        parts = [p for p in raw.split('/') if p and p != '..']
        # Try active UI
        p1 = config.UI_PUBLIC
        for seg in parts:
            p1 = p1 / seg
        try:
            if p1.exists():
                return str(p1)
        except Exception:
            pass
        # Try fallback UI if available
        try:
            p2 = getattr(config, 'UI_PUBLIC_FALLBACK', None)
        except Exception:
            p2 = None
        if p2 is not None:
            for seg in parts:
                p2 = p2 / seg
            try:
                if p2.exists():
                    return str(p2)
            except Exception:
                pass
        # Default to path under active UI even if it may not exist
        return str(p1)

    def _read_json(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b'{}'
        try:
            return json.loads(raw.decode('utf-8') or '{}')
        except Exception:
            return {}

    def _write_json(self, obj, code=200):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # Add no-cache headers for all responses (HTML/JS/CSS), to avoid stale UI assets
    def end_headers(self):
        try:
            # Prevent caching of static UI files and API responses
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        except Exception:
            pass
        try:
            super().end_headers()
        except Exception:
            # Fallback: ignore header errors
            pass

    def do_GET(self):
        try:
            # Quietly satisfy favicon requests to avoid 404 noise/broken pipe
            if self.path == '/favicon.ico':
                try:
                    icon_path = config.UI_PUBLIC / 'favicon.ico'
                    if icon_path.exists():
                        return super().do_GET()
                    # No icon file; respond with 204 No Content
                    self.send_response(204)
                    self.end_headers()
                    return
                except Exception:
                    self.send_response(204)
                    self.end_headers()
                    return
            if self.path.startswith('/api/paper/meta'):
                return self.api_paper_meta()
            if self.path.startswith('/api/diagnose'):
                return self.api_diagnose()
            if self.path.startswith('/api/papers'):
                return self.api_papers()
            if self.path.startswith('/api/index/status'):
                return self.api_index_status()
            if self.path.startswith('/api/llm/settings'):
                return self.api_llm_settings_get()
            if self.path.startswith('/api/search/settings'):
                return self.api_search_settings_get()
            if self.path.startswith('/api/ui/info'):
                return self.api_ui_info()
            if self.path.startswith('/api/debug/tokenize'):
                return self.api_debug_tokenize()
            if self.path.startswith('/api/chat/context'):
                return self.api_chat_context()
            if self.path.startswith('/api/chat/state'):
                return self.api_chat_state()
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.close_connection = True
            except Exception:
                pass
            return

    def do_POST(self):
        try:
            if self.path.startswith('/api/search'):
                return self.api_search()
            if self.path.startswith('/api/index/build'):
                return self.api_index_build()
            # launch endpoints removed
            if self.path.startswith('/api/llm/settings'):
                return self.api_llm_settings_post()
            if self.path.startswith('/api/search/settings'):
                return self.api_search_settings_post()
            if self.path.startswith('/api/llm/test'):
                return self.api_llm_test()
            if self.path.startswith('/api/ask'):
                return self.api_ask()
            # Chat session API
            if self.path.startswith('/api/chat/start'):
                return self.api_chat_start()
            if self.path.startswith('/api/chat/add_paper'):
                return self.api_chat_add_paper()
            if self.path.startswith('/api/chat/remove_paper'):
                return self.api_chat_remove_paper()
            if self.path.startswith('/api/chat/clear'):
                return self.api_chat_clear()
            if self.path.startswith('/api/chat/message'):
                return self.api_chat_message()
            # /api/ask_read removed during redesign
            return super().do_POST()
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.close_connection = True
            except Exception:
                pass
            return

    # --- API endpoints ---
    def api_paper_meta(self):
        try:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query or '')
            ids_raw = (qs.get('ids') or [''])[0]
            ids = [s.strip() for s in (ids_raw or '').split(',') if s.strip()]
            if not ids:
                return self._write_json({'ok': True, 'meta': {}}, 200)
            meta_dir = config.DATA_DIR / 'meta'
            try:
                meta_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            def cache_path(arxid: str) -> Path:
                safe = arxid.replace('/', '_')
                return meta_dir / f"{safe}.json"

            import json as _json
            meta: dict[str, dict] = {}
            missing: list[str] = []
            for pid in ids:
                cp = cache_path(pid)
                try:
                    if cp.exists():
                        m = _json.loads(cp.read_text(encoding='utf-8'))
                        if isinstance(m, dict) and m.get('arxiv_id'):
                            meta[pid] = m
                            continue
                except Exception:
                    pass
                missing.append(pid)

            # Try arXiv API in batches if any are missing
            if missing:
                try:
                    self._fetch_and_cache_arxiv_meta(missing, cache_path, meta)
                except Exception:
                    # Network or parse failure: attempt offline fallback
                    pass
            # Offline fallback for any still missing
            still = [pid for pid in ids if pid not in meta]
            if still:
                for pid in still:
                    fm = self._offline_fallback_meta(pid)
                    if fm:
                        try:
                            # cache fallback too
                            cp = cache_path(pid)
                            cp.write_text(_json.dumps(fm, ensure_ascii=False), encoding='utf-8')
                        except Exception:
                            pass
                        meta[pid] = fm

            return self._write_json({'ok': True, 'meta': meta}, 200)
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)}, 200)

    def _fetch_and_cache_arxiv_meta(self, ids: list[str], cache_path_fn, out_meta: dict):
        # Batch request to arXiv Atom API; network may be blocked — handle errors upstream
        import urllib.parse
        import urllib.request
        import urllib.error
        import ssl
        import time
        from xml.etree import ElementTree as ET
        if not ids:
            return
        # arXiv allows comma-separated id_list
        base = 'http://export.arxiv.org/api/query'
        # split in batches of 20 to keep URLs manageable
        B = 20
        for i in range(0, len(ids), B):
            chunk = ids[i:i+B]
            q = urllib.parse.urlencode({'id_list': ','.join(chunk)})
            url = f"{base}?{q}"
            req = urllib.request.Request(url, headers={'User-Agent': 'ProjectSearchBar/1.0'})
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=7, context=ctx) as resp:
                data = resp.read()
            # Parse Atom
            root = ET.fromstring(data)
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            for entry in root.findall('atom:entry', ns):
                try:
                    # id like: http://arxiv.org/abs/1012.1190v1
                    rid = (entry.findtext('atom:id', default='', namespaces=ns) or '')
                    arxid = rid.rsplit('/', 1)[-1]
                    # Strip version suffix to normalize key
                    if 'v' in arxid and arxid.split('v')[-1].isdigit():
                        arxid_key = arxid.split('v')[0]
                    else:
                        arxid_key = arxid
                    title = (entry.findtext('atom:title', default='', namespaces=ns) or '').strip()
                    summary = (entry.findtext('atom:summary', default='', namespaces=ns) or '').strip()
                    # Authors
                    authors = []
                    for a in entry.findall('atom:author', ns):
                        nm = a.findtext('atom:name', default='', namespaces=ns) or ''
                        nm = nm.strip()
                        if nm:
                            authors.append(nm)
                    # Categories
                    subjects = []
                    for c in entry.findall('atom:category', ns):
                        term = c.attrib.get('term')
                        if term:
                            subjects.append(term)
                    # DOI (optional)
                    doi = None
                    doi_el = entry.find('arxiv:doi', ns)
                    if doi_el is not None and (doi_el.text or '').strip():
                        doi = doi_el.text.strip()
                    m = {
                        'arxiv_id': arxid_key,
                        'title': title,
                        'authors': authors,
                        'subjects': subjects,
                        'abstract': summary,
                        'doi': doi,
                        'source': 'arxiv',
                        'ts': int(time.time()),
                    }
                    out_meta[arxid_key] = m
                    # Cache
                    try:
                        cp = cache_path_fn(arxid_key)
                        cp.write_text(__import__('json').dumps(m, ensure_ascii=False), encoding='utf-8')
                    except Exception:
                        pass
                except Exception:
                    continue

    def _offline_fallback_meta(self, arxid: str) -> dict | None:
        # Attempt a very rough fallback using local vectors
        import json as _json
        # candidates under data/vectors2 or data/vectors
        v2 = config.DATA_DIR / 'vectors2' / arxid
        v1 = config.DATA_DIR / 'vectors' / arxid
        for root in (v2, v1):
            try:
                cj = root / 'chunks.jsonl'
                if cj.exists():
                    # Take first paragraph chunk as abstract-ish
                    first_text = None
                    with cj.open('r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            try:
                                o = _json.loads(line)
                                if isinstance(o, dict) and (o.get('kind') == 'paragraph'):
                                    first_text = (o.get('text') or '').strip()
                                    break
                            except Exception:
                                continue
                    if first_text:
                        return {
                            'arxiv_id': arxid,
                            'title': arxid,
                            'authors': [],
                            'subjects': [],
                            'abstract': first_text[:1200],
                            'doi': None,
                            'source': 'offline',
                            'ts': 0,
                        }
            except Exception:
                continue
        return None
    def api_debug_tokenize(self):
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query or '')
            q = (qs.get('q') or [''])[0]
            res = tok.debug_tokenize(q or '')
            return self._write_json({'ok': True, 'q': q, 'tokens': res}, 200)
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)}, 200)
    def api_diagnose(self):
        ok = config.DB_PATH.exists()
        stats = {}
        if ok:
            try:
                # Ensure we see the latest DB after any swap
                self.refresh_db_if_changed()
                con = self.db()
                stats = {
                    'tokens': con.execute('SELECT COUNT(1) FROM token').fetchone()[0],
                    'papers': con.execute('SELECT COUNT(1) FROM paper').fetchone()[0],
                    'chunks': con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0],
                    'postings': con.execute('SELECT COUNT(1) FROM posting').fetchone()[0],
                }
            except sqlite3.OperationalError as e:
                # Likely busy during build; report as not-ok but informative
                ok = False
                stats = {'error': 'busy', 'detail': str(e)}
            except Exception as e:
                ok = False
                stats = {'error': str(e)}
        return self._write_json({'ok': ok, 'db': str(config.DB_PATH), 'stats': stats})

    def api_papers(self):
        try:
            con = self.db()
            rows = con.execute('SELECT id, arxiv_id FROM paper').fetchall()
            counts = dict(con.execute('SELECT paper_id, COUNT(1) FROM chunk GROUP BY paper_id').fetchall())
            papers = [
                {'id': pid, 'arxiv_id': aid, 'chunks': int(counts.get(pid, 0))}
                for (pid, aid) in rows
            ]
            return self._write_json({'ok': True, 'papers': papers})
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})

    def api_search(self):
        body = self._read_json()
        query = (body.get('query') or '').strip()
        # Scoring options
        scoring = str(body.get('scoring', 'tfidf') or 'tfidf').strip().lower()
        bm25_opts = body.get('bm25') or {}
        try:
            bm25_k1 = float(bm25_opts.get('k1', 1.2) if isinstance(bm25_opts, dict) else 1.2)
        except Exception:
            bm25_k1 = 1.2
        try:
            bm25_b = float(bm25_opts.get('b', 0.75) if isinstance(bm25_opts, dict) else 0.75)
        except Exception:
            bm25_b = 0.75
        svd_opts = body.get('svd') or {}
        # Client-supplied knobs (clamped below)
        top_k = int(body.get('top_k', 20) or 20)
        kind = body.get('kind') or 'both'
        max_candidates = int(body.get('max_candidates', 10000) or 10000)
        try:
            top_m_tokens = int(body.get('top_m_tokens', 12) or 12)
        except Exception:
            top_m_tokens = 12
        # Optional paging (future use)
        try:
            offset = int(body.get('offset') or 0)
        except Exception:
            offset = 0

        # Safety caps to prevent pathological queries from freezing UI/DB
        try:
            MAX_RESULTS = int(os.environ.get('PROJECTSEARCHBAR_MAX_RESULTS') or '5000')
        except Exception:
            MAX_RESULTS = 5000
        try:
            MAX_CANDS = int(os.environ.get('PROJECTSEARCHBAR_MAX_CANDIDATES') or '20000')
        except Exception:
            MAX_CANDS = 20000
        top_k = max(1, min(top_k, MAX_RESULTS))
        max_candidates = max(100, min(max_candidates, MAX_CANDS))
        top_m_tokens = int(body.get('top_m_tokens', 12) or 12)
        per_paper_k = int(body.get('per_paper_k', 1) or 1)

        if not query:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        try:
            # Ensure we see the latest DB after any swap
            self.refresh_db_if_changed()
            con = self.db()
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})
        q_tokens = tokenize_query(query)
        # Optional compatibility: allow disabling filters for legacy behavior
        NO_FILTER = os.environ.get('PROJECTSEARCHBAR_NO_FILTER', '0') == '1'
        # Filter noisy tokens from long prompts: drop stopwords and very short tokens
        if not NO_FILTER:
            q_tokens = [t for t in q_tokens if len(t) >= 2 and t not in STOPWORDS]
        if not q_tokens:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # TF in query
        tfq = {}
        for t in q_tokens:
            tfq[t] = tfq.get(t, 0) + 1

        # Map tokens to ids and dfs
        placeholders = ','.join('?' * len(tfq))
        cur = con.execute(f"SELECT term, id, df FROM token WHERE term IN ({placeholders})", list(tfq.keys()))
        term_to_id: dict[str, int] = {}
        term_to_df: dict[str, int] = {}
        for term, tid, df in cur.fetchall():
            term_to_id[term] = int(tid)
            term_to_df[term] = int(df)
        if not term_to_id:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # Global N
        N = con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1

        # Drop extremely common tokens to avoid scanning the world (requires N)
        if not NO_FILTER:
            try:
                MAX_DF_FRAC = float(os.environ.get('PROJECTSEARCHBAR_MAX_DF_FRAC') or '0.15')
            except Exception:
                MAX_DF_FRAC = 0.15
            try:
                MAX_DF_ABS = int(os.environ.get('PROJECTSEARCHBAR_MAX_DF') or '200000')
            except Exception:
                MAX_DF_ABS = 200000
            def too_common(df: int) -> bool:
                return (df >= MAX_DF_ABS) or (N and (df / float(N)) >= MAX_DF_FRAC)
            filtered_tfq = {t:c for (t,c) in tfq.items() if not too_common(term_to_df.get(t, 0))}
            if filtered_tfq:
                tfq = filtered_tfq
                # Rebuild mappings to only include filtered terms
                term_to_id = {t:term_to_id[t] for t in tfq.keys() if t in term_to_id}
                term_to_df = {t:term_to_df[t] for t in tfq.keys() if t in term_to_df}
            # Ensure we keep at least a few informative tokens for long prompts
            try:
                MIN_KEEP = int(os.environ.get('PROJECTSEARCHBAR_MIN_KEEP') or '3')
            except Exception:
                MIN_KEEP = 3
            if len(tfq) < max(1, MIN_KEEP):
                # Pick top MIN_KEEP by lowest df from the original mappings
                scored = sorted([(term_to_df.get(t, 0), t) for t in term_to_id.keys()])
                rescue = [t for _, t in scored[:max(1, MIN_KEEP)]]
                tfq = {t: tfq.get(t, 1) for t in rescue if t in term_to_id}
                term_to_id = {t: term_to_id[t] for t in tfq.keys()}
                term_to_df = {t: term_to_df.get(t, 0) for t in tfq.keys()}
        if not tfq:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # Compute idf and q vector
        q_vec = {}
        idf_by_tid: dict[int, float] = {}
        df_by_tid: dict[int, int] = {}
        for term, cnt in tfq.items():
            tid = term_to_id.get(term)
            if tid is None:
                continue
            df = term_to_df.get(term, 0)
            idf = math.log((1.0 + N) / (1.0 + float(df))) + 1.0
            idf_by_tid[tid] = idf
            df_by_tid[tid] = int(df or 0)
            q_val = (1.0 + math.log(float(cnt))) * idf
            q_vec[tid] = q_val
        # L2 normalize
        q_norm = math.sqrt(sum(v*v for v in q_vec.values())) or 1.0
        for tid in list(q_vec.keys()):
            q_vec[tid] /= q_norm
        if not q_vec:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # Candidate generation via top-M rare tokens (adaptive to avoid huge scans)
        by_rarity = sorted([(idf_by_tid.get(tid, 0.0), tid) for tid in q_vec.keys()], reverse=True)
        chosen_tids = [tid for _, tid in by_rarity[:top_m_tokens]] or list(q_vec.keys())
        # Adaptive reduction if the combined DF across chosen tokens is too large
        try:
            SCAN_BUDGET = int(os.environ.get('PROJECTSEARCHBAR_SCAN_BUDGET') or '2000000')
        except Exception:
            SCAN_BUDGET = 2000000
        def total_df(tids: list[int]) -> int:
            return sum(int(df_by_tid.get(t, 0)) for t in tids)
        while len(chosen_tids) > 1 and total_df(chosen_tids) > SCAN_BUDGET:
            chosen_tids = chosen_tids[:-1]

        # Build candidate list
        if not chosen_tids:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})
        ph = ','.join('?' * len(chosen_tids))
        if kind and kind != 'both':
            cand_sql = f"""
                SELECT p.chunk_id, COUNT(*) as matches
                FROM posting p
                JOIN chunk c ON c.id = p.chunk_id
                WHERE p.token_id IN ({ph}) AND c.kind = ?
                GROUP BY p.chunk_id
                ORDER BY matches DESC
                LIMIT ? OFFSET ?
            """
            cand_params = list(chosen_tids) + [kind, max_candidates, max(0, offset)]
        else:
            cand_sql = f"""
                SELECT chunk_id, COUNT(*) as matches
                FROM posting
                WHERE token_id IN ({ph})
                GROUP BY chunk_id
                ORDER BY matches DESC
                LIMIT ? OFFSET ?
            """
            cand_params = list(chosen_tids) + [max_candidates, max(0, offset)]
        # Optional time budget guard; can be disabled via env
        start_time = time.time()
        NO_TIMEOUT = os.environ.get('PROJECTSEARCHBAR_NO_TIMEOUT', '0') == '1'
        def guard():
            if NO_TIMEOUT:
                return 0
            if (time.time() - start_time) > 2.0:
                return 1
            return 0
        try:
            try:
                con.set_progress_handler(guard, 10000)  # type: ignore[attr-defined]
            except Exception:
                pass
            candidates = [row[0] for row in con.execute(cand_sql, cand_params).fetchall()]
        except sqlite3.OperationalError as e:
            # Database is busy/locked; surface friendly error
            msg = str(e)
            if 'interrupted' in msg.lower():
                return self._write_json({'ok': False, 'error': 'timeout', 'detail': 'candidate query exceeded time budget'})
            return self._write_json({'ok': False, 'error': 'busy', 'detail': msg})
        finally:
            try:
                con.set_progress_handler(None, 0)  # type: ignore[attr-defined]
            except Exception:
                pass
        if not candidates:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # Compute scores on candidates (parallelizable)
        scores = {}
        bm25_scores: dict[int, float] = {}
        CHUNK_SIZE = 1200
        # Determine worker count
        def parse_workers():
            w = os.environ.get('PROJECTSEARCHBAR_WORKERS', 'auto')
            if isinstance(w, str) and w.lower() == 'auto':
                try:
                    c = os.cpu_count() or 4
                    return max(1, min(8, c))
                except Exception:
                    return 4
            try:
                n = int(w)
                return max(1, n)
            except Exception:
                return 1
        workers = 1  # force single-thread to compute TF–IDF and BM25 reliably together

        if workers <= 1 or len(candidates) <= CHUNK_SIZE*2:
            # Single-threaded scoring path: compute TF–IDF dot and BM25 together
            for i in range(0, len(candidates), CHUNK_SIZE):
                sub = candidates[i:i+CHUNK_SIZE]
                ph_c = ','.join('?' * len(sub))
                ph_t = ','.join('?' * len(q_vec))
                # Pre-fetch document lengths for BM25 (sum tf over all terms in chunk)
                len_by_cid: dict[int, float] = {}
                try:
                    rows_len = con.execute(
                        f"SELECT chunk_id, SUM(tf) FROM posting WHERE chunk_id IN ({ph_c}) GROUP BY chunk_id",
                        sub,
                    ).fetchall()
                    len_by_cid = {int(cid): float(total or 0.0) for cid, total in rows_len}
                except Exception:
                    len_by_cid = {}
                sql = f"SELECT chunk_id, token_id, tf FROM posting WHERE chunk_id IN ({ph_c}) AND token_id IN ({ph_t})"
                params = sub + list(q_vec.keys())
                try:
                    start_time = time.time()
                    def guard2():
                        if NO_TIMEOUT:
                            return 0
                        if (time.time() - start_time) > 2.5:
                            return 1
                        return 0
                    try:
                        con.set_progress_handler(guard2, 20000)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    rows = con.execute(sql, params).fetchall()
                except sqlite3.OperationalError as e:
                    msg = str(e)
                    if 'interrupted' in msg.lower():
                        return self._write_json({'ok': False, 'error': 'timeout', 'detail': 'scoring batch exceeded time budget'})
                    return self._write_json({'ok': False, 'error': 'busy', 'detail': msg})
                finally:
                    try:
                        con.set_progress_handler(None, 0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # BM25 idf once per term (strictly-positive variant)
                bm25_idf: dict[int, float] = {}
                for ttid, dfv in df_by_tid.items():
                    try:
                        num = (float(N) - float(dfv) + 0.5)
                        den = (float(dfv) + 0.5)
                        val = math.log(1.0 + max(0.0, num) / (den or 1.0))
                    except Exception:
                        val = 0.0
                    bm25_idf[ttid] = float(val if val > 0.0 else 0.0)
                avgdl = self._bm25_avgdl()
                for cid, tid, tf in rows:
                    qv = q_vec.get(int(tid))
                    if qv is None:
                        continue
                    tfw = 1.0 + math.log(float(tf))
                    idf = idf_by_tid.get(int(tid), 1.0)
                    dot = qv * (tfw * idf)
                    scores[cid] = scores.get(cid, 0.0) + dot
                    # BM25 contribution
                    dl = float(len_by_cid.get(int(cid), 0.0))
                    k1 = bm25_k1; b = bm25_b
                    denom = (float(tf) + k1 * (1.0 - b + b * (dl / (avgdl or 1.0))))
                    w = ((k1 + 1.0) * float(tf)) / (denom or 1.0)
                    w *= bm25_idf.get(int(tid), 0.0)
                    bm25_scores[cid] = bm25_scores.get(cid, 0.0) + w
        else:
            # Parallel scoring across slices of candidates
            import concurrent.futures as _fut
            def _mk_conn():
                c = sqlite3.connect(str(config.DB_PATH), timeout=10.0, check_same_thread=False)
                try:
                    c.execute('PRAGMA journal_mode=WAL')
                    c.execute('PRAGMA synchronous=NORMAL')
                    c.execute('PRAGMA busy_timeout=10000')
                    c.execute('PRAGMA temp_store=MEMORY')
                    try: c.execute('PRAGMA cache_size=-100000')
                    except Exception: pass
                    try: c.execute('PRAGMA mmap_size=1073741824')
                    except Exception: pass
                except Exception:
                    pass
                return c
            # Partition candidates into roughly equal slices
            n = workers
            L = len(candidates)
            step = (L + n - 1) // n
            slices = [(i, min(i+step, L)) for i in range(0, L, step)]
            def worker(a_b):
                a, b = a_b
                local = {}
                conn2 = _mk_conn()
                try:
                    for i in range(a, b, CHUNK_SIZE):
                        sub = candidates[i:i+CHUNK_SIZE]
                        if not sub:
                            continue
                        ph_c = ','.join('?' * len(sub))
                        ph_t = ','.join('?' * len(q_vec))
                        sql = f"SELECT chunk_id, token_id, tf FROM posting WHERE chunk_id IN ({ph_c}) AND token_id IN ({ph_t})"
                        params = sub + list(q_vec.keys())
                        rows = conn2.execute(sql, params).fetchall()
                        for cid, tid, tf in rows:
                            qv = q_vec.get(int(tid))
                            if qv is None:
                                continue
                            tfw = 1.0 + math.log(float(tf))
                            idf = idf_by_tid.get(int(tid), 1.0)
                            local[cid] = local.get(cid, 0.0) + (qv * (tfw * idf))
                finally:
                    try: conn2.close()
                    except Exception: pass
                return local
            with _fut.ThreadPoolExecutor(max_workers=workers) as ex:
                for partial in ex.map(worker, slices):
                    for k, v in partial.items():
                        scores[k] = scores.get(k, 0.0) + v

        if not scores:
            return self._write_json({'ok': True, 'results': [], 'dims': ["", ""], 'scanned': 0})

        # Finalize cosine with chunk.norm
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_candidates]
        cids = [cid for cid, _ in top]
        results = []
        for i in range(0, len(cids), CHUNK_SIZE):
            sub = cids[i:i+CHUNK_SIZE]
            ph_c = ','.join('?' * len(sub))
            for cid, paper_id, kind_val, text, norm in con.execute(
                f"SELECT id, paper_id, kind, text, norm FROM chunk WHERE id IN ({ph_c})",
                sub,
            ).fetchall():
                s = scores.get(cid, 0.0)
                cos = s / (norm or 1.0)
                b25 = bm25_scores.get(int(cid), 0.0)
                results.append((cos, cid, paper_id, kind_val, text, b25))

        # Attach arXiv ids and limit per paper
        out = []
        paper_cache: dict[int, str] = {}
        def paper_arxiv(pid: int) -> str:
            if pid in paper_cache:
                return paper_cache[pid]
            row = con.execute('SELECT arxiv_id FROM paper WHERE id=?', (pid,)).fetchone()
            aid = row[0] if row and row[0] else str(pid)
            paper_cache[pid] = aid
            return aid

        # Choose primary ranking
        if scoring == 'bm25':
            sorted_res = sorted(results, key=lambda e: e[5], reverse=True)
        else:
            sorted_res = sorted(results, key=lambda e: e[0], reverse=True)

        # Optional two-pass re-rank (e.g., BM25 candidates → cosine re-rank)
        two_pass = body.get('two_pass') or {}
        try:
            tp_enabled = bool(two_pass.get('enabled', False))
            tp_primary = (two_pass.get('primary') or 'bm25').strip().lower()
            tp_secondary = (two_pass.get('secondary') or 'cosine').strip().lower()
            tp_topN = int(two_pass.get('topN', 2000) or 2000)
        except Exception:
            tp_enabled = False; tp_primary = 'bm25'; tp_secondary = 'cosine'; tp_topN = 2000
        if tp_enabled:
            try:
                # Base ordering by primary metric
                if tp_primary == 'cosine':
                    base = sorted(results, key=lambda e: e[0], reverse=True)
                else:
                    base = sorted(results, key=lambda e: e[5], reverse=True)
                head = base[:min(tp_topN, len(base))]
                # Re-rank head by secondary
                if tp_secondary == 'bm25':
                    head = sorted(head, key=lambda e: e[5], reverse=True)
                else:
                    head = sorted(head, key=lambda e: e[0], reverse=True)
                # Append tail (keep original primary ordering for the rest)
                tail = base[len(head):]
                sorted_res = head + tail
            except Exception:
                pass

        # Optional SVD/LSI re-ranking on top-N candidates
        svd_applied = False
        svd_used_count = 0
        svd_assets_ok = False
        # Pre-check SVD assets presence regardless of whether SVD is requested
        try:
            svd_root = config.DATA_DIR / 'svd'
            model = svd_root / 'model.json'
            comp = svd_root / 'components.npy'
            tmapf = svd_root / 'token_map.json'
            svds = svd_root / 'svd.sqlite'
            if model.exists() and comp.exists() and tmapf.exists() and svds.exists():
                svd_assets_ok = True
        except Exception:
            pass
        try:
            use_svd = bool(svd_opts.get('enabled', False))
            topN = int(svd_opts.get('topN', 2000) or 2000)
            alpha = float(svd_opts.get('alpha', 0.5) or 0.5)
        except Exception:
            use_svd = False; topN = 2000; alpha = 0.5
        if use_svd:
            # Lazy-load SVD assets once
            self._svd_try_load()
            comps = getattr(SearchRequestHandler, '_svd_components', None)
            tmap = getattr(SearchRequestHandler, '_svd_token_map', None)
            svd_con = getattr(SearchRequestHandler, '_svd_db', None)
            if comps is not None and tmap is not None and svd_con is not None:
                svd_assets_ok = True
                try:
                    import numpy as _np  # type: ignore
                    # Build query embedding: q_emb = sum_t q_val * comps[:, col(t)]
                    q_emb = _np.zeros(comps.shape[0], dtype=_np.float32)
                    for tid, qv in q_vec.items():
                        col = tmap.get(int(tid)) if isinstance(tmap, dict) else None
                        if col is None:
                            continue
                        q_emb += float(qv) * comps[:, int(col)].astype(_np.float32)
                    # Normalize
                    nrm = float(_np.linalg.norm(q_emb)) or 1.0
                    q_emb /= nrm
                    # Take topN by original score
                    base = sorted_res[:min(topN, len(sorted_res))]
                    svd_used_count = len(base)
                    cidsN = [int(cid) for (_cos, cid, _pid, _k, _t, _b25) in base]
                    # Fetch embeddings in small batches
                    rescored = []
                    B = 500
                    for i in range(0, len(cidsN), B):
                        sub = cidsN[i:i+B]
                        ph = ','.join('?' * len(sub))
                        rows = svd_con.execute(f'SELECT id, v FROM chunk_svd WHERE id IN ({ph})', sub).fetchall()
                        vec_map = {int(r[0]): r[1] for r in rows}
                        for (orig, cid, pid, kind_val, text, _b25) in base[i:i+B]:
                            blob = vec_map.get(int(cid))
                            if blob is None:
                                # If missing, keep original score
                                blend = float(orig)
                            else:
                                v = _np.frombuffer(blob, dtype=_np.float32)
                                svd_cos = float(_np.dot(q_emb, v))
                                blend = float(alpha * float(orig) + (1.0 - alpha) * svd_cos)
                            rescored.append((blend, cid, pid, kind_val, text, _b25))
                    # If any tail beyond topN, append with original scores to preserve length
                    if len(sorted_res) > len(base):
                        rescored.extend(sorted_res[len(base):])
                    # Replace sorted_res with blended ordering
                    sorted_res = sorted(rescored, key=lambda e: e[0], reverse=True)
                    svd_applied = True
                except Exception:
                    pass
        by = {}
        selected = []
        for cos_val, cid, pid, kind_val, text, b25_val in sorted_res:
            aid = paper_arxiv(int(pid))
            base = aid.rsplit('v', 1)[0] if 'v' in aid else aid
            cnt = by.get(base, 0)
            if cnt >= per_paper_k:
                continue
            by[base] = cnt + 1
            primary = float(b25_val) if scoring == 'bm25' else float(cos_val)
            selected.append((primary, cos_val, b25_val, cid, pid, kind_val, text))
            if len(selected) >= top_k:
                break

        for primary, cos_val, b25_val, cid, pid, kind_val, text in selected:
            out.append({
                'paperId': paper_arxiv(int(pid)),
                'chunkId': int(cid),
                'kind': kind_val,
                'text': (text or '')[:2000],
                'score': float(primary),
                'cosine': float(cos_val),
                'bm25': float(b25_val),
                'scoring': scoring,
            })

        # Pick 2 token labels for viz
        dims = ["", ""]
        if q_vec:
            tids = list(q_vec.keys())
            ph = ','.join('?' * len(tids))
            id_term = {tid: term for (tid, term) in self.db().execute(
                f"SELECT id, term FROM token WHERE id IN ({ph})", tids
            ).fetchall()}
            q_sorted = sorted(q_vec.items(), key=lambda kv: abs(kv[1]), reverse=True)
            dims = [id_term.get(q_sorted[0][0], ''), id_term.get(q_sorted[1][0], '') if len(q_sorted) > 1 else '']

        return self._write_json({
            'ok': True,
            'results': out,
            'dims': dims,
            'scanned': len(candidates),
            'svd_applied': bool(svd_applied),
            'svd_used_count': int(svd_used_count),
            'svd_assets': bool(svd_assets_ok),
            'svd_topN': int(topN) if 'topN' in locals() else None,
            'svd_alpha': float(alpha) if 'alpha' in locals() else None,
        })

    # --- BM25 helpers ---
    _bm25_cache = {'sig': None, 'avgdl': None}

    @classmethod
    def _bm25_avgdl(cls) -> float:
        try:
            sig = cls._db_signature()
            if cls._bm25_cache.get('sig') == sig and isinstance(cls._bm25_cache.get('avgdl'), (int, float)):
                return float(cls._bm25_cache.get('avgdl') or 1.0)
            con = cls.db()
            total_tf_row = con.execute('SELECT SUM(tf) FROM posting').fetchone()
            total_tf = float(total_tf_row[0] or 0.0)
            N = float(con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1.0)
            avgdl = (total_tf / N) if N > 0 else 1.0
            cls._bm25_cache['sig'] = sig
            cls._bm25_cache['avgdl'] = avgdl
            return float(avgdl or 1.0)
        except Exception:
            return 1.0

    def api_index_build(self):
        # Starts a background index build: vectorize + merge
        body = self._read_json()
        reset = bool(body.get('reset', False))
        mode = str(body.get('mode', 'index-only')).strip().lower()
        limit = body.get('limit')
        # Optional build tuning from UI settings
        low_mem = bool(body.get('low_mem', False))
        fast_build = body.get('fast_build')
        try:
            fast_build = bool(fast_build) if fast_build is not None else None
        except Exception:
            fast_build = None
        post_batch = body.get('post_batch')
        try:
            post_batch = int(post_batch) if post_batch is not None else None
        except Exception:
            post_batch = None
        commit_papers = body.get('commit_papers')
        try:
            commit_papers = int(commit_papers) if commit_papers is not None else None
        except Exception:
            commit_papers = None
        try:
            limit = int(limit) if limit is not None else None
        except Exception:
            limit = None

        status_path = config.DATA_DIR / 'build_status.json'
        tmp_status_path = status_path.with_suffix('.tmp')
        def write_status(obj: dict):
            try:
                # Always attach build_id if present, and timestamp
                payload = {**obj, 'ts': time.time()}
                if SearchRequestHandler._build_id:
                    payload['build_id'] = SearchRequestHandler._build_id
                tmp_status_path.write_text(json.dumps(payload), encoding='utf-8')
                try:
                    os.replace(str(tmp_status_path), str(status_path))
                except Exception:
                    tmp_status_path.replace(status_path)
            except Exception:
                pass

        # Background worker to avoid blocking the UI
        def compute_total_papers() -> int:
            try:
                scan = config.VECTORS_DIR
                all_dirs = [p for p in sorted(scan.iterdir()) if p.is_dir()]
                papers = [p for p in all_dirs if (p / 'chunks.jsonl').exists()]
                if limit:
                    papers = papers[:limit]
                return len(papers)
            except Exception:
                return 0

        def worker():
            global os
            try:
                # Import tools fresh each run to pick up code changes
                import importlib
                vec_mod = importlib.import_module('ProjectSearchBar.tools.batch_vectorize')
                idx_mod = importlib.import_module('ProjectSearchBar.tools.index_merge')
                try:
                    importlib.reload(vec_mod)
                    importlib.reload(idx_mod)
                except Exception:
                    pass
                # Ensure no open connections hold locks during reset/build
                SearchRequestHandler.close_db()
                write_status({'ok': True, 'state': 'starting'})
                # Vectorize (optional)
                do_vectorize = (mode == 'full')
                try:
                    # If vectors folder already has subfolders, we can skip unless explicitly "full"
                    if not do_vectorize:
                        subdirs = [p for p in config.VECTORS_DIR.iterdir() if p.is_dir()]
                        do_vectorize = (len(subdirs) == 0)
                except Exception:
                    pass
                if do_vectorize:
                    vec_args = [
                        '--src', str(config.PAPERS_SRC),
                        '--out', str(config.VECTORS_DIR),
                    ]
                    if limit:
                        vec_args += ['--limit', str(limit)]
                    # Auto workers based on CPU
                    try:
                        import os as _os
                        w = _os.environ.get('PROJECTSEARCHBAR_WORKERS') or 'auto'
                        vec_args += ['--workers', w]
                    except Exception:
                        pass
                    write_status({'ok': True, 'state': 'vectorizing'})
                    print('[index_build] Vectorizing from', config.PAPERS_SRC, 'to', config.VECTORS_DIR)
                    vec_mod.main(vec_args)  # type: ignore[arg-type]
                else:
                    print('[index_build] Skipping vectorize step (vectors present).')
                # Early announce indexing counters baseline to avoid UI jump
                try:
                    total = compute_total_papers()
                    write_status({'ok': True, 'state': 'indexing', 'papers_total': total, 'papers_done': 0, 'chunks': 0, 'postings': 0})
                except Exception:
                    pass
                # Select build strategy
                build_mode = (mode or 'index-only').strip().lower()
                if build_mode == 'sharded':
                    # Use sharded builder for large datasets
                    try:
                        import importlib
                        shard_mod = importlib.import_module('ProjectSearchBar.tools.sharded_index')
                    except Exception as e:
                        write_status({'ok': False, 'state': 'error', 'error': f'sharded_import: {e}'})
                        return
                    # Prepare args
                    try:
                        shards = int(os.environ.get('PROJECTSEARCHBAR_SHARDS') or body.get('shards') or 4)
                    except Exception:
                        shards = 4
                    try:
                        workers = int(os.environ.get('PROJECTSEARCHBAR_SHARD_WORKERS') or body.get('workers') or 4)
                    except Exception:
                        workers = 4
                    write_status({'ok': True, 'state': 'indexing', 'phase': 'sharded'})
                    print('[index_build] Sharded merge from', config.VECTORS_DIR, 'to', config.DB_PATH, 'shards=', shards, 'workers=', workers)
                    # Call sharded builder (it handles atomic swap internally)
                    shard_args = [
                        '--scan', str(config.VECTORS_DIR),
                        '--out-db', str(config.DB_PATH),
                        '--shards', str(max(1, shards)),
                        '--workers', str(max(1, workers)),
                        '--reset',
                    ]
                    # Close DB to avoid lock during swap
                    SearchRequestHandler.close_db()
                    shard_mod.main(shard_args)  # type: ignore[arg-type]
                    write_status({'ok': True, 'state': 'done'})
                else:
                    # Single DB build path
                    build_db = config.DATA_DIR / 'index.build.sqlite'
                    try:
                        if build_db.exists():
                            build_db.unlink()
                    except Exception:
                        pass
                    idx_args = [
                        '--scan', str(config.VECTORS_DIR),
                        '--db', str(build_db),
                    ]
                    # Always reset when building a temp DB
                    idx_args.append('--reset')
                    if limit:
                        idx_args += ['--limit-papers', str(limit)]
                    write_status({'ok': True, 'state': 'indexing'})
                    print('[index_build] Merging index from', config.VECTORS_DIR, 'to', build_db)
                    # Pass status path to indexer for fine-grained progress
                    os.environ['PROJECTSEARCHBAR_STATUS'] = str(status_path)
                    if SearchRequestHandler._build_id:
                        os.environ['PROJECTSEARCHBAR_BUILD_ID'] = SearchRequestHandler._build_id
                    print('[index_build] Status path:', os.environ.get('PROJECTSEARCHBAR_STATUS'))
                    # Apply optional tuning from settings
                    if low_mem:
                        os.environ['PROJECTSEARCHBAR_LOW_MEM'] = '1'
                        # low_mem implies we disable the aggressive fast-build PRAGMAs unless explicitly forced
                        if fast_build is None:
                            os.environ['PROJECTSEARCHBAR_FAST_BUILD'] = '0'
                    if fast_build is not None:
                        os.environ['PROJECTSEARCHBAR_FAST_BUILD'] = '1' if fast_build else '0'
                    if post_batch is not None and post_batch > 0:
                        os.environ['PROJECTSEARCHBAR_POST_BATCH'] = str(post_batch)
                    if commit_papers is not None and commit_papers > 0:
                        os.environ['PROJECTSEARCHBAR_COMMIT_PAPERS'] = str(commit_papers)
                    idx_mod.main(idx_args)  # type: ignore[arg-type]
                    # Atomically swap in the new DB
                    write_status({'ok': True, 'state': 'swapping'})
                    print('[index_build] Swapping new DB into place:', build_db, '->', config.DB_PATH)
                    try:
                        # Ensure server is not holding a handle
                        SearchRequestHandler.close_db()
                    except Exception:
                        pass
                    try:
                        os.replace(str(build_db), str(config.DB_PATH))
                    except Exception:
                        # Fallback: rename
                        build_db.replace(config.DB_PATH)
                    write_status({'ok': True, 'state': 'done'})
            except Exception as e:
                print('[index_build] error:', e)
                write_status({'ok': False, 'state': 'error', 'error': str(e)})
            finally:
                # Re-open lazily on next request
                SearchRequestHandler.close_db()
                with SearchRequestHandler._build_lock:
                    SearchRequestHandler._build_active = False
                    SearchRequestHandler._build_id = None

        # Prevent starting multiple concurrent builds
        with SearchRequestHandler._build_lock:
            if SearchRequestHandler._build_active:
                return self._write_json({'ok': False, 'started': False, 'reason': 'busy'})
            # New build id per run
            try:
                import uuid
                SearchRequestHandler._build_id = uuid.uuid4().hex
            except Exception:
                SearchRequestHandler._build_id = str(int(time.time() * 1000))
            SearchRequestHandler._build_active = True
            # Advertise baseline indexing counters immediately to avoid initial jump
            try:
                total = compute_total_papers()
                # Clear any old status file before new baseline
                try:
                    status_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                write_status({'ok': True, 'state': 'indexing', 'papers_total': total, 'papers_done': 0, 'chunks': 0, 'postings': 0})
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()
        # Also return the computed total so the UI can display an immediate baseline
        return self._write_json({'ok': True, 'started': True, 'build_id': SearchRequestHandler._build_id, 'papers_total': locals().get('total', None)})

    def api_index_status(self):
        status_path = config.DATA_DIR / 'build_status.json'
        if status_path.exists():
            try:
                obj = json.loads(status_path.read_text(encoding='utf-8'))
            except Exception:
                obj = {'ok': False, 'state': 'unknown'}
        else:
            obj = {'ok': True, 'state': 'idle'}
        return self._write_json(obj)

    # launch endpoints removed

    # --- LLM settings and ask ---
    def _llm_settings_path(self) -> Path:
        return config.DATA_DIR / 'llm_settings.json'

    def _read_llm_settings(self) -> dict:
        p = self._llm_settings_path()
        try:
            if p.exists():
                return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}

    def _write_llm_settings(self, obj: dict) -> None:
        p = self._llm_settings_path()
        try:
            p.write_text(json.dumps(obj), encoding='utf-8')
            try:
                p.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
        except Exception:
            pass

    # --- Search settings (persisted) ---
    def _search_settings_path(self) -> Path:
        return config.DATA_DIR / 'search_settings.json'

    def _read_search_settings(self) -> dict:
        p = self._search_settings_path()
        try:
            if p.exists():
                return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}

    def _write_search_settings(self, obj: dict) -> None:
        p = self._search_settings_path()
        try:
            p.write_text(json.dumps(obj), encoding='utf-8')
            try:
                p.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
        except Exception:
            pass

    def api_search_settings_get(self):
        s = self._read_search_settings()
        return self._write_json({'ok': True, 'settings': s})

    def api_search_settings_post(self):
        body = self._read_json()
        obj = body.get('settings') if isinstance(body, dict) else None
        if not isinstance(obj, dict):
            return self._write_json({'ok': False, 'error': 'invalid_payload'}, 200)
        # whitelist keys
        allowed = {
            'kind', 'perPaper', 'maxResults', 'scoring',
            'bm25K1', 'bm25B',
            'svdEnable', 'svdTopN', 'svdAlpha',
            'twoPassEnable', 'twoPassTopN'
        }
        clean = {k: obj[k] for k in list(obj.keys()) if k in allowed}
        # Merge with existing settings to avoid dropping unspecified keys
        try:
            prev = self._read_search_settings()
            if not isinstance(prev, dict):
                prev = {}
        except Exception:
            prev = {}
        merged = dict(prev)
        merged.update(clean)
        self._write_search_settings(merged)
        return self._write_json({'ok': True, 'settings': merged})

    def api_llm_settings_get(self):
        s = self._read_llm_settings()
        provider = s.get('provider') or 'openai'
        model = s.get('model') or 'gpt-5-mini'
        # Normalize legacy aliases
        if isinstance(model, str):
            alias = model.strip().lower()
            if alias == 'chatgpt-5':
                model = 'gpt-5'
            elif alias == 'chatgpt-5-mini':
                model = 'gpt-5-mini'
        has_key = bool(s.get('openai_api_key'))
        return self._write_json({'ok': True, 'provider': provider, 'model': model, 'has_key': has_key})

    def api_ui_info(self):
        try:
            root = getattr(config, 'UI_PUBLIC', None)
            fb = getattr(config, 'UI_PUBLIC_FALLBACK', None)
            index_path = None
            try:
                if root is not None:
                    p = root / 'index.html'
                    if p.exists():
                        index_path = str(p)
            except Exception:
                index_path = None
            sig = None
            has_scoring = False
            if index_path:
                try:
                    st = os.stat(index_path)
                    sig = {'mtime': st.st_mtime, 'size': st.st_size}
                    with open(index_path, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                        has_scoring = ('id="optScoring"' in txt) and ('id="optBm25K1"' in txt)
                except Exception:
                    pass
            return self._write_json({
                'ok': True,
                'ui_root': str(root) if root is not None else None,
                'ui_fallback': str(fb) if fb is not None else None,
                'index': index_path,
                'index_sig': sig,
                'has_bm25_controls': bool(has_scoring)
            })
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})

    def api_llm_settings_post(self):
        body = self._read_json()
        s = self._read_llm_settings()
        provider = (body.get('provider') or 'openai').strip().lower()
        model = body.get('model') or s.get('model') or 'gpt-5-mini'
        # Normalize aliases
        if isinstance(model, str):
            m = model.strip().lower()
            if m == 'chatgpt-5':
                model = 'gpt-5'
            elif m == 'chatgpt-5-mini':
                model = 'gpt-5-mini'
        api_key = body.get('api_key')
        out = dict(s)
        out['provider'] = provider
        out['model'] = model
        if api_key:
            out['openai_api_key'] = api_key
        self._write_llm_settings(out)
        return self._write_json({'ok': True})

    def api_llm_test(self):
        # Connectivity and auth test for LLM provider
        offline = os.environ.get('PROJECTSEARCHBAR_OFFLINE', '0') == '1'
        s = self._read_llm_settings()
        provider = (s.get('provider') or 'openai').strip().lower()
        model = s.get('model') or 'gpt-5-mini'
        if provider != 'openai':
            return self._write_json({'ok': False, 'error': 'provider_not_supported', 'provider': provider})
        api_key = s.get('openai_api_key') or ''
        if not api_key:
            return self._write_json({'ok': False, 'error': 'no_api_key'})
        if offline:
            return self._write_json({'ok': False, 'error': 'offline_mode', 'detail': 'PROJECTSEARCHBAR_OFFLINE=1'})
        # Minimal test message
        messages = [
            {'role': 'system', 'content': 'You are a test. Reply with OK.'},
            {'role': 'user', 'content': 'Say OK'},
        ]
        ok, out = self._openai_chat(model, messages, api_key, max_tokens=None)
        if ok:
            return self._write_json({'ok': True, 'model': model, 'sample': (out or '')[:100]})
        return self._write_json({'ok': False, 'error': 'llm_error', 'detail': out, 'model': model})

    def _openai_chat(self, model: str, messages: list[dict], api_key: str, max_tokens: int | None = None) -> tuple[bool, str]:
        # Honor offline mode and shorter request timeout to avoid UI timeouts
        if os.environ.get('PROJECTSEARCHBAR_OFFLINE', '0') == '1':
            return False, 'offline'
        url = 'https://api.openai.com/v1/chat/completions'
        payload = {'model': model, 'messages': messages}
        try:
            if max_tokens is not None:
                payload['max_tokens'] = int(max_tokens)
        except Exception:
            pass
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {api_key}')
        try:
            ctx = ssl.create_default_context()
            try:
                timeout = float(os.environ.get('PROJECTSEARCHBAR_LLM_TIMEOUT') or '25')
            except Exception:
                timeout = 25.0
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                j = json.loads(resp.read().decode('utf-8'))
                try:
                    txt = j['choices'][0]['message']['content']
                    return True, str(txt)
                except Exception:
                    return False, json.dumps(j)
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode('utf-8')
            except Exception:
                err = str(e)
            return False, err
        except Exception as e:
            return False, str(e)

    def _offline_summarize_blocks(self, context_blocks: list[tuple[str, str]], query: str, max_chars: int = 900) -> str:
        # Simple offline response if LLM is unavailable
        parts = ["Offline mode: LLM unavailable. Showing excerpts related to your question.\n"]
        take = min(3, len(context_blocks))
        for pid, txt in context_blocks[:take]:
            snippet = (txt or '').strip().replace('\r', ' ').replace('\n', ' ')
            snippet = ' '.join(snippet.split())[:max_chars]
            parts.append(f"- [{pid}] {snippet}...")
        parts.append("\nTip: Save an OpenAI API key in Settings to enable full answers.")
        return '\n'.join(parts)

    def _strip_latex_comments(self, s: str) -> str:
        out_lines = []
        for line in (s or '').splitlines():
            if '%' in line:
                i = line.find('%')
                # ignore escaped %
                if i > 0 and line[i-1] == '\\':
                    out_lines.append(line)
                else:
                    out_lines.append(line[:i])
            else:
                out_lines.append(line)
        return '\n'.join(out_lines)

    def _read_raw_latex(self, arxiv_id: str) -> str:
        # Cache file
        cache_dir = config.DATA_DIR / 'ai_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        base = str(arxiv_id).strip()
        underscore = base.replace('/', '_')
        cache_file = cache_dir / f"{underscore}.txt"
        try:
            if cache_file.exists():
                return cache_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            pass
        import tarfile
        src_root = Path(config.PAPERS_SRC)
        texts: list[str] = []
        # Try folder forms first
        for cand in (underscore, base):
            pdir = src_root / cand
            if pdir.exists() and pdir.is_dir():
                # Read all .tex sources
                for p in pdir.rglob('*.tex'):
                    try:
                        texts.append(p.read_text(encoding='utf-8', errors='ignore'))
                    except Exception:
                        pass
                # Append any .bbl bibliography (often holds references when using BibTeX)
                try:
                    for bbl in pdir.rglob('*.bbl'):
                        try:
                            texts.append(bbl.read_text(encoding='utf-8', errors='ignore'))
                        except Exception:
                            pass
                except Exception:
                    pass
                break
        if not texts:
            # Try common archive names
            for pref in (underscore, base):
                for ext in ('.tar.gz', '.tgz', '.tar'):
                    arc = src_root / f"{pref}{ext}"
                    if not arc.exists():
                        continue
                    try:
                        mode = 'r:gz' if ext in ('.tar.gz', '.tgz') else 'r'
                        with tarfile.open(arc, mode) as tf:
                            for m in tf.getmembers():
                                if not m.isfile():
                                    continue
                                name = m.name.lower()
                                if (name.endswith('.tex') or name.endswith('.bbl')) and m.size < 5_000_000:
                                    f = tf.extractfile(m)
                                    if f:
                                        try:
                                            texts.append(f.read().decode('utf-8', errors='ignore'))
                                        except Exception:
                                            pass
                        if texts:
                            break
                    except Exception:
                        pass
                if texts:
                    break
        combined = self._strip_latex_comments('\n\n'.join(texts)) if texts else ''
        try:
            cache_file.write_text(combined, encoding='utf-8')
        except Exception:
            pass
        return combined

    # ---------------- Chat session API ----------------
    def _chat_default_budget(self) -> int:
        try:
            return max(1000, int(os.environ.get('PROJECTSEARCHBAR_CHAT_BUDGET') or '6000'))
        except Exception:
            return 6000

    def _chat_require(self, sid: str) -> dict:
        with SearchRequestHandler._chat_lock:
            data = SearchRequestHandler._chat_sessions.get(sid)
            if not data:
                raise KeyError('invalid_session')
            return data

    def api_chat_start(self):
        try:
            import uuid
            sid = uuid.uuid4().hex
        except Exception:
            sid = str(int(time.time() * 1000))
        with SearchRequestHandler._chat_lock:
            SearchRequestHandler._chat_sessions[sid] = {
                'created': time.time(),
                'papers': [],
                'default_token_budget': self._chat_default_budget(),
            }
        return self._write_json({'ok': True, 'session_id': sid})

    def api_chat_state(self):
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            sid = (qs.get('session_id') or [''])[0]
            if not sid:
                return self._write_json({'ok': False, 'error': 'missing_session'})
            data = self._chat_require(sid)
            return self._write_json({'ok': True, 'session_id': sid, 'papers': data.get('papers', [])})
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})

    def api_chat_context(self):
        """Return full raw context for attached papers without sending a message.
        GET /api/chat/context?session_id=...&limit_chars=...
        """
        try:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            sid = (qs.get('session_id') or [''])[0]
            limit_raw = qs.get('limit_chars')
            limit = None
            try:
                if limit_raw and limit_raw[0]:
                    limit = max(1, int(limit_raw[0]))
            except Exception:
                limit = None
            if not sid:
                return self._write_json({'ok': False, 'error': 'missing_session'})
            data = self._chat_require(sid)
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})
        papers = list(data.get('papers') or [])
        out = []
        for p in papers:
            aid = str(p.get('id'))
            mode = p.get('mode') or 'latex'
            content = ''
            if mode == 'latex':
                content = self._read_raw_latex(aid)
            if not content:
                try:
                    self.refresh_db_if_changed()
                    con = self.db()
                    row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (aid,)).fetchone()
                    if row:
                        pid = int(row[0])
                        texts = [ (t or '').strip() for (t,) in con.execute('SELECT text FROM chunk WHERE paper_id=? ORDER BY id', (pid,)).fetchall() if t ]
                        content = '\n\n'.join(texts)
                        mode = 'vectors'
                except Exception:
                    pass
            content = (content or '')
            if isinstance(limit, int) and limit > 0 and len(content) > limit:
                content2 = content[:limit]
                trunc = True
            else:
                content2 = content
                trunc = False
            out.append({'paper': aid, 'mode': mode, 'chars': len(content), 'truncated': trunc, 'full': content2})
        return self._write_json({'ok': True, 'session_id': sid, 'papers': out})

    def api_chat_add_paper(self):
        body = self._read_json()
        sid = (body.get('session_id') or '').strip()
        paper_id = str(body.get('paper_id') or '').strip()
        mode = str(body.get('mode') or 'latex').strip().lower()
        if mode not in ('latex', 'vectors', 'auto'):
            mode = 'latex'
        if not sid or not paper_id:
            return self._write_json({'ok': False, 'error': 'missing_params'})
        try:
            data = self._chat_require(sid)
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        # Probe content to report preview/size; prefer LaTeX
        text = ''
        chosen_mode = 'latex'
        if mode in ('latex', 'auto'):
            text = self._read_raw_latex(paper_id)
        if not text and mode in ('vectors', 'auto'):
            # Fallback to vectors (join chunk texts)
            try:
                self.refresh_db_if_changed()
                con = self.db()
                row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (paper_id,)).fetchone()
                if row:
                    pid = int(row[0])
                    rows = con.execute('SELECT text FROM chunk WHERE paper_id=? ORDER BY id', (pid,)).fetchall()
                    texts = [ (t[0] or '').strip() for t in rows if t and t[0] ]
                    text = '\n\n'.join(texts)
                    chosen_mode = 'vectors'
            except Exception:
                pass
        if not text:
            # Try vectors directory directly if DB mapping failed
            try:
                pdir = config.VECTORS_DIR / paper_id.replace('/', '_')
                ch = pdir / 'chunks.jsonl'
                if ch.exists():
                    head = []
                    cnt = 0
                    with ch.open('r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                t = (rec.get('text') or '').strip()
                                if t:
                                    head.append(t)
                                    cnt += len(t)
                                    if cnt > 100000:
                                        break
                            except Exception:
                                pass
                    text = '\n\n'.join(head)
                    chosen_mode = 'vectors'
            except Exception:
                pass
        head = (text or '')[:800]
        info = {'id': paper_id, 'mode': chosen_mode, 'chars': len(text or '')}
        with SearchRequestHandler._chat_lock:
            # Avoid dup entries
            papers = data.setdefault('papers', [])
            if not any(p.get('id') == paper_id for p in papers):
                papers.append(info)
        return self._write_json({'ok': True, 'paper_id': paper_id, 'mode': chosen_mode, 'chars': len(text or ''), 'preview': head})

    def api_chat_remove_paper(self):
        body = self._read_json()
        sid = (body.get('session_id') or '').strip()
        paper_id = str(body.get('paper_id') or '').strip()
        if not sid or not paper_id:
            return self._write_json({'ok': False, 'error': 'missing_params'})
        try:
            data = self._chat_require(sid)
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        with SearchRequestHandler._chat_lock:
            arr = data.get('papers', [])
            data['papers'] = [p for p in arr if str(p.get('id')) != paper_id]
        return self._write_json({'ok': True})

    def api_chat_clear(self):
        body = self._read_json()
        sid = (body.get('session_id') or '').strip()
        if not sid:
            return self._write_json({'ok': False, 'error': 'missing_session'})
        try:
            data = self._chat_require(sid)
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        with SearchRequestHandler._chat_lock:
            data['papers'] = []
        return self._write_json({'ok': True})

    def api_chat_message(self):
        body = self._read_json()
        sid = (body.get('session_id') or '').strip()
        text = (body.get('text') or '').strip()
        try:
            token_budget = int(body.get('token_budget') or 0)
        except Exception:
            token_budget = 0
        if not token_budget:
            token_budget = self._chat_default_budget()
        if not sid or not text:
            return self._write_json({'ok': False, 'error': 'missing_params'})
        try:
            data = self._chat_require(sid)
        except KeyError:
            return self._write_json({'ok': False, 'error': 'invalid_session'})
        papers = list(data.get('papers') or [])
        if not papers:
            # Graceful: answer without context via generic /api/ask path
            s = self._read_llm_settings()
            provider = (s.get('provider') or 'openai').strip().lower()
            model = s.get('model') or 'gpt-5-mini'
            if provider != 'openai':
                return self._write_json({'ok': True, 'answer': 'No papers attached. Add papers to discuss them.', 'model': 'offline', 'offline': True})
            api_key = s.get('openai_api_key') or ''
            if not api_key:
                return self._write_json({'ok': False, 'error': 'no_api_key'})
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant. When using math, write LaTeX with $...$ or $$...$$.'},
                {'role': 'user', 'content': text},
            ]
            ok, out = self._openai_chat(str(model), messages, api_key, max_tokens=None)
            if ok:
                return self._write_json({'ok': True, 'answer': out, 'model': model})
            return self._write_json({'ok': True, 'answer': 'Could not contact the LLM.', 'model': 'offline', 'offline': True})
        # Build context from attached papers up to budget
        max_chars = max(1000, token_budget * 4)
        blocks: list[tuple[str, str]] = []
        previews_items: list[dict] = []
        total = 0
        for p in papers:
            aid = str(p.get('id'))
            mode = p.get('mode') or 'latex'
            content_src = ''
            if mode == 'latex':
                content_src = self._read_raw_latex(aid)
            if not content_src:
                # Fallback to all chunk texts
                try:
                    self.refresh_db_if_changed()
                    con = self.db()
                    row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (aid,)).fetchone()
                except Exception:
                    row = None
                if row:
                    try:
                        pid = int(row[0])
                        texts = [ (t or '').strip() for (t,) in con.execute('SELECT text FROM chunk WHERE paper_id=? ORDER BY id', (pid,)).fetchall() if t ]
                        content_src = '\n\n'.join(texts)
                        mode = 'vectors'
                    except Exception:
                        pass
                else:
                    # Try vectors dir as last resort
                    try:
                        pdir = config.VECTORS_DIR / aid.replace('/', '_')
                        chp = pdir / 'chunks.jsonl'
                        if chp.exists():
                            lines = []
                            with chp.open('r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    try:
                                        rec = json.loads(line)
                                        t = (rec.get('text') or '').strip()
                                        if t:
                                            lines.append(t)
                                    except Exception:
                                        pass
                            content_src = '\n\n'.join(lines)
                            mode = 'vectors'
                    except Exception:
                        pass
            content_src = (content_src or '').strip()
            if not content_src:
                continue
            sent = content_src
            truncated = False
            if total + len(sent) > max_chars:
                sent = sent[:max(0, max_chars - total)]
                truncated = True
            blocks.append((aid, sent))
            previews_items.append({
                'paper': aid,
                'mode': mode,
                'sent_chars': len(sent),
                'total_chars': len(content_src),
                'truncated': bool(truncated),
                'full': content_src,
            })
            total += len(sent)
            if total >= max_chars:
                break
        if not blocks:
            return self._write_json({'ok': False, 'error': 'no_context'})
        # Compose prompt
        parts = [
            'You are a careful assistant. Read the following excerpts to answer the user. Cite papers as [paperId].\n\n'
        ]
        for pid, txt in blocks:
            parts.append(f"[Source {pid}]\n{txt}\n\n")
        parts.append(f"Question: {text}\n\nAnswer succinctly with citations like [paperId].")
        prompt = ''.join(parts)
        # Build full previews with no truncation for debugging/inspection
        previews = previews_items
        s = self._read_llm_settings()
        provider = (s.get('provider') or 'openai').strip().lower()
        model = s.get('model') or 'gpt-5-mini'
        if provider != 'openai':
            offline = self._offline_summarize_blocks(blocks, text)
            return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': 'mixed', 'papers': [pid for pid,_ in blocks], 'previews': previews}})
        api_key = s.get('openai_api_key') or ''
        if not api_key:
            return self._write_json({'ok': False, 'error': 'no_api_key'})
        messages = [
            {'role': 'system', 'content': 'You answer using provided excerpts only. Use LaTeX $...$ or $$...$$ for math. Cite sources as [paperId].'},
            {'role': 'user', 'content': prompt},
        ]
        ok, out = self._openai_chat(str(model), messages, api_key, max_tokens=None)
        if not ok:
            offline = self._offline_summarize_blocks(blocks, text)
            return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': 'mixed', 'papers': [pid for pid,_ in blocks], 'previews': previews, 'chars_used_total': sum(p.get('sent_chars',0) for p in previews), 'chars_total_attached': sum(p.get('total_chars',0) for p in previews)}})
        return self._write_json({'ok': True, 'answer': out, 'model': model, 'used': {'mode': 'mixed', 'papers': [pid for pid,_ in blocks], 'previews': previews, 'chars_used_total': sum(p.get('sent_chars',0) for p in previews), 'chars_total_attached': sum(p.get('total_chars',0) for p in previews)}})

    def api_ask(self):
        body = self._read_json()
        query = (body.get('query') or '').strip()
        if not query:
            return self._write_json({'ok': False, 'error': 'empty'})
        s = self._read_llm_settings()
        provider = (s.get('provider') or 'openai').strip().lower()
        model = body.get('model') or s.get('model') or 'gpt-5-mini'
        # Normalize aliases
        if isinstance(model, str):
            m = model.strip().lower()
            if m == 'chatgpt-5':
                model = 'gpt-5'
            elif m == 'chatgpt-5-mini':
                model = 'gpt-5-mini'
        if provider != 'openai':
            # Graceful offline-like response for unsupported providers
            hint = ('Provider not supported. Set an OpenAI API key in Settings to enable answers.')
            return self._write_json({'ok': True, 'answer': hint, 'model': 'offline', 'offline': True})
        api_key = s.get('openai_api_key') or ''
        if not api_key:
            return self._write_json({'ok': False, 'error': 'no_api_key'})
        system = body.get('system') or (
            'You are a helpful assistant that reads academic papers. '
            'When writing math, use LaTeX delimiters: inline math with $...$ and display math with $$...$$. '
            'Prefer LaTeX macros (\\frac, \\int, \\sum, \\cdot, \\infty) instead of Unicode symbols. '
            'Do not escape backslashes in LaTeX. Keep text concise and well structured.'
        )
        # Optional read mode in /api/ask
        read_opts = body.get('read') or {}
        read_enabled = bool(read_opts.get('enabled', False))
        if not read_enabled:
            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': query},
            ]
            ok, out = self._openai_chat(model, messages, api_key, max_tokens=None)
            if ok:
                return self._write_json({'ok': True, 'answer': out, 'model': model})
            fallback = 'Could not contact the LLM. If offline, set your API key in Settings.'
            return self._write_json({'ok': True, 'answer': fallback, 'model': 'offline', 'offline': True, 'error': out})

        try:
            top_k_papers = max(1, int(read_opts.get('top_k_papers') or 5))
        except Exception:
            top_k_papers = 5
        try:
            chunks_per_paper = max(1, int(read_opts.get('chunks_per_paper') or 5))
        except Exception:
            chunks_per_paper = 5
        try:
            token_budget = max(1000, int(read_opts.get('token_budget') or 6000))
        except Exception:
            token_budget = 6000

        # Helper: build context from vectors under a char budget
        def _build_vectors_context(con: sqlite3.Connection, chunks_by_paper: dict[str, list[int]], max_chars: int) -> tuple[list[tuple[str,str]], int]:
            import re as _re
            out_blocks: list[tuple[str, str]] = []
            total = 0
            for paper_id, chunk_ids in chunks_by_paper.items():
                if not chunk_ids:
                    continue
                ph2 = ','.join('?' * len(chunk_ids))
                rows2 = con.execute(f"SELECT id, text FROM chunk WHERE id IN ({ph2})", list(chunk_ids)).fetchall()
                texts: list[str] = []
                for _, txt in rows2:
                    t = (txt or '').strip()
                    if not t:
                        continue
                    # Remove placeholder markers like [MATH_123]
                    t = _re.sub(r"\[MATH_\d+\]", " ", t)
                    texts.append(t)
                if not texts:
                    continue
                joined = '\n\n'.join(texts)
                if total + len(joined) > max_chars:
                    joined = joined[:max(0, max_chars-total)]
                out_blocks.append((paper_id, joined))
                total += len(joined)
                if total >= max_chars:
                    break
            return out_blocks, total

        # Helper: map internal numeric paper ids to arXiv ids
        def _to_arxiv_ids(con: sqlite3.Connection | None, ids: list[str]) -> list[str]:
            out_ids: list[str] = []
            for pid in ids:
                s = str(pid)
                if ('.' in s) or ('/' in s) or ('v' in s):
                    out_ids.append(s)
                else:
                    try:
                        if con is not None:
                            row = con.execute('SELECT arxiv_id FROM paper WHERE id=?', (int(s),)).fetchone()
                            aid = (row[0] if row and row[0] else s)
                            out_ids.append(str(aid))
                        else:
                            out_ids.append(s)
                    except Exception:
                        out_ids.append(s)
            return out_ids

        # If client provided a reading list and latex/full_paper is requested → full LaTeX mode
        try:
            client_papers = read_opts.get('paper_ids') or []
        except Exception:
            client_papers = []
        mode_req = str(read_opts.get('mode') or '').strip().lower()
        full_paper = bool(read_opts.get('full_paper', False))
        if isinstance(client_papers, list) and client_papers and (mode_req == 'latex' or full_paper):
            # Build context from full LaTeX (fallback to vectors per paper if needed)
            max_chars = token_budget * 4
            blocks: list[tuple[str, str]] = []
            total = 0
            try:
                self.refresh_db_if_changed()
                con = self.db()
            except Exception as e:
                return self._write_json({'ok': False, 'error': str(e)})
            for aid in client_papers[:max(1, top_k_papers)]:
                txt = self._read_raw_latex(str(aid))
                if not txt:
                    # Fallback: collect all chunk texts for this paper from DB
                    try:
                        row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (str(aid),)).fetchone()
                        if row:
                            pid = int(row[0])
                            texts = [ (t or '').strip() for (_cid,t) in con.execute('SELECT id,text FROM chunk WHERE paper_id=? ORDER BY id', (pid,)).fetchall() if t ]
                            txt = '\n\n'.join(texts)
                    except Exception:
                        pass
                txt = (txt or '').strip()
                if not txt:
                    continue
                if total + len(txt) > max_chars:
                    txt = txt[:max(0, max_chars-total)]
                blocks.append((str(aid), txt))
                total += len(txt)
                if total >= max_chars:
                    break
            if not blocks:
                return self._write_json({'ok': False, 'error': 'no_context'})
            parts = ["You are a careful assistant. Read the following excerpts to answer the user's question. Cite papers as [paperId].\n\n"]
            for pid, txt in blocks:
                parts.append(f"[Source {pid}]\n{txt}\n\n")
            parts.append(f"Question: {query}\n\nAnswer succinctly with citations like [paperId].")
            prompt = ''.join(parts)
            # Previews for UI
            previews = []
            try:
                for pid, txt in blocks[:max(1, top_k_papers)]:
                    previews.append({'paper': str(pid), 'chars': len(txt or ''), 'head': (txt or '')[:800]})
            except Exception:
                pass
            messages = [
                {'role':'system','content':'You answer using provided excerpts only. Use LaTeX $...$ or $$...$$ for math. Cite sources as [paperId].'},
                {'role':'user','content': prompt},
            ]
            try:
                max_out = int(os.environ.get('PROJECTSEARCHBAR_CHAT_MAX_OUT') or 600)
            except Exception:
                max_out = 600
            ok, out = self._openai_chat(model, messages, api_key, max_tokens=max_out)
            if not ok:
                offline = self._offline_summarize_blocks(blocks, query)
                return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': 'latex', 'papers': [str(p) for p,_ in blocks], 'previews': previews}, 'error': out})
            return self._write_json({'ok': True, 'answer': out, 'model': model, 'used': {'mode': 'latex', 'papers': [str(p) for p,_ in blocks], 'previews': previews}})

        # Build compact selection (rare tokens → candidate chunks → group by paper) [vectors]
        try:
            self.refresh_db_if_changed()
            con = self.db()
            # Optional client-provided paper order (arXiv ids)
            client_papers = read_opts.get('paper_ids') or []
            q_tokens = tokenize_query(query)
            if not q_tokens:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'Query produced no tokens'})
            tfq: dict[str,int] = {}
            for t in q_tokens:
                tfq[t] = tfq.get(t, 0) + 1
            placeholders = ','.join('?' * len(tfq))
            cur = con.execute(f"SELECT term, id, df FROM token WHERE term IN ({placeholders})", list(tfq.keys()))
            term_to_id: dict[str,int] = {}
            term_to_df: dict[str,int] = {}
            for term, tid, df in cur.fetchall():
                term_to_id[term] = int(tid)
                term_to_df[term] = int(df or 0)
            if not term_to_id:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No matching tokens'})
            N = con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1
            idf_by_tid: dict[int,float] = {}
            for term, cnt in tfq.items():
                tid = term_to_id.get(term)
                if tid is None:
                    continue
                df = term_to_df.get(term, 0)
                idf_by_tid[int(tid)] = math.log((1.0 + N) / (1.0 + float(df))) + 1.0
            chosen_tids = [tid for (_w, tid) in sorted([(w, t) for t, w in idf_by_tid.items()], reverse=True)[:12]] or list(idf_by_tid.keys())
            if not chosen_tids:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No informative tokens'})
            ph = ','.join('?' * len(chosen_tids))
            # Restrict to selected papers if provided
            if isinstance(client_papers, list) and client_papers:
                php = ','.join('?' * len(client_papers))
                rows = con.execute(
                    f"SELECT p.chunk_id, COUNT(*) AS m FROM posting p JOIN chunk c ON c.id=p.chunk_id JOIN paper pr ON pr.id=c.paper_id WHERE p.token_id IN ({ph}) AND pr.arxiv_id IN ({php}) GROUP BY p.chunk_id ORDER BY m DESC LIMIT ?",
                    list(chosen_tids) + [str(x) for x in client_papers] + [max(2000, top_k_papers * chunks_per_paper * 20)]
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT chunk_id, COUNT(*) AS m FROM posting WHERE token_id IN ({ph}) GROUP BY chunk_id ORDER BY m DESC LIMIT ?",
                    list(chosen_tids) + [max(2000, top_k_papers * chunks_per_paper * 20)]
                ).fetchall()
            if not rows:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No candidate chunks'})
            chunk_ids = [int(cid) for (cid, _m) in rows]
            out_rows = []
            for i in range(0, len(chunk_ids), 1000):
                sub = chunk_ids[i:i+1000]
                phc = ','.join('?' * len(sub))
                out_rows.extend(con.execute(f"SELECT id, paper_id FROM chunk WHERE id IN ({phc})", sub).fetchall())
            tmp_map: dict[str, list[int]] = {}
            paper_order: list[str] = []
            for cid, pid in out_rows:
                key = str(int(pid))
                arr = tmp_map.get(key)
                if arr is None:
                    arr = []
                    tmp_map[key] = arr
                    paper_order.append(key)
                if len(arr) < chunks_per_paper:
                    arr.append(int(cid))
            chunks_by_paper: dict[str, list[int]] = {k: v for k, v in tmp_map.items()}
            if isinstance(client_papers, list) and client_papers:
                # Keep the client's paper order but drop ones with no chunks
                ordered = []
                for aid in client_papers:
                    row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (str(aid),)).fetchone()
                    if not row:
                        continue
                    key = str(int(row[0]))
                    if key in chunks_by_paper and key not in ordered:
                        ordered.append(key)
                if not ordered:
                    ordered = paper_order[:top_k_papers]
                else:
                    ordered = ordered[:top_k_papers]
            else:
                ordered = paper_order[:top_k_papers]
        except Exception as e:
            return self._write_json({'ok': False, 'error': 'no_selection', 'detail': f'failed selection: {e}'})

        # Build vectors context and ask
        max_chars = token_budget * 4
        # Full-paper mode: if client supplied paper_ids and requested full content under budget
        if bool(read_opts.get('full_paper', False)) and isinstance(read_opts.get('paper_ids'), list) and read_opts.get('paper_ids'):
            blocks_fp: list[tuple[str, str]] = []
            total = 0
            for aid in (read_opts.get('paper_ids') or [])[:top_k_papers]:
                row = con.execute('SELECT id FROM paper WHERE arxiv_id=?', (str(aid),)).fetchone()
                if not row:
                    continue
                pid = int(row[0])
                texts: list[str] = []
                import re as _re
                for cid, txt in con.execute('SELECT id, text FROM chunk WHERE paper_id=? ORDER BY id', (pid,)).fetchall():
                    t = (txt or '').strip()
                    if not t:
                        continue
                    t = _re.sub(r"\[MATH_\d+\]", " ", t)
                    texts.append(t)
                    if (sum(len(x) for x in texts) + total) > max_chars:
                        break
                if not texts:
                    continue
                joined = '\n\n'.join(texts)
                if total + len(joined) > max_chars:
                    joined = joined[:max(0, max_chars-total)]
                blocks_fp.append((str(aid), joined))
                total += len(joined)
                if total >= max_chars:
                    break
            if not blocks_fp:
                return self._write_json({'ok': False, 'error': 'no_context'})
            parts = ["You are a careful assistant. Read the following excerpts to answer the user's question. Cite papers as [paperId].\n\n"]
            for pid, txt in blocks_fp:
                parts.append(f"[Source {pid}]\n{txt}\n\n")
            parts.append(f"Question: {query}\n\nAnswer succinctly with citations like [paperId].")
            prompt = ''.join(parts)
            try:
                max_out = int(max(200, min(800, token_budget // 2)))
            except Exception:
                max_out = 400
            previews = []
            try:
                for pid, txt in blocks_fp[:max(1, top_k_papers)]:
                    t = (txt or '')
                    previews.append({'paper': str(pid), 'chars': len(t), 'head': t[:800]})
            except Exception:
                pass
            used_papers = [str(a) for a in (read_opts.get('paper_ids') or [])[:top_k_papers]]
            messages = [
                {'role':'system','content':'You answer using provided excerpts only. Use LaTeX $...$ or $$...$$ for math. Cite sources as [paperId].'},
                {'role':'user','content': prompt},
            ]
            ok, out = self._openai_chat(model, messages, api_key, max_tokens=max_out)
            if not ok:
                offline = self._offline_summarize_blocks(blocks_fp, query)
                return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': 'vectors', 'papers': used_papers, 'chunks': {}, 'previews': previews}, 'error': out})
            return self._write_json({'ok': True, 'answer': out, 'model': model, 'used': {'mode': 'vectors', 'papers': used_papers, 'chunks': {}, 'previews': previews}})
        limited: dict[str, list[int]] = {}
        for pid in ordered:
            ids = list(chunks_by_paper.get(pid, []))
            if ids:
                limited[pid] = ids[:chunks_per_paper]
        blocks, _ = _build_vectors_context(con, limited, max_chars)
        if not blocks:
            return self._write_json({'ok': False, 'error': 'no_context'})
        parts = ["You are a careful assistant. Read the following excerpts to answer the user's question. Cite papers as [paperId].\n\n"]
        for pid, txt in blocks:
            parts.append(f"[Source {pid}]\n{txt}\n\n")
        parts.append(f"Question: {query}\n\nAnswer succinctly with citations like [paperId].")
        prompt = ''.join(parts)
        # Cap output tokens to reduce timeouts for long answers
        previews = []
        try:
            for pid, txt in blocks[:max(1, top_k_papers)]:
                t = (txt or '')
                previews.append({'paper': str(pid), 'chars': len(t), 'head': t[:800]})
        except Exception:
            pass
        used_papers_raw = [pid for pid,_ in blocks]
        used_papers = _to_arxiv_ids(con, used_papers_raw)
        messages = [
            {'role':'system','content':'You answer using provided excerpts only. Use LaTeX $...$ or $$...$$ for math. Cite sources as [paperId].'},
            {'role':'user','content': prompt},
        ]
        try:
            max_out = int(os.environ.get('PROJECTSEARCHBAR_CHAT_MAX_OUT') or max(300, min(800, token_budget // 2)))
        except Exception:
            max_out = max(300, min(800, token_budget // 2))
        ok, out = self._openai_chat(model, messages, api_key, max_tokens=max_out)
        if not ok:
            offline = self._offline_summarize_blocks(blocks, query)
            return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': 'vectors', 'papers': used_papers, 'chunks': limited, 'previews': previews}, 'error': out})
        return self._write_json({'ok': True, 'answer': out, 'model': model, 'used': {'mode': 'vectors', 'papers': used_papers, 'chunks': limited, 'previews': previews}})

    # ---- Ask-with-reading (removed) ----
    # The previous ask_read endpoint and helpers have been removed as part of a redesign.
        # End of supported LLM handlers
        # Note: Read-papers implementation removed below; leftover code is wrapped as a string.
        '''
            return self._write_json({'ok': False, 'error': 'no_api_key'})

        # Always perform a compact internal search to pick top papers/chunks from the query.
        # This avoids reliance on fragile client-side selection and ensures non-empty context.
        chunks_by_paper: dict[str, list[int]] = {}
        papers_order: list[str] = []
        try:
            # Ensure we see the latest DB after any swap
            self.refresh_db_if_changed()
            con = self.db()
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})
        try:
            # Light-weight candidate generation based on rare query tokens
            q_tokens = tokenize_query(query)
            if not q_tokens:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'Query produced no tokens'})
            # Map terms → ids/dfs
            tfq = {}
            for t in q_tokens:
                tfq[t] = tfq.get(t, 0) + 1
            placeholders = ','.join('?' * len(tfq))
            cur = con.execute(f"SELECT term, id, df FROM token WHERE term IN ({placeholders})", list(tfq.keys()))
            term_to_id = {term: int(tid) for (term, tid, _df) in cur.fetchall()}
            if not term_to_id:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No matching tokens'})
            # Build simple scoring weight per token (idf-like)
            N = con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1
            idf_by_tid = {}
            for term, cnt in tfq.items():
                tid = term_to_id.get(term)
                if tid is None:
                    continue
                df = con.execute('SELECT df FROM token WHERE id=?', (tid,)).fetchone()
                dfv = int(df[0]) if df and df[0] is not None else 0
                idf_by_tid[tid] = math.log((1.0 + N) / (1.0 + float(dfv))) + 1.0
            # Choose top tokens by rarity (cap to keep it fast)
            chosen_tids = [tid for (_w, tid) in sorted([(w, t) for t, w in idf_by_tid.items()], reverse=True)[:12]] or list(idf_by_tid.keys())
            if not chosen_tids:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No informative tokens'})
            ph = ','.join('?' * len(chosen_tids))
            rows = con.execute(
                f"SELECT chunk_id, COUNT(*) AS m FROM posting WHERE token_id IN ({ph}) GROUP BY chunk_id ORDER BY m DESC LIMIT ?",
                list(chosen_tids) + [max(800, top_k_papers * chunks_per_paper * 30)]
            ).fetchall()
            if not rows:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No candidate chunks'})
            # Fetch chunk → paper and select top per paper
            chunk_ids = [int(cid) for (cid, _m) in rows]
            out_rows = []
            for i in range(0, len(chunk_ids), 1000):
                sub = chunk_ids[i:i+1000]
                phc = ','.join('?' * len(sub))
                out_rows.extend(con.execute(f"SELECT id, paper_id FROM chunk WHERE id IN ({phc})", sub).fetchall())
            tmp_map: dict[str, list[int]] = {}
            for cid, pid in out_rows:
                key = str(int(pid))
                arr = tmp_map.get(key)
                if arr is None:
                    arr = []
                    tmp_map[key] = arr
                if len(arr) < chunks_per_paper:
                    arr.append(int(cid))
            # Preserve a stable order by first appearance
            papers_order = list(tmp_map.keys())[:top_k_papers]
            chunks_by_paper = {k: v for k, v in tmp_map.items()}
            # If client provided a selection, union it (capped by chunks_per_paper)
            if client_chunks:
                for pid, arr in client_chunks.items():
                    lst = chunks_by_paper.setdefault(str(pid), [])
                    for cid in (arr or []):
                        if len(lst) < chunks_per_paper and int(cid) not in lst:
                            lst.append(int(cid))
                # Merge paper order with any client-preferred order
                merged = []
                for pid in (client_order or []):
                    sp = str(pid)
                    if sp in chunks_by_paper and sp not in merged:
                        merged.append(sp)
                for pid in papers_order:
                    if pid not in merged:
                        merged.append(pid)
                papers_order = merged[:top_k_papers]
        except Exception as e:
            return self._write_json({'ok': False, 'error': 'no_selection', 'detail': f'auto-select failed: {e}'})
            try:
                # Ensure we see the latest DB after any swap
                self.refresh_db_if_changed()
                con = self.db()
            except Exception as e:
                return self._write_json({'ok': False, 'error': str(e)})
            try:
                # Light-weight candidate generation based on rare query tokens
                q_tokens = tokenize_query(query)
                if not q_tokens:
                    return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'Query produced no tokens'})
                # Map terms → ids/dfs
                tfq = {}
                for t in q_tokens:
                    tfq[t] = tfq.get(t, 0) + 1
                placeholders = ','.join('?' * len(tfq))
                cur = con.execute(f"SELECT term, id, df FROM token WHERE term IN ({placeholders})", list(tfq.keys()))
                term_to_id = {term: int(tid) for (term, tid, _df) in cur.fetchall()}
                if not term_to_id:
                    return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No matching tokens'})
                # Build simple scoring weight per token (idf-like)
                N = con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1
                idf_by_tid = {}
                for term, cnt in tfq.items():
                    tid = term_to_id.get(term)
                    if tid is None:
                        continue
                    df = con.execute('SELECT df FROM token WHERE id=?', (tid,)).fetchone()
                    dfv = int(df[0]) if df and df[0] is not None else 0
                    idf_by_tid[tid] = math.log((1.0 + N) / (1.0 + float(dfv))) + 1.0
                # Choose top tokens by rarity (cap to keep it fast)
                chosen_tids = [tid for (_w, tid) in sorted([(w, t) for t, w in idf_by_tid.items()], reverse=True)[:12]] or list(idf_by_tid.keys())
                if not chosen_tids:
                    return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No informative tokens'})
                ph = ','.join('?' * len(chosen_tids))
                rows = con.execute(f"SELECT chunk_id, COUNT(*) AS m FROM posting WHERE token_id IN ({ph}) GROUP BY chunk_id ORDER BY m DESC LIMIT ?", list(chosen_tids) + [max(2000, top_k_papers * chunks_per_paper * 20)]).fetchall()
                if not rows:
                    return self._write_json({'ok': False, 'error': 'no_selection', 'detail': 'No candidate chunks'})
                # Fetch chunk → paper and select top per paper
                chunk_ids = [int(cid) for (cid, _m) in rows]
                # Batch fetch chunk metadata
                out_rows = []
                for i in range(0, len(chunk_ids), 1000):
                    sub = chunk_ids[i:i+1000]
                    phc = ','.join('?' * len(sub))
                    out_rows.extend(con.execute(f"SELECT id, paper_id FROM chunk WHERE id IN ({phc})", sub).fetchall())
                # Build selection maps
                chunks_by_paper = {}
                papers_order = []
                for cid, pid in out_rows:
                    pid = str(int(pid))
                    arr = chunks_by_paper.get(pid)
                    if arr is None:
                        arr = []
                        chunks_by_paper[pid] = arr
                        papers_order.append(pid)
                    if len(arr) < chunks_per_paper:
                        arr.append(int(cid))
                # Convert internal numeric paper ids to arXiv ids for ordering output later
                # Note: we keep numeric keys in chunks_by_paper since downstream reads chunk ids directly
            except Exception as e:
                return self._write_json({'ok': False, 'error': 'no_selection', 'detail': f'auto-select failed: {e}'})

        # Limit papers and chunks per paper
        limited_chunks: dict[str, list[int]] = {}
        ordered = papers_order or list(chunks_by_paper.keys())
        for pid in ordered[:top_k_papers]:
            ids = list(chunks_by_paper.get(pid, []))
            if ids:
                limited_chunks[pid] = ids[:chunks_per_paper]
        # If client provided zero chunk ids (mismatch or bug), run a compact internal search
        try:
            provided_chunks = sum(len(v) for v in limited_chunks.values())
        except Exception:
            provided_chunks = 0
        if provided_chunks == 0:
            try:
                self.refresh_db_if_changed()
                con = self.db()
                # Generate selection purely server-side
                q_tokens = tokenize_query(query)
                tfq = {}
                for t in q_tokens:
                    tfq[t] = tfq.get(t, 0) + 1
                if tfq:
                    placeholders = ','.join('?' * len(tfq))
                    cur = con.execute(f"SELECT term, id, df FROM token WHERE term IN ({placeholders})", list(tfq.keys()))
                    term_to_id = {term: int(tid) for (term, tid, _df) in cur.fetchall()}
                    if term_to_id:
                        N = con.execute('SELECT COUNT(1) FROM chunk').fetchone()[0] or 1
                        idf_by_tid = {}
                        for term, cnt in tfq.items():
                            tid = term_to_id.get(term)
                            if tid is None: continue
                            df = con.execute('SELECT df FROM token WHERE id=?', (tid,)).fetchone()
                            dfv = int(df[0]) if df and df[0] is not None else 0
                            idf_by_tid[tid] = math.log((1.0 + N) / (1.0 + float(dfv))) + 1.0
                        chosen_tids = [tid for (_w, tid) in sorted([(w, t) for t, w in idf_by_tid.items()], reverse=True)[:12]] or list(idf_by_tid.keys())
                        if chosen_tids:
                            ph = ','.join('?' * len(chosen_tids))
                            rows = con.execute(f"SELECT chunk_id, COUNT(*) AS m FROM posting WHERE token_id IN ({ph}) GROUP BY chunk_id ORDER BY m DESC LIMIT ?", list(chosen_tids) + [max(2000, top_k_papers * chunks_per_paper * 20)]).fetchall()
                            chunk_ids = [int(cid) for (cid, _m) in rows]
                            # Map chunk -> paper
                            ordered = []
                            tmp_map: dict[str, list[int]] = {}
                            for i in range(0, len(chunk_ids), 1000):
                                sub = chunk_ids[i:i+1000]
                                phc = ','.join('?' * len(sub))
                                for cid, pid in con.execute(f"SELECT id, paper_id FROM chunk WHERE id IN ({phc})", sub).fetchall():
                                    key = str(int(pid))
                                    if key not in tmp_map:
                                        tmp_map[key] = []
                                        ordered.append(key)
                                    if len(tmp_map[key]) < chunks_per_paper:
                                        tmp_map[key].append(int(cid))
                            # Adopt new selection (use numeric paper ids; we map to arXiv in output)
                            limited_chunks = {k: v for k, v in tmp_map.items()}
                            ordered = ordered[:top_k_papers]
            except Exception:
                pass

        # Approx chars budget ~ tokens*4
        max_chars = token_budget * 4
        context_blocks: list[tuple[str, str]] = []
        try:
            con = self.db()
        except Exception as e:
            return self._write_json({'ok': False, 'error': str(e)})
        mode_used = mode
        if mode == 'vectors':
            blocks, _ = self._read_vectors_texts(con, limited_chunks, max_chars)
            context_blocks = blocks
        else:
            # Raw LaTeX: read full paper(s), then trim to budget evenly
            per_paper_chars = max(1000, max_chars // max(1, len(ordered)))
            for pid in ordered[:top_k_papers]:
                txt = self._read_raw_latex(pid)
                txt = (txt or '').strip()
                if not txt:
                    continue
                context_blocks.append((pid, txt[:per_paper_chars]))
            # Fallback: if LaTeX not found locally, fall back to vectors mode
            if not context_blocks:
                blocks_fallback, _ = self._read_vectors_texts(con, limited_chunks, max_chars)
                if blocks_fallback:
                    context_blocks = blocks_fallback
                    mode_used = 'vectors'

        if not context_blocks:
            return self._write_json({'ok': False, 'error': 'no_context'})

        # Build prompt
        parts = ["You are a careful assistant. Read the following excerpts to answer the user's question. Cite papers as [paperId].\n\n"]
        for pid, txt in context_blocks:
            parts.append(f"[Source {pid}]\n{txt}\n\n")
        parts.append(f"Question: {query}\n\nAnswer succinctly with citations like [paperId].")
        prompt = ''.join(parts)

        # Try LLM; if unavailable or times out, fall back to offline summary
        # Limit generated tokens for responsiveness on larger prompts
        try:
            max_out = int(max(200, min(800, token_budget // 2)))
        except Exception:
            max_out = 400
        # Prepare small debug previews for UI/banner (first ~120 chars per block)
        previews = []
        try:
            for pid, txt in context_blocks[:max(1, top_k_papers)]:
                t = (txt or '')
                previews.append({'paper': str(pid), 'chars': len(t), 'head': t[:800]})
        except Exception:
            pass

        # Helper to map internal numeric paper ids to arXiv ids for user-facing sources
        def to_arxiv_list(ids: list[str]) -> list[str]:
            out_ids: list[str] = []
            try:
                con2 = self.db()
            except Exception:
                con2 = None
            for pid in ids:
                s = str(pid)
                if ('.' in s) or ('/' in s) or ('v' in s):
                    out_ids.append(s)
                else:
                    try:
                        if con2 is not None:
                            row = con2.execute('SELECT arxiv_id FROM paper WHERE id=?', (int(s),)).fetchone()
                            aid = (row[0] if row and row[0] else s)
                            out_ids.append(str(aid))
                        else:
                            out_ids.append(s)
                    except Exception:
                        out_ids.append(s)
            return out_ids

        ok, out = self._openai_chat(model, [
            {'role':'system','content':'You answer using provided excerpts only. Use LaTeX $...$ or $$...$$ for math. Cite sources as [paperId].'},
            {'role':'user','content': prompt},
        ], api_key, max_tokens=max_out)
        if not ok:
            offline = self._offline_summarize_blocks(context_blocks, query)
            used_papers_raw = [pid for pid,_ in context_blocks]
            used_papers = to_arxiv_list(used_papers_raw)
            return self._write_json({'ok': True, 'answer': offline, 'model': 'offline', 'offline': True, 'used': {'mode': mode_used, 'papers': used_papers, 'chunks': limited_chunks, 'previews': previews}, 'error': out})

        used_papers_raw = [pid for pid,_ in context_blocks]
        used_papers = to_arxiv_list(used_papers_raw)
        return self._write_json({'ok': True, 'answer': out, 'model': model, 'used': {'mode': mode_used, 'papers': used_papers, 'chunks': limited_chunks, 'previews': previews}})
        '''
