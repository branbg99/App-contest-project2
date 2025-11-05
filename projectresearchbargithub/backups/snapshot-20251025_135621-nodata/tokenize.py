from __future__ import annotations

import re
from typing import List, Tuple


# Public regex for finding inline/display math segments
MATH_PATTERN = re.compile(
    r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[\s\S]*?\$",
    re.MULTILINE,
)


def strip_math_delims(s: str) -> str:
    if s.startswith('$$') and s.endswith('$$'):
        return s[2:-2]
    if s.startswith('$') and s.endswith('$'):
        return s[1:-1]
    if s.startswith('\\[') and s.endswith('\\]'):
        return s[2:-2]
    if s.startswith('\\(') and s.endswith('\\)'):
        return s[2:-2]
    return s


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


# Canonicalize common math variant commands
CANON_MATH = {
    r'\\varepsilon': r'\\epsilon',
    r'\\varphi': r'\\phi',
    r'\\vartheta': r'\\theta',
    r'\\varsigma': r'\\sigma',
}

# English word -> LaTeX command hints (helps text queries match math tokens)
WORD_TO_LATEX = {
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

# Unicode and symbol → LaTeX expansions (query-time helpers)
UNICODE_TO_LATEX: dict[str, tuple[str, ...] | str] = {
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
    '≈': r'\\approx',
    '≪': r'\\ll',
    '≫': r'\\gg',
    '∑': r'\\sum',
    '→': r'\\to',
    '←': r'\\leftarrow',
    '↔': r'\\leftrightarrow',
    '⇒': r'\\implies',
    '⇔': r'\\iff',
    '⋅': r'\\cdot',
    '·': r'\\cdot',
    '∗': r'\\ast',
    '‖': (r'\\Vert', r'\\|'),  # bridge both forms of norms
    '∥': (r'\\Vert', r'\\|'),
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

# Greek letters (Unicode) → LaTeX command
GREEK_TO_LATEX: dict[str, str] = {
    'α': r'\\alpha','β': r'\\beta','γ': r'\\gamma','δ': r'\\delta','ε': r'\\epsilon','ζ': r'\\zeta','η': r'\\eta','θ': r'\\theta','ι': r'\\iota','κ': r'\\kappa','λ': r'\\lambda','μ': r'\\mu','ν': r'\\nu','ξ': r'\\xi','ο': 'o','π': r'\\pi','ρ': r'\\rho','σ': r'\\sigma','τ': r'\\tau','υ': r'\\upsilon','φ': r'\\phi','χ': r'\\chi','ψ': r'\\psi','ω': r'\\omega',
    'Α': r'\\Alpha','Β': r'\\Beta','Γ': r'\\Gamma','Δ': r'\\Delta','Ε': r'\\Epsilon','Ζ': r'\\Zeta','Η': r'\\Eta','Θ': r'\\Theta','Ι': r'\\Iota','Κ': r'\\Kappa','Λ': r'\\Lambda','Μ': r'\\Mu','Ν': r'\\Nu','Ξ': r'\\Xi','Ο': 'O','Π': r'\\Pi','Ρ': r'\\Rho','Σ': r'\\Sigma','Τ': r'\\Tau','Υ': r'\\Upsilon','Φ': r'\\Phi','Χ': r'\\Chi','Ψ': r'\\Psi','Ω': r'\\Omega',
}

# Blackboard bold common sets (Unicode) → (command, arg)
BLACKBOARD_TO_LATEX: dict[str, Tuple[str, str]] = {
    'ℝ': (r'\\mathbb', 'r'),
    'ℂ': (r'\\mathbb', 'c'),
    'ℤ': (r'\\mathbb', 'z'),
    'ℚ': (r'\\mathbb', 'q'),
    'ℕ': (r'\\mathbb', 'n'),
    'ℙ': (r'\\mathbb', 'p'),
}


def tokenize_text(s: str) -> List[str]:
    # Lowercase and keep broad alphanumerics/underscore; strip punctuation/backslashes
    s = (s or '').lower()
    s = s.replace('\u2212', '-')  # normalize Unicode minus
    s = re.sub(r"[^0-9A-Za-z_\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF]+", " ", s)
    toks = [t for t in s.split() if t]
    out: List[str] = []
    for t in toks:
        if t.startswith('\\'):
            out.append(t)
        else:
            out.append(_singularize(t))
    return out


def tokenize_math(latex: str) -> List[str]:
    s = strip_math_delims(latex or '')
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


def extract_math_segments(q: str) -> List[str]:
    return [m.group(0) for m in MATH_PATTERN.finditer(q or '')]


def _expand_and_normalize_tokens(tokens: List[str], raw_text: str | None = None) -> List[str]:
    out: List[str] = []
    seen_once: set[str] = set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith('math_') and t[5:].isdigit():
            i += 1; continue
        if t.startswith('\\') and t in CANON_MATH:
            t = CANON_MATH[t]
        out.append(t)
        i += 1

    extras: set[str] = set()
    # Gentle singularization for non-LaTeX tokens and add word→LaTeX hints
    for t in out:
        if not t.startswith('\\'):
            s = _singularize(t)
            if s and s != t:
                extras.add(s)
        for la in WORD_TO_LATEX.get(t, []):
            extras.add(la)
    # ASCII operator bridges
    for idx in range(len(out) - 1):
        a, b = out[idx], out[idx + 1]
        if a == '-' and b == '>':
            extras.add(r'\\to')
        if a == '=' and b == '>':
            extras.add(r'\\implies')
        if a == '<' and b == '-':
            extras.add(r'\\leftarrow')
        if a == '<' and b == '>':
            extras.add(r'\\leftrightarrow')
        if a == '<' and b == '=':
            extras.add(r'\\leq')
        if a == '>' and b == '=':
            extras.add(r'\\geq')
        if a == '!' and b == '=':
            extras.add(r'\\neq')
        if a == '<' and b == '<':
            extras.add(r'\\ll')
        if a == '>' and b == '>':
            extras.add(r'\\gg')

    # Phrase-based helpers
    if raw_text:
        rt = raw_text.lower()
        if 'for all' in rt:
            extras.add(r'\\forall')
        if 'there exists' in rt:
            extras.add(r'\\exists')
        if 'if and only if' in rt:
            extras.add(r'\\iff')
        # ASCII operator detection in plain text (outside math)
        if '->' in rt:
            extras.add(r'\\to')
        if '=>' in rt:
            extras.add(r'\\implies')
        if '<->' in rt or '\u2194' in rt:
            extras.add(r'\\leftrightarrow')
        if '<=' in rt:
            extras.add(r'\\leq')
        if '>=' in rt:
            extras.add(r'\\geq')
        if '!=' in rt:
            extras.add(r'\\neq')
        if '<<' in rt:
            extras.add(r'\\ll')
        if '>>' in rt:
            extras.add(r'\\gg')

    for la in sorted(extras):
        if la not in seen_once:
            out.append(la)
            seen_once.add(la)
    return out


def tokenize_query(q: str) -> List[str]:
    # Base tokens from text and math
    text_tokens = tokenize_text(q)
    math_tokens: List[str] = []
    for seg in extract_math_segments(q):
        math_tokens.extend(tokenize_math(seg))

    # Unicode symbol expansions + Greek/blackboard letters
    extra: List[str] = []
    seen: set[str] = set()
    for ch in q:
        v = UNICODE_TO_LATEX.get(ch)
        if v:
            if isinstance(v, tuple):
                for tok in v:
                    if tok not in seen:
                        extra.append(tok); seen.add(tok)
            else:
                if v not in seen:
                    extra.append(v); seen.add(v)
        g = GREEK_TO_LATEX.get(ch)
        if g and g not in seen:
            extra.append(g); seen.add(g)
        bb = BLACKBOARD_TO_LATEX.get(ch)
        if bb:
            a, b = bb
            if a not in seen:
                extra.append(a); seen.add(a)
            if b not in seen:
                extra.append(b); seen.add(b)

    # Merge and normalize
    combined = text_tokens + math_tokens + extra
    combined = _expand_and_normalize_tokens(combined, raw_text=q)
    return combined


def debug_tokenize(q: str) -> dict:
    text_tokens = tokenize_text(q)
    math_segments = extract_math_segments(q)
    math_tokens: List[str] = []
    for seg in math_segments:
        math_tokens.extend(tokenize_math(seg))
    final_tokens = tokenize_query(q)
    return {
        'text_tokens': text_tokens,
        'math_segments': math_segments,
        'math_tokens': math_tokens,
        'final': final_tokens,
    }
