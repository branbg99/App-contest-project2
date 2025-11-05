type SearchResult = {
  paperId: string;
  chunkId: number;
  kind: string;
  text: string;
  score: number;
};

const qEl = document.getElementById('query') as HTMLInputElement;
const btnSearch = document.getElementById('btnSearch') as HTMLButtonElement;
const btnBuild = document.getElementById('btnBuild') as HTMLButtonElement | null;
const topKEl = document.getElementById('topK') as HTMLInputElement;
const kindEl = document.getElementById('kind') as HTMLSelectElement;
const perPaperEl = document.getElementById('perPaperK') as HTMLInputElement;
const resultsEl = document.getElementById('results') as HTMLOListElement;
const statusEl = document.getElementById('status') as HTMLSpanElement;
const bannerEl = document.getElementById('banner') as HTMLDivElement;
const dbInfoEl = document.getElementById('dbInfo') as HTMLDivElement;
const btnLaunch = document.getElementById('btnLaunch') as HTMLButtonElement | null;

async function diagnose() {
  try {
    const r = await fetch('/api/diagnose');
    const j = await r.json();
    if (j.ok) {
      bannerEl.textContent = `DB: ${j.db}`;
      dbInfoEl.textContent = `tokens: ${j.stats.tokens}, papers: ${j.stats.papers}, chunks: ${j.stats.chunks}`;
    } else {
      bannerEl.textContent = 'DB not ready';
      dbInfoEl.textContent = j.stats?.error || '';
    }
  } catch {
    bannerEl.textContent = 'Diagnose failed';
  }
}

async function search() {
  const query = (qEl?.value || '').trim();
  const top_k = parseInt(topKEl?.value || '20', 10);
  const kind = kindEl?.value || 'both';
  const per_paper_k = parseInt(perPaperEl?.value || '1', 10);
  if (!query) return;
  statusEl.textContent = 'Searching...';
  btnSearch.disabled = true;
  resultsEl.innerHTML = '';
  try {
    const r = await fetch('/api/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k, kind, per_paper_k })
    });
    const j = await r.json();
    if (!j.ok) {
      statusEl.textContent = j.error === 'busy' ? 'Index is building; try again shortly.' : (j.error || 'Error searching');
      return;
    }
    statusEl.textContent = `Found ${j.results.length} (scanned ${j.scanned})`;
    (j.results as SearchResult[]).forEach(rec => {
      const li = document.createElement('li');
      li.className = 'result';
      const preview = (rec.text || '').replace(/\s+/g, ' ').slice(0, 400);
      li.innerHTML = `
        <div><strong>${rec.paperId}</strong> — <span class="muted">${rec.kind}</span> — score ${rec.score.toFixed(4)}</div>
        <div class="muted">${preview}</div>
        <div>
          <a target="_blank" href="https://arxiv.org/abs/${rec.paperId}">arXiv</a>
          · <a target="_blank" href="https://arxiv.org/pdf/${rec.paperId}.pdf">PDF</a>
        </div>
      `;
      resultsEl.appendChild(li);
    });
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Error searching';
  } finally {
    btnSearch.disabled = false;
  }
}

async function buildIndex() {
  if (!btnBuild) return;
  btnBuild.disabled = true;
  const orig = btnBuild.textContent || 'Build Index';
  btnBuild.textContent = 'Building...';
  try {
    const r = await fetch('/api/index/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reset: false, mode: 'index-only' }) });
    const j = await r.json();
    if (j.ok && j.started) {
      const totalFromStart = (j && j.papers_total) || 0;
      bannerEl.textContent = totalFromStart ? `Indexing (0/${totalFromStart})...` : 'Index build started (runs in background).';
      // Per-run timestamp to ignore stale status from older builds
      const runStart = Date.now() / 1000;
      const buildId = j.build_id || null;
      // Monotonic smoothing to avoid flashing counters
      const MAX_PAPER_STEP = 1000; // cap visible step per poll for readability
      const last = { done: 0, total: 0, chunks: 0, postings: 0, normSeen: 0, normTotal: 0 };
      let baselineSet = false;
      let tries = 0;
      const t = setInterval(async () => {
        tries++;
        try {
          const rs = await fetch('/api/index/status');
          const s = await rs.json();
          const ts = (s && typeof s.ts === 'number') ? s.ts : 0;
          if (ts && ts < runStart) {
            // Ignore stale status from previous runs
            return;
          }
          if (buildId && s.build_id !== buildId) {
            // Ignore other concurrent/previous builds
            return;
          }
          if (s.state === 'vectorizing') bannerEl.textContent = 'Vectorizing (scanning files)...';
          else if (s.state === 'indexing') {
            let suffix = '';
            if (s.papers_total && (s.papers_done !== undefined || (s as any).display_papers_done !== undefined)) {
              const total = Math.max(last.total || 0, s.papers_total || 0);
              let doneRaw = ((s as any).display_papers_done !== undefined ? (s as any).display_papers_done : s.papers_done) || 0;
              if (!baselineSet) { doneRaw = 0; baselineSet = true; }
              const proposed = Math.min(Math.max(last.done || 0, doneRaw), total);
              const limited = proposed > (last.done || 0)
                ? Math.min(proposed, (last.done || 0) + MAX_PAPER_STEP)
                : proposed;
              last.total = total; last.done = limited;
              suffix = ` (${limited}/${total})`;
            }
            if (s.phase === 'norms' && s.total_norms) {
              const tot = Math.max(last.normTotal || 0, s.total_norms || 0);
              const dn = Math.min(Math.max(last.normSeen || 0, s.done_norms || 0), tot);
              last.normTotal = tot; last.normSeen = dn;
              suffix = ` (norms ${dn}/${tot})`;
            }
            else if (s.phase === 'norms' && (s.postings_seen || s.postings_total)) {
              const tot = Math.max(last.normTotal || 0, s.postings_total || 0);
              const seen = Math.min(Math.max(last.normSeen || 0, s.postings_seen || 0), tot || Infinity);
              last.normTotal = tot; last.normSeen = seen;
              suffix = ` (norms postings ${seen}${tot?`/${tot}`:''})`;
            }
            else if (s.chunks || s.postings) {
              const chunks = Math.max(last.chunks || 0, s.chunks || 0);
              const posts = Math.max(last.postings || 0, s.postings || 0);
              last.chunks = chunks; last.postings = posts;
              suffix += ` (chunks ${chunks}${posts?`, postings ${posts}`:''})`;
            }
            bannerEl.textContent = 'Indexing (writing DB)...' + suffix;
          }
          else if (s.state === 'swapping') bannerEl.textContent = 'Finalizing (swapping new DB)...';
          else if (s.state === 'done') { bannerEl.textContent = 'Index build complete.'; clearInterval(t); try{diagnose();}catch(e){} btnBuild.disabled = false; btnBuild.textContent = orig; }
          else if (s.state === 'error') { bannerEl.textContent = `Index error: ${s.error || ''}`; clearInterval(t); btnBuild.disabled = false; btnBuild.textContent = orig; }
        } catch {}
        if (tries > 900) clearInterval(t);
      }, 1000);
    } else {
      bannerEl.textContent = j && j.reason === 'busy' ? 'A build is already running.' : 'Failed to start build';
      btnBuild.disabled = false;
      btnBuild.textContent = orig;
    }
  } catch (e) {
    console.error(e);
    bannerEl.textContent = 'Error starting build';
    btnBuild.disabled = false;
    btnBuild.textContent = orig;
  }
}

btnSearch?.addEventListener('click', search);
qEl?.addEventListener('keydown', (e: KeyboardEvent) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'enter') search();
});
btnBuild?.addEventListener('click', buildIndex);
btnLaunch?.addEventListener('click', async () => {
  try {
    bannerEl.textContent = 'Opening in browser...';
    const r = await fetch('/api/launch', { method: 'POST' });
    const j = await r.json();
    bannerEl.textContent = j.ok ? 'Launched in default browser.' : (j.error || 'Launch failed');
  } catch {
    bannerEl.textContent = 'Launch failed';
  }
});
diagnose();
