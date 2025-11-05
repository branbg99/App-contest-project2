(() => {
  const qEl = document.getElementById('query');
  const btnSearch = document.getElementById('btnSearch');
  const btnSettings = document.getElementById('btnSettings');
  const modal = document.getElementById('modalSettings');
  const btnCloseSettings = document.getElementById('btnCloseSettings');
  const btnStartBuild = document.getElementById('btnStartBuild');
  const optLowMem = document.getElementById('optLowMem');
  const optPostBatch = document.getElementById('optPostBatch');
  const optCommitPapers = document.getElementById('optCommitPapers');
  const optMode = document.getElementById('optMode');
  const optLimit = document.getElementById('optLimit');
  const optShards = document.getElementById('optShards');
  const optShardWorkers = document.getElementById('optShardWorkers');
  const resultsEl = document.getElementById('results');
  const statusEl = document.getElementById('status');
  const btnCopyStatus = document.getElementById('btnCopyStatus');
  const bannerEl = document.getElementById('banner');
  const dbInfoEl = document.getElementById('dbInfo');
  const btnLaunch = document.getElementById('btnLaunch');

  async function diagnose() {
    try {
      const r = await fetch('/api/diagnose');
      const j = await r.json();
      if (j.ok) {
        bannerEl.textContent = `DB: ${j.db}`;
        dbInfoEl.textContent = `tokens: ${j.stats.tokens}, papers: ${j.stats.papers}, chunks: ${j.stats.chunks}`;
      } else {
        bannerEl.textContent = 'DB not ready';
        dbInfoEl.textContent = (j.stats && j.stats.error) || '';
      }
    } catch (e) {
      bannerEl.textContent = 'Diagnose failed';
    }
  }

  async function search() {
    const query = (qEl && qEl.value || '').trim();
    // Always pull latest saved settings from server (pywebview can isolate localStorage)
    let saved = {};
    try {
      const r = await fetch('/api/search/settings');
      const j = await r.json();
      if (j && j.ok && j.settings) saved = j.settings;
    } catch (e) {}
    // Merge with local defaults as fallback
    try {
      const loc = JSON.parse(localStorage.getItem('PSB_SEARCH_OPTS')||'{}');
      saved = Object.assign({}, loc, saved);
      localStorage.setItem('PSB_SEARCH_OPTS', JSON.stringify(saved));
    } catch (e) {}
    // Apply in-UI overrides if present (lets you try settings without Save)
    try {
      const elKind = document.getElementById('optSearchKind');
      const elPerPaper = document.getElementById('optSearchPerPaper');
      const elMaxRes = document.getElementById('optSearchMax');
      const elScoring = document.getElementById('optScoring');
      const elBm25K1 = document.getElementById('optBm25K1');
      const elBm25B  = document.getElementById('optBm25B');
      const elSvdEn  = document.getElementById('optSvdEnable');
      const elSvdTop = document.getElementById('optSvdTopN');
      const elSvdA   = document.getElementById('optSvdAlpha');
      const elTwoEn  = document.getElementById('optTwoPassEnable');
      const elTwoTop = document.getElementById('optTwoPassTopN');
      if (elKind && elKind.value) saved.kind = elKind.value;
      if (elPerPaper && elPerPaper.value) saved.perPaper = parseInt(elPerPaper.value, 10);
      if (elMaxRes && elMaxRes.value) saved.maxResults = parseInt(elMaxRes.value, 10);
      if (elScoring && elScoring.value) saved.scoring = elScoring.value;
      if (elBm25K1 && elBm25K1.value) saved.bm25K1 = parseFloat(elBm25K1.value);
      if (elBm25B && elBm25B.value) saved.bm25B = parseFloat(elBm25B.value);
      if (elSvdEn && 'checked' in elSvdEn) saved.svdEnable = !!elSvdEn.checked;
      if (elSvdTop && elSvdTop.value) saved.svdTopN = parseInt(elSvdTop.value, 10);
      if (elSvdA && elSvdA.value) saved.svdAlpha = parseFloat(elSvdA.value);
      if (elTwoEn && 'checked' in elTwoEn) saved.twoPassEnable = !!elTwoEn.checked;
      if (elTwoTop && elTwoTop.value) saved.twoPassTopN = parseInt(elTwoTop.value, 10);
    } catch (e) {}

    const kind = (saved && saved.kind) || 'both';
    const DEFAULT_MAX = 5000;
    const per_paper_k = Number.isFinite(saved.perPaper) && saved.perPaper > 0 ? saved.perPaper : 1;
    const top_k = Number.isFinite(saved.maxResults) && saved.maxResults > 0 ? saved.maxResults : DEFAULT_MAX;
    const max_candidates = top_k;
    const scoring = (saved && saved.scoring) || 'tfidf';
    const twoPassEnable = !!(saved && saved.twoPassEnable);
    const twoPassTopN = (Number.isFinite(saved && saved.twoPassTopN) && saved.twoPassTopN > 0) ? saved.twoPassTopN : 2000;
    const bm25 = (scoring === 'bm25') ? {
      k1: (typeof saved.bm25K1 === 'number' && saved.bm25K1 > 0) ? saved.bm25K1 : 1.2,
      b: (typeof saved.bm25B === 'number' && saved.bm25B >= 0 && saved.bm25B <= 1) ? saved.bm25B : 0.75,
    } : null;
    if (!query) return;
    statusEl.textContent = 'Searching...';
    btnSearch.disabled = true;
    resultsEl.innerHTML = '';
    try {
      const svd = (saved && saved.svdEnable) ? {
        enabled: true,
        topN: (Number.isFinite(saved.svdTopN) && saved.svdTopN > 0) ? saved.svdTopN : 2000,
        alpha: (typeof saved.svdAlpha === 'number' && saved.svdAlpha >= 0 && saved.svdAlpha <= 1) ? saved.svdAlpha : 0.5
      } : null;
      const payload = { query, top_k, kind, per_paper_k, max_candidates, scoring };
      if (twoPassEnable) payload.two_pass = { enabled: true, primary: 'bm25', secondary: 'cosine', topN: twoPassTopN };
      if (bm25) payload.bm25 = bm25;
      if (svd) payload.svd = svd;
      const r = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const j = await r.json();
      if (!j.ok) {
        if (j.error === 'busy') statusEl.textContent = 'Index is building; try again shortly.';
        else if (j.error === 'timeout') statusEl.textContent = 'Search timed out; try a more specific query or lower limits.';
        else statusEl.textContent = j.error || 'Error searching';
        return;
      }
      const limitedNote = (top_k && top_k < 1000000) ? ` — showing up to ${top_k}` : '';
      const scoringName = twoPassEnable ? 'Two-pass (BM25 → cosine)' : ((scoring === 'bm25') ? 'BM25' : 'TF–IDF (cosine)');
      let extras = '';
      const wantSvd = !!(svd && svd.enabled);
      // Prefer server-reported topN/alpha if present
      const tnSrv = (typeof j.svd_topN === 'number') ? j.svd_topN : null;
      const aSrv = (typeof j.svd_alpha === 'number') ? j.svd_alpha : null;
      const tn = tnSrv != null ? tnSrv : ((svd && typeof svd.topN === 'number') ? svd.topN : null);
      const a  = aSrv  != null ? aSrv  : ((svd && typeof svd.alpha === 'number') ? svd.alpha : null);
      const hasSrv = (typeof j.svd_applied !== 'undefined');
      if (wantSvd || (hasSrv && j.svd_applied)) {
        // Always show an SVD line if either requested or confirmed by server
        const tnTxt = (tn != null) ? tn : '–';
        const aTxt  = (a  != null) ? a  : '–';
        extras = ` — SVD re-rank: topN=${tnTxt}, α=${aTxt}`;
        if (hasSrv) {
          if (j.svd_applied) {
            const used = (typeof j.svd_used_count === 'number') ? j.svd_used_count : '–';
            extras += ` (applied, used=${used})`;
          } else if (wantSvd && j.svd_assets === false) {
            // Only show 'assets missing' if user requested SVD
            extras += ` (assets missing)`;
          } else {
            // requested but not applied, or server declined silently
            if (wantSvd) extras += ` (not applied)`;
          }
        }
      }
      const statusText = `Found ${j.results.length} (scanned ${j.scanned})${limitedNote} — Ranking by: ${scoringName}${extras}`;
      statusEl.textContent = statusText;
      try {
        const termBody = document.getElementById('termBody');
        if (termBody && statusText) {
          const d=document.createElement('div'); d.className='line';
          const ts=document.createElement('span'); ts.className='ts'; ts.textContent=new Date().toLocaleTimeString();
          const sp=document.createElement('span'); sp.textContent = statusText;
          d.appendChild(ts); d.appendChild(sp); termBody.appendChild(d); termBody.scrollTop = termBody.scrollHeight;
        }
      } catch(e){}
      const paperIds = [];
      let idx = 0; // preserve original order for reset sorting
      let maxBM25 = 0;
      try { maxBM25 = j.results.reduce((m, rec) => Math.max(m, Number(rec && rec.bm25 || 0)), 0); } catch(e){ maxBM25 = 0; }
      if (!(maxBM25 > 0)) maxBM25 = 1;
      j.results.forEach((rec) => {
        const li = document.createElement('li');
        li.className = 'result';
        li.setAttribute('data-paper-id', rec.paperId);
        const preview = (rec.text || '').replace(/\s+/g, ' ').slice(0, 400);
        const cosine = (typeof rec.cosine === 'number') ? rec.cosine : (rec.scoring === 'bm25' ? 0 : (rec.score || 0));
        const bm25v = (typeof rec.bm25 === 'number') ? rec.bm25 : 0;
        const cosPct = Math.max(0, Math.min(100, cosine * 100));
        const cosRaw = Number.isFinite(cosine) ? cosine.toFixed(3) : '0.000';
        const bm25Pct = Math.max(0, Math.min(100, (bm25v / (maxBM25 || 1)) * 100));
        const bm25Raw = Number.isFinite(bm25v) ? bm25v.toFixed(3) : '0.000';
        try {
          li.setAttribute('data-score', String(rec.score));
          li.setAttribute('data-cosine', String(cosine || 0));
          li.setAttribute('data-bm25', String(bm25v || 0));
          li.setAttribute('data-index', String(idx++));
        } catch(e){}
        const srcTag = preview ? 'local preview' : '';
        li.innerHTML = `
          <div class="res-head">
            <div><a class="title" draggable="false" href="https://arxiv.org/abs/${rec.paperId}" target="_blank">${rec.paperId}</a> — <span class="muted">${rec.kind}</span></div>
            <span class="score" title="cosine and BM25">
              <span class="metric" title="cosine">
                <span class="score-bar"><i style="width:${cosPct.toFixed(1)}%"></i></span>
                <span class="score-num">cos ${(cosPct).toFixed(1)}% (${cosRaw})</span>
              </span>
              <span class="metric" title="BM25 (relative)">
                <span class="score-bar"><i style="width:${bm25Pct.toFixed(1)}%"></i></span>
                <span class="score-num">bm25 ${(bm25Pct).toFixed(1)}% (${bm25Raw})</span>
              </span>
            </span>
            <button class="secondary" data-ai="${rec.paperId}" title="Open AI chat for this paper">AI</button>
          </div>
          <div class="muted meta-abstract">${preview || ' '}</div>
          <div class="muted meta-source">${srcTag ? `Source: ${escapeHtml(srcTag)}` : ''}</div>
          <div><a draggable="false" target="_blank" href="https://arxiv.org/abs/${rec.paperId}">arXiv</a> · <a draggable="false" target="_blank" href="https://arxiv.org/pdf/${rec.paperId}.pdf">PDF</a></div>
        `;
        // Note: drag disabled to allow mouse text selection/copy within results
        resultsEl.appendChild(li);
        if (paperIds.indexOf(rec.paperId) === -1) paperIds.push(rec.paperId);
      });
      try {
        if (paperIds.length) {
          const url = '/api/paper/meta?ids=' + encodeURIComponent(paperIds.join(','));
          const r2 = await fetch(url);
          const m = await r2.json();
          if (m && m.ok && m.meta) {
            paperIds.forEach((pid) => {
              const li = resultsEl.querySelector('li.result[data-paper-id="' + pid + '"]');
              const mm = m.meta[pid];
              if (!li || !mm) return;
              const title = (mm.title || pid).trim();
              const authors = Array.isArray(mm.authors) ? mm.authors.slice(0,4).join(', ') + (mm.authors.length>4?` +${mm.authors.length-4}`:'') : '';
              const subjects = Array.isArray(mm.subjects) ? mm.subjects.slice(0,3) : [];
              const abstract = (mm.abstract || '').replace(/\s+/g,' ').trim();
              const headLeft = li.querySelector('.res-head > div');
              if (headLeft) {
                const link = `<a class="title" draggable="false" href="https://arxiv.org/abs/${pid}" target="_blank">${escapeHtml(title)}</a>`;
                const auth = authors ? ` — <span class="muted authors">${escapeHtml(authors)}</span>` : '';
                const subs = subjects.length ? ` ${subjects.map(s=>`<span class=\"chip-subject\">${escapeHtml(String(s))}</span>`).join(' ')}` : '';
                headLeft.innerHTML = `${link}${auth}${subs}`;
              }
              const absEl = li.querySelector('.meta-abstract');
              const srcEl = li.querySelector('.meta-source');
              if (absEl && abstract) {
                absEl.textContent = abstract.slice(0, 500) + (abstract.length>500?'…':'');
                if (srcEl) srcEl.textContent = 'Source: ' + (mm.source === 'arxiv' ? 'arXiv abstract' : (mm.source === 'offline_latex' ? 'local LaTeX' : 'local preview'));
              } else {
                if (srcEl) srcEl.textContent = 'Source: local preview';
              }
            });
          }
        }
      } catch (e) {}
      if (!resultsEl._psbBound) {
        resultsEl._psbBound = true;
        resultsEl.addEventListener('click', (ev) => {
          // If the user is selecting text, don't trigger button actions.
          try {
            const sel = window.getSelection && window.getSelection();
            if (sel && !sel.isCollapsed && String(sel).trim().length > 0) {
              return; // allow copy selection without side effects
            }
          } catch (e) {}
          const t = ev.target;
          if (t && t.getAttribute && t.getAttribute('data-ai')) {
            const pid = t.getAttribute('data-ai');
            try { if (window._psbOpenPaperChat) window._psbOpenPaperChat(pid); } catch(e) {}
            ev.stopPropagation();
            ev.preventDefault();
            return false;
          }
        });
      }
      try { localStorage.setItem('PSB_LAST_RESULTS', JSON.stringify(paperIds)); } catch(e) {}
      try { statusEl._baseText = statusText; } catch(e){}
      try {
        const cosBtn = document.getElementById('btnSortCos');
        const bmBtn = document.getElementById('btnSortBm25');
        if (cosBtn && cosBtn.classList.contains('active')) {
          sortResults('cos', cosBtn.getAttribute('data-dir') || 'desc');
        } else if (bmBtn && bmBtn.classList.contains('active')) {
          sortResults('bm25', bmBtn.getAttribute('data-dir') || 'desc');
        }
      } catch(e){}
    } catch (e) {
      console.error(e);
      statusEl.textContent = 'Error searching';
    } finally {
      btnSearch.disabled = false;
    }
  }

  function openSettings() { modal && modal.classList.add('show'); }
  function closeSettings() { modal && modal.classList.remove('show'); }

  async function startBuildWithSettings() {
    if (!btnStartBuild) return;
    const low_mem = !!(optLowMem && optLowMem.checked);
    const post_batch_val = optPostBatch && optPostBatch.value ? parseInt(optPostBatch.value, 10) : null;
    const commit_papers_val = optCommitPapers && optCommitPapers.value ? parseInt(optCommitPapers.value, 10) : null;
    const mode = (optMode && optMode.value) || 'index-only';
    const limit_val = optLimit && optLimit.value ? parseInt(optLimit.value, 10) : null;
    const shards_val = optShards && optShards.value ? parseInt(optShards.value, 10) : null;
    const shard_workers_val = optShardWorkers && optShardWorkers.value ? parseInt(optShardWorkers.value, 10) : null;
    btnStartBuild.disabled = true;
    const orig = btnStartBuild.textContent;
    btnStartBuild.textContent = 'Starting...';
    try {
      const payload = { reset: false, mode, low_mem };
      if (post_batch_val && post_batch_val > 0) payload.post_batch = post_batch_val;
      if (commit_papers_val && commit_papers_val > 0) payload.commit_papers = commit_papers_val;
      if (Number.isFinite(limit_val) && limit_val > 0) payload.limit = limit_val;
      if (mode === 'sharded'){
        if (Number.isFinite(shards_val) && shards_val > 0) payload.shards = shards_val;
        if (Number.isFinite(shard_workers_val) && shard_workers_val > 0) payload.workers = shard_workers_val;
      }
      const r = await fetch('/api/index/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const j = await r.json();
      if (j.ok && j.started) {
        const totalFromStart = (j && j.papers_total) || 0;
        bannerEl.textContent = totalFromStart ? `Indexing (0/${totalFromStart})...` : 'Index build started (background).';
        const runStart = Date.now() / 1000;
        const buildId = j.build_id || null;
        const MAX_PAPER_STEP = 1000;
        let last = { done: 0, total: 0, chunks: 0, postings: 0, normSeen: 0, normTotal: 0 };
        let baselineSet = false;
        let tries = 0;
        const t = setInterval(async () => {
          tries++;
          try {
            const rs = await fetch('/api/index/status');
            const s = await rs.json();
            const ts = (s && typeof s.ts === 'number') ? s.ts : 0;
            if (ts && ts < runStart) return;
            if (buildId && s.build_id !== buildId) return;
            if (s.state === 'vectorizing') bannerEl.textContent = 'Vectorizing (scanning files)...';
            else if (s.state === 'indexing') {
              let suffix = '';
              if (s.papers_total && (s.papers_done !== undefined || (s && s.display_papers_done !== undefined))) {
                const total = Math.max(last.total || 0, s.papers_total || 0);
                let doneRaw = ((s && s.display_papers_done !== undefined ? s.display_papers_done : s.papers_done) || 0);
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
              } else if (s.phase === 'norms' && (s.postings_seen || s.postings_total)) {
                const tot = Math.max(last.normTotal || 0, s.postings_total || 0);
                const seen = Math.min(Math.max(last.normSeen || 0, s.postings_seen || 0), tot || Infinity);
                last.normTotal = tot; last.normSeen = seen;
                suffix = ` (norms postings ${seen}${tot?`/${tot}`:''})`;
              } else if (s.chunks || s.postings) {
                const chunks = Math.max(last.chunks || 0, s.chunks || 0);
                const posts = Math.max(last.postings || 0, s.postings || 0);
                last.chunks = chunks; last.postings = posts;
                suffix += ` (chunks ${chunks}${posts?`, postings ${posts}`:''})`;
              }
              bannerEl.textContent = 'Indexing (writing DB)...' + suffix;
            }
            else if (s.state === 'swapping') bannerEl.textContent = 'Finalizing (swapping new DB)...';
            else if (s.state === 'done') { bannerEl.textContent = 'Index build complete.'; clearInterval(t); try{diagnose();}catch(e){} btnStartBuild.disabled = false; btnStartBuild.textContent = orig || 'Start Index Build'; }
            else if (s.state === 'error') { bannerEl.textContent = `Index error: ${s.error || ''}`; clearInterval(t); btnStartBuild.disabled = false; btnStartBuild.textContent = orig || 'Start Index Build'; }
          } catch (e) {}
          if (tries > 900) clearInterval(t);
        }, 1000);
      } else {
        bannerEl.textContent = j && j.reason === 'busy' ? 'A build is already running.' : 'Failed to start build';
        btnStartBuild.disabled = false;
        btnStartBuild.textContent = orig || 'Start Index Build';
      }
    } catch (e) {
      console.error(e);
      bannerEl.textContent = 'Error starting build';
      btnStartBuild.disabled = false;
      btnStartBuild.textContent = orig || 'Start Index Build';
    }
  }

  btnSearch && btnSearch.addEventListener('click', search);
  if (btnCopyStatus && !btnCopyStatus._psbBound) {
    btnCopyStatus._psbBound = true;
    btnCopyStatus.addEventListener('click', async () => {
      try {
        const txt = (statusEl && statusEl.textContent) || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(txt);
        } else {
          const ta = document.createElement('textarea'); ta.value = txt; ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
        }
        const orig = btnCopyStatus.textContent; btnCopyStatus.textContent = 'Copied'; setTimeout(()=>{ btnCopyStatus.textContent=orig||'Copy'; }, 1200);
      } catch(e) {
        const orig = btnCopyStatus.textContent; btnCopyStatus.textContent = 'Copy failed'; setTimeout(()=>{ btnCopyStatus.textContent=orig||'Copy'; }, 1500);
      }
    });
  }
  qEl && qEl.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'enter') search(); });
  btnSettings && btnSettings.addEventListener('click', openSettings);
  btnCloseSettings && btnCloseSettings.addEventListener('click', closeSettings);
  modal && modal.addEventListener('click', (e) => { if (e.target === modal) closeSettings(); });
  btnStartBuild && btnStartBuild.addEventListener('click', () => { closeSettings(); startBuildWithSettings(); });

  // Sorting controls: toggle between highest and lowest for cosine and BM25
  (function setupSorting(){
    const cosBtn = document.getElementById('btnSortCos');
    const bmBtn = document.getElementById('btnSortBm25');
    const resetBtn = document.getElementById('btnSortReset');
    if (!cosBtn || cosBtn._psbBound) return;
    function setStatusSuffix(txt){
      try { const base = statusEl && (statusEl._baseText || statusEl.textContent) || ''; statusEl.textContent = base + (txt ? (' — Sorted: ' + txt) : ''); } catch(e){}
    }
    function setActive(which){
      if (!cosBtn || !bmBtn) return;
      if (which === 'cos') { cosBtn.classList.add('active'); bmBtn.classList.remove('active'); }
      else if (which === 'bm25') { bmBtn.classList.add('active'); cosBtn.classList.remove('active'); }
      else { cosBtn.classList.remove('active'); bmBtn.classList.remove('active'); }
    }
    window.sortResults = function(metric, dir){
      try{
        const items = Array.from(resultsEl.querySelectorAll('li.result'));
        if (!items.length) return;
        const key = metric === 'bm25' ? 'bm25' : 'cosine';
        const get = (li) => parseFloat(li.getAttribute('data-' + key) || '0') || 0;
        const frag = document.createDocumentFragment();
        if (metric === 'reset') {
          items.sort((a,b)=> (parseInt(a.getAttribute('data-index')||'0',10) - parseInt(b.getAttribute('data-index')||'0',10)));
          setStatusSuffix(''); setActive(null);
        } else {
          const sign = (dir === 'asc') ? 1 : -1; // asc: low→high, desc: high→low
          items.sort((a,b)=> sign * (get(a) - get(b)));
          const label = (metric==='bm25'?'bm25':'cos') + ' ' + (dir==='asc'?'low→high':'high→low');
          setStatusSuffix(label);
          setActive(metric);
        }
        items.forEach(li => frag.appendChild(li));
        resultsEl.innerHTML = '';
        resultsEl.appendChild(frag);
      } catch(e){}
    }
    function toggleBtn(btn, metric){
      const cur = (btn.getAttribute('data-dir') || 'desc');
      const next = cur === 'desc' ? 'asc' : 'desc';
      btn.setAttribute('data-dir', next);
      btn.textContent = (metric==='bm25'?'bm25 ':'cos ') + (next==='desc'?'▲':'▼');
      window.sortResults(metric, next);
    }
    cosBtn.addEventListener('click', ()=> toggleBtn(cosBtn, 'cos'));
    bmBtn.addEventListener('click', ()=> toggleBtn(bmBtn, 'bm25'));
    if (resetBtn) resetBtn.addEventListener('click', ()=> { window.sortResults('reset'); });
    cosBtn._psbBound = bmBtn._psbBound = true;
  })();

  (function(){
    const input = document.getElementById('llmInput');
    const btnSend = document.getElementById('btnLlmSend');
    function postToIframe(msg){
      try { const frame = document.getElementById('llmFrame'); frame && frame.contentWindow && frame.contentWindow.postMessage(msg, '*'); } catch(e){}
    }
    async function fallbackSend(){
      if (typeof window._psbSend === 'function') return;
      const q = ((input && input.value) || '').trim();
      if (!q) return; if (input) input.value = '';
      try { postToIframe({ type:'llm:add', role:'user', text:q }); postToIframe({ type:'llm:loading', show:true, text:'Thinking' });
        const r = await fetch('/api/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ query:q }) });
        const j = await r.json(); postToIframe({ type:'llm:loading', show:false });
        if(!j.ok){ postToIframe({ type:'llm:add', role:'assistant', text: (j.error || 'Ask failed.') }); return; }
        postToIframe({ type:'llm:add', role:'assistant', text: j.answer || '' });
      } catch(e){ try { postToIframe({ type:'llm:loading', show:false }); } catch(_){} postToIframe({ type:'llm:add', role:'assistant', text:'Error contacting server.' }); }
    }
    if (btnSend) btnSend.addEventListener('click', fallbackSend);
    if (input) input.addEventListener('keydown', (e)=>{ if ((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='enter') fallbackSend(); });
  })();

  // Ensure BM25 controls and Save placement exist even if an older HTML is cached
  (function ensureSettingsUI(){
    try {
      const modal = document.getElementById('modalSettings');
      if (!modal) return;
      const header = modal.querySelector('.panel > header');
      const headerActions = header && header.querySelector('div');
      let saveBtn = document.getElementById('btnSaveSettings');
      // Create Save button if missing
      if (!saveBtn && headerActions) {
        saveBtn = document.createElement('button');
        saveBtn.id = 'btnSaveSettings';
        saveBtn.className = 'btn';
        saveBtn.title = 'Save all settings to server (persist across restarts)';
        saveBtn.textContent = 'Save';
        headerActions.insertBefore(saveBtn, headerActions.firstChild);
      }
      // Move Save into header (left of Close) if it exists elsewhere
      if (saveBtn && headerActions && saveBtn.parentElement !== headerActions) {
        headerActions.insertBefore(saveBtn, headerActions.firstChild);
      }
      // Inject BM25 controls if not present
      const secSearch = document.getElementById('sec-search');
      if (secSearch && !document.getElementById('optScoring')) {
        const row = document.createElement('div');
        row.className = 'row2';
        row.innerHTML = `
          <div class="field">
            <label class="label" for="optScoring" title="Primary ranking metric: TF–IDF (cosine) or BM25. Both scores are always displayed; this only controls ordering. Tip: BM25 tends to recall more lexical matches; cosine normalizes by length. Ignored when Two‑pass is enabled.">Scoring</label>
            <select id="optScoring" title="Primary ranking metric: TF–IDF (cosine) or BM25. Both scores are always displayed; this only controls ordering. Tip: BM25 tends to recall more lexical matches; cosine normalizes by length. Ignored when Two‑pass is enabled.">
              <option value="tfidf">TF–IDF (cosine)</option>
              <option value="bm25">BM25</option>
            </select>
          </div>
          <div class="field">
            <label class="label" title="BM25 parameters (used when BM25 ranks results): k1 controls term frequency saturation; b controls length normalization (0=no length penalty, 1=full). Typical: k1≈1.2–2.0, b≈0.6–0.8.">BM25 (k1, b)</label>
            <div style="display:flex; gap:8px;">
              <input id="optBm25K1" type="number" step="0.1" min="0" placeholder="1.2" title="k1: increases the impact of repeated terms; higher = stronger TF saturation (diminishing returns slower)" />
              <input id="optBm25B" type="number" step="0.05" min="0" max="1" placeholder="0.75" title="b: length normalization. 0=no length penalty (long chunks favored), 1=full normalization (long chunks penalized more)" />
            </div>
          </div>`;
        // Insert after the first existing row in Search Settings
        const firstRow = secSearch.querySelector('.row2');
        if (firstRow && firstRow.nextSibling) secSearch.insertBefore(row, firstRow.nextSibling);
        else secSearch.appendChild(row);
      }
      // Attach a robust Save handler (persists to server + localStorage)
      saveBtn = document.getElementById('btnSaveSettings');
      if (saveBtn && !saveBtn._psbBound) {
        saveBtn._psbBound = true;
        saveBtn.addEventListener('click', async () => {
          try {
            const kindSel = document.getElementById('optSearchKind');
            const perPaper = document.getElementById('optSearchPerPaper');
            const maxRes = document.getElementById('optSearchMax');
            const scoringSel = document.getElementById('optScoring');
            const bm25K1 = document.getElementById('optBm25K1');
            const bm25B = document.getElementById('optBm25B');
            const svdEnable = document.getElementById('optSvdEnable');
            const svdTopN = document.getElementById('optSvdTopN');
            const svdAlpha = document.getElementById('optSvdAlpha');
            // Two-pass controls (if present)
            const twoPassEnable = document.getElementById('optTwoPassEnable');
            const twoPassTopN = document.getElementById('optTwoPassTopN');
            const o = {
              kind: kindSel && kindSel.value || 'both',
              perPaper: perPaper && perPaper.value ? parseInt(perPaper.value,10) : null,
              maxResults: maxRes && maxRes.value ? parseInt(maxRes.value,10) : null,
              scoring: scoringSel && scoringSel.value || 'tfidf',
              bm25K1: bm25K1 && bm25K1.value ? parseFloat(bm25K1.value) : null,
              bm25B: bm25B && bm25B.value ? parseFloat(bm25B.value) : null,
              svdEnable: !!(svdEnable && svdEnable.checked),
              svdTopN: svdTopN && svdTopN.value ? parseInt(svdTopN.value,10) : null,
              svdAlpha: svdAlpha && svdAlpha.value ? parseFloat(svdAlpha.value) : null,
              twoPassEnable: !!(twoPassEnable && twoPassEnable.checked),
              twoPassTopN: twoPassTopN && twoPassTopN.value ? parseInt(twoPassTopN.value,10) : null,
            };
            try { localStorage.setItem('PSB_SEARCH_OPTS', JSON.stringify(o)); } catch(e){}
            await fetch('/api/search/settings', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ settings: o }) });
            const banner = document.getElementById('banner'); if (banner) banner.textContent = 'Settings saved';
          } catch (e) {
            const banner = document.getElementById('banner'); if (banner) banner.textContent = 'Failed to save settings';
          }
        });
      }
    } catch (e) { /* ignore */ }
  })();

  diagnose();

  function escapeHtml(s){ return String(s).replace(/[&<>"']/g, (m)=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m])); }
  if (typeof window !== 'undefined' && !window._psbOpenPaperChat) { window._psbOpenPaperChat = function(pid){ try{ const b=document.getElementById('banner'); if(b) b.textContent = 'AI chat is available in UI2 layout.'; }catch(e){} }; }
})();
