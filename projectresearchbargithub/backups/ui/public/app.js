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
  // Search options moved to Settings; defaults come from localStorage or unlimited
  const resultsEl = document.getElementById('results');
  const statusEl = document.getElementById('status');
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
    let saved = (() => { try { return JSON.parse(localStorage.getItem('PSB_SEARCH_OPTS')||'{}'); } catch(e){ return {}; } })();
    // If no local settings, try to fetch from server-persisted settings
    if (!saved || Object.keys(saved).length === 0) {
      try {
        const r = await fetch('/api/search/settings');
        const j = await r.json();
        if (j && j.ok && j.settings) {
          saved = j.settings;
          try { localStorage.setItem('PSB_SEARCH_OPTS', JSON.stringify(saved)); } catch(e){}
        }
      } catch (e) { /* ignore */ }
    }
    const kind = (saved && saved.kind) || 'both';
    const DEFAULT_MAX = 5000; // safety cap for responsiveness
    // Default to 1 result per paper unless user sets otherwise to avoid duplicate papers dominating the list
    const per_paper_k = Number.isFinite(saved.perPaper) && saved.perPaper > 0 ? saved.perPaper : 1;
    const top_k = Number.isFinite(saved.maxResults) && saved.maxResults > 0 ? saved.maxResults : DEFAULT_MAX;
    const max_candidates = top_k; // align with requested results
    const scoring = (saved && saved.scoring) || 'tfidf';
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
      statusEl.textContent = `Found ${j.results.length} (scanned ${j.scanned})${limitedNote}`;
      const paperIds = [];
      // Precompute BM25 max for proportional bar rendering (avoid division by zero)
      let maxBM25 = 0;
      try { maxBM25 = j.results.reduce((m, rec) => Math.max(m, Number(rec && rec.bm25 || 0)), 0); } catch(e){ maxBM25 = 0; }
      if (!(maxBM25 > 0)) maxBM25 = 1;
      j.results.forEach((rec) => {
        const li = document.createElement('li');
        li.className = 'result';
        li.setAttribute('draggable','true');
        li.setAttribute('data-paper-id', rec.paperId);
        const preview = (rec.text || '').replace(/\s+/g, ' ').slice(0, 400);
        const cosine = (typeof rec.cosine === 'number') ? rec.cosine : (rec.scoring === 'bm25' ? 0 : (rec.score || 0));
        const bm25v = (typeof rec.bm25 === 'number') ? rec.bm25 : 0;
        const cosPct = Math.max(0, Math.min(100, cosine * 100));
        const bm25Pct = Math.max(0, Math.min(100, (bm25v / (maxBM25 || 1)) * 100));
        try { li.setAttribute('data-score', String(rec.score)); } catch(e){}
        li.innerHTML = `
          <div class="res-head">
            <div><a class="title" href="https://arxiv.org/abs/${rec.paperId}" target="_blank">${rec.paperId}</a> — <span class="muted">${rec.kind}</span></div>
            <span class="score" title="cosine and BM25">
              <span class="score-bar" title="cosine"><i style="width:${cosPct.toFixed(1)}%"></i></span>
              <span class="score-num">cos ${(cosPct).toFixed(1)}%</span>
              <span class="score-bar" title="BM25 (relative)"><i style="width:${bm25Pct.toFixed(1)}%"></i></span>
              <span class="score-num">bm25 ${(bm25Pct).toFixed(1)}%</span>
            </span>
            <button class="secondary" data-ai="${rec.paperId}" title="Open AI chat for this paper">AI</button>
          </div>
          <div class="muted meta-abstract">${preview}</div>
          <div><a target="_blank" href="https://arxiv.org/abs/${rec.paperId}">arXiv</a> · <a target="_blank" href="https://arxiv.org/pdf/${rec.paperId}.pdf">PDF</a></div>
        `;
        li.addEventListener('dragstart', (ev) => {
          try { ev.dataTransfer.setData('application/x-psb-paper-id', rec.paperId); ev.dataTransfer.effectAllowed = 'copyMove'; } catch(e){}
        });
        resultsEl.appendChild(li);
        if (paperIds.indexOf(rec.paperId) === -1) paperIds.push(rec.paperId);
      });
      // Fetch paper metadata (title, authors, abstract, subjects) and upgrade result cards
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
              // Update header left block with title + authors + subjects
              const headLeft = li.querySelector('.res-head > div');
              if (headLeft) {
                const link = `<a class="title" href="https://arxiv.org/abs/${pid}" target="_blank">${escapeHtml(title)}</a>`;
                const auth = authors ? ` — <span class="muted authors">${escapeHtml(authors)}</span>` : '';
                const subs = subjects.length ? ` ${subjects.map(s=>`<span class=\"chip-subject\">${escapeHtml(String(s))}</span>`).join(' ')}` : '';
                headLeft.innerHTML = `${link}${auth}${subs}`;
              }
              const absEl = li.querySelector('.meta-abstract');
              if (absEl && abstract) {
                absEl.textContent = abstract.slice(0, 500) + (abstract.length>500?'…':'');
              }
            });
          }
        }
      } catch (e) {
        // ignore meta errors; keep base results
      }
      // AI button
      resultsEl.addEventListener('click', (ev) => {
        const t = ev.target;
        if (t && t.getAttribute && t.getAttribute('data-ai')) {
          const pid = t.getAttribute('data-ai');
          try { if (window._psbOpenPaperChat) window._psbOpenPaperChat(pid); } catch(e) {}
          return;
        }
      });
      try { localStorage.setItem('PSB_LAST_RESULTS', JSON.stringify(paperIds)); } catch(e) {}
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
        // Track a per-run start timestamp to ignore stale status from old runs
        const runStart = Date.now() / 1000;
        const buildId = j.build_id || null;
        // Monotonic smoothing to avoid flashing counters
        const MAX_PAPER_STEP = 1000; // cap visible step per poll for readability
        let last = { done: 0, total: 0, chunks: 0, postings: 0, normSeen: 0, normTotal: 0 };
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
              // Smooth papers counters
              if (s.papers_total && (s.papers_done !== undefined || s.display_papers_done !== undefined)) {
                const total = Math.max(last.total || 0, s.papers_total || 0);
                let doneRaw = (s.display_papers_done !== undefined ? s.display_papers_done : s.papers_done) || 0;
                // Force a 0 baseline on first accepted status to avoid jumping to old counts
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
  qEl && qEl.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'enter') search(); });
  btnSettings && btnSettings.addEventListener('click', openSettings);
  btnCloseSettings && btnCloseSettings.addEventListener('click', closeSettings);
  modal && modal.addEventListener('click', (e) => { if (e.target === modal) closeSettings(); });
  btnStartBuild && btnStartBuild.addEventListener('click', () => { closeSettings(); startBuildWithSettings(); });

  // --- Robust fallback for LLM dock ---
  // If inline chat script fails to initialize (so window._psbSend/_psbSendRead stay undefined),
  // attach minimal handlers here so the buttons still work.
  (function(){
    const input = document.getElementById('llmInput');
    const topQuery = document.getElementById('query');
    const btnSend = document.getElementById('btnLlmSend');
    // Read Papers button removed
    function postToIframe(msg){
      try {
        const frame = document.getElementById('llmFrame');
        frame && frame.contentWindow && frame.contentWindow.postMessage(msg, '*');
      } catch(e){}
    }
    async function fallbackSend(){
      if (typeof window._psbSend === 'function') return; // primary handler exists
      const banner = document.getElementById('banner');
      const q = ((input && input.value) || '').trim();
      if (!q) return;
      if (input) input.value = '';
      try {
        if (banner) banner.textContent = 'Sending…';
        postToIframe({ type:'llm:add', role:'user', text:q });
        postToIframe({ type:'llm:loading', show:true, text:'Thinking' });
        const r = await fetch('/api/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ query:q }) });
        const j = await r.json();
        postToIframe({ type:'llm:loading', show:false });
        if(!j.ok){
          let msg = j.error || 'Ask failed.';
          if (j.error === 'no_api_key') msg = 'No API key set. Open Settings and save your OpenAI API key.';
          postToIframe({ type:'llm:add', role:'assistant', text: msg });
          if (banner) banner.textContent = 'Ask failed';
          return;
        }
        postToIframe({ type:'llm:add', role:'assistant', text: j.answer || '' });
        if (banner) banner.textContent = 'Done.';
      } catch(e){
        try { postToIframe({ type:'llm:loading', show:false }); } catch(_){}
        postToIframe({ type:'llm:add', role:'assistant', text:'Error contacting server.' });
        const b = document.getElementById('banner'); if (b) b.textContent='Error contacting server';
      }
    }
    if (btnSend) btnSend.addEventListener('click', fallbackSend);
    if (input) input.addEventListener('keydown', (e)=>{ if ((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='enter') fallbackSend(); });
    // Read Papers fallback removed
})();
diagnose();

// Simple HTML escaper for safety when injecting meta
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, (m)=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]));
}

// Fallback stub: if UI does not provide the paper chat opener, show a banner hint
if (typeof window !== 'undefined' && !window._psbOpenPaperChat) {
  window._psbOpenPaperChat = function(pid){ try{ const b=document.getElementById('banner'); if(b) b.textContent = 'AI chat is available in UI2 layout.'; }catch(e){} };
}
})();
