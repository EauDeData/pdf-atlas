/* Doc explorer — vanilla JS, no build step.
   Everything heavy is paginated server-side: search = offset slices of a cached ranked list,
   browse = keyset cursor. The corpus keeps growing while the page is open (progress poll). */
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => (n ?? 0).toLocaleString('en-US');
const DPR = Math.min(2, window.devicePixelRatio || 1);
const PAGE = 20;

const S = {
  stats: null, topics: {}, colls: [],
  mode: 'semantic', q: '', next: 0, loading: false, gen: 0, shown: 0,
  filt: {collection: '', topic: '', yf: '', yt: ''},
  browseCursor: -1, browseLoading: false, browseGen: 0,
  lastIndexed: null,
};

async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  if (!r.ok) { const t = await r.text(); throw new Error(t || r.statusText); }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r.arrayBuffer();
}

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('on'), 2400);
}

// images fade in only once decoded (no pop)
function lazyImg(img, src) {
  img.decoding = 'async'; img.loading = 'lazy';
  img.onload = () => img.classList.add('ok');
  img.src = src;
  return img;
}
const pageSrc = (doc, p, w) => `/api/render/${doc}/${p}.jpg?w=${Math.round(w * DPR)}`;
const imgSrc = (id, w) => `/api/image/${id}.jpg?w=${Math.round(w * DPR)}`;

function topicColor(kind, t) {
  const tp = S.topics[kind];
  return tp && t >= 0 && tp.colors[t] ? tp.colors[t] : '#4d4d4b';
}
function topicName(kind, t, n = 3) {
  const tp = S.topics[kind];
  return tp && t >= 0 && tp.labels[t] ? tp.labels[t].slice(0, n).join(' · ') : '—';
}
const TOPIC_KIND = k => ({docs: 'text', images: 'image'})[k] || k;   // map kind -> its key in topics.json
const COLL_COLORS = c => `hsl(${(S.colls.indexOf(c) * 360 / Math.max(1, S.colls.length) + 200) % 360} 55% 60%)`;

// ======================================================================= progress (live)
async function pollProgress() {
  try {
    const p = await api('/api/progress');
    const pct = p.indexed_pct || 0;
    $('#idxline i').style.width = pct + '%';
    const done = p.done && p.done.ingest && p.done.topics;
    $('#idxline').classList.toggle('done', !!done);
    $('#idxchip').classList.toggle('done', !!done);
    $('#idxtext').textContent = done ? `${fmt(p.indexed_docs)} docs indexed` :
      `Indexing ${pct.toFixed(1)}% · ${fmt(p.indexed_docs)} / ${fmt(p.total_docs)}`;
    renderProgress(p);
    if (S.lastIndexed !== null && p.indexed_docs > S.lastIndexed) {
      const d = p.indexed_docs - S.lastIndexed;
      if (d >= 50) toast(`+${fmt(d)} documents indexed`);
      growth();
    }
    if (S.lastIndexed === null || p.indexed_docs > S.lastIndexed) S.lastIndexed = p.indexed_docs;
    // documents get their map position a little after they are indexed (topics stage)
    const placed = (p.db || {}).topic_docs || 0;
    if (S.lastPlaced != null && placed > S.lastPlaced) growth();
    S.lastPlaced = placed;
  } catch (e) { $('#idxtext').textContent = 'pipeline offline'; }
  setTimeout(pollProgress, document.hidden ? 20000 : 5000);
}

function renderProgress(p) {
  if ($('#progress').hidden) return;
  const st = (p.stages || []).map(s => `
    <div class="stage">
      <span class="lb">${esc(s.label)}</span><span class="pc">${s.pct.toFixed(2)}%</span>
      <span class="tr"><i style="width:${s.pct}%"></i></span>
      <span class="sub mono">${fmt(s.done)} / ${fmt(s.total)} ${s.unit}${s.rate_per_min ? ` · ${s.rate_per_min}/min` : ''}${s.eta_min ? ` · eta ${Math.round(s.eta_min)} min` : ''}</span>
    </div>`).join('');
  const e = p.extracted || {}, d = p.db || {};
  $('#p-body').innerHTML = st + `
    <div class="pfacts mono">
      <div>pages read<b>${fmt(e.pages)}</b></div><div>image refs<b>${fmt(e.images)}</b></div>
      <div>text chunks<b>${fmt(d.chunks)}</b></div><div>annoy segments<b>${(d.segments || []).length}</b></div>
      <div>text-less (scans)<b>${fmt(e.scans)}</b></div><div>errors<b>${fmt(e.errors)}</b></div>
    </div>
    <p class="mono faint" style="margin-top:14px">${(p.gpus || []).map(g => `gpu${g.index} ${(g.ours / 1024).toFixed(1)}GB`).join(' · ')} — <a class="red" href="/progress" target="_blank">full view ↗</a></p>`;
}

function openProgress(on) {
  const d = $('#progress');
  if (on) { d.hidden = false; requestAnimationFrame(() => d.classList.add('on')); pollProgressOnce(); }
  else { d.classList.remove('on'); setTimeout(() => d.hidden = true, 200); }
}
async function pollProgressOnce() { try { renderProgress(await api('/api/progress')); } catch {} }

// called when new documents land: refresh the cheap things, throttled to one per 30 s, with a
// trailing refresh so the last change is never dropped (the server also caches the map for 30 s)
let growthT = 0, growthTimer = 0;
function growth() {
  const wait = 31000 - (Date.now() - growthT);
  if (wait > 0) { if (!growthTimer) growthTimer = setTimeout(() => { growthTimer = 0; growth(); }, wait); return; }
  growthT = Date.now();
  loadStats().then(() => Fig.reload(true));
  clearTimeout(growth.follow); growth.follow = setTimeout(() => Fig.reload(true), 32000);  // outlive the server's map cache
}

// ======================================================================= stats / collections / topics
async function loadStats() {
  let s;
  try { s = await api('/api/stats'); } catch (e) { $('#hero-count').textContent = 'The index is being created — this page fills in as documents arrive.'; return; }
  S.stats = s; S.topics = s.topics || {};
  if (s.title && !S.titled) {   // "Medieval charters atlas" -> "Medieval charters" / outlined "atlas"
    S.titled = true;
    const w = s.title.trim().split(/\s+/), last = w.length > 1 ? w.pop() : '';
    $('#t1').textContent = w.join(' '); $('#t2').textContent = last; $('#t2').parentElement.hidden = !last;
    $('#navtitle').textContent = '/ ' + s.title; document.title = s.title;
  }
  S.colls = s.all_collections || [];
  $('#hero-count').textContent = `${fmt(s.docs)} documents, ${fmt(s.pages)} pages, ${fmt(s.chunks)} passages and ${fmt(s.image_vectors)} figures searchable so far.`;
  $('#models').textContent = `${s.models.text} · ${s.models.image}`;
  fillFilters();
  renderCollections();
  renderYears();
  renderTopics();
  Fig.legend();
}

function fillFilters() {
  const fc = $('#f-coll'), cur = fc.value;
  fc.innerHTML = '<option value="">All</option>' + S.colls.map(c => `<option${c === cur ? ' selected' : ''}>${esc(c)}</option>`).join('');
  const ft = $('#f-topic'), curt = ft.value;
  const kind = S.mode === 'image' ? 'image' : 'text';
  const tp = S.topics[kind];
  ft.innerHTML = '<option value="">All</option>' + (tp ? tp.labels.map((l, i) => `<option value="${i}"${String(i) === curt ? ' selected' : ''}>${String(i).padStart(2, '0')} ${esc(l.slice(0, 3).join(' · '))}</option>`).join('') : '');
}

function renderCollections() {
  const s = S.stats, cat = s.catalog_collections || {};
  const rows = s.collections.slice();
  for (const c of Object.keys(cat)) if (!rows.find(r => r.collection === c)) rows.push({collection: c, n: 0, pages: 0, images: 0});
  rows.sort((a, b) => (cat[b.collection] || b.n) - (cat[a.collection] || a.n));
  const max = Math.max(1, ...rows.map(r => cat[r.collection] || r.n));
  $('#collbars').innerHTML = rows.map(r => {
    const tot = cat[r.collection] || r.n;
    return `<li data-c="${esc(r.collection)}" title="${fmt(r.pages)} pages · ${fmt(r.images)} images">
      <span class="nm mono">${esc(r.collection)}</span>
      <span class="tr"><b style="width:${tot / max * 100}%"></b><i style="width:${r.n / max * 100}%"></i></span>
      <span class="n mono">${fmt(r.n)} / ${fmt(tot)}</span></li>`;
  }).join('');
  $('#facts').innerHTML = [['documents', s.docs], ['pages', s.pages], ['passages', s.chunks], ['figures', s.image_vectors],
    ['collections', s.collections.length], ['gigabytes', Math.round((s.bytes || 0) / 1e9)]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${fmt(v)}</dd></div>`).join('');
}

function renderYears() {
  const cv = $('#years'), ys = S.stats.years || [];
  const w = cv.clientWidth, h = 220;
  cv.width = w * DPR; cv.height = h * DPR;
  const g = cv.getContext('2d'); g.scale(DPR, DPR); g.clearRect(0, 0, w, h);
  if (!ys.length) return;
  const y0 = Math.max(1850, ys[0].year), y1 = Math.min(2026, ys[ys.length - 1].year);
  const by = {}; ys.forEach(r => by[r.year] = r.n);
  const max = Math.max(...ys.filter(r => r.year >= y0).map(r => r.n), 1);
  const bw = (w - 30) / (y1 - y0 + 1);
  g.fillStyle = '#ff3b2f';
  for (let y = y0; y <= y1; y++) {
    const v = by[y] || 0, bh = Math.sqrt(v / max) * (h - 30);
    g.fillRect(30 + (y - y0) * bw, h - 18 - bh, Math.max(1, bw - 1), bh);
  }
  g.fillStyle = '#4d4d4b'; g.font = '10px "Martian Mono", monospace';
  for (let y = Math.ceil(y0 / 25) * 25; y <= y1; y += 25) g.fillText(y, 30 + (y - y0) * bw - 12, h - 4);
  g.fillText('√n', 0, 12);
  cv.onclick = ev => {
    const r = cv.getBoundingClientRect(), y = Math.round(y0 + (ev.clientX - r.left - 30) / bw);
    if (y >= y0 && y <= y1) { $('#f-y0').value = y - 2; $('#f-y1').value = y + 2; filtersChanged(); location.hash = '#browse'; }
  };
}

let topicKind = 'text';
const topicObs = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  topicObs.unobserve(e.target); loadTopicSamples(e.target);
}), {rootMargin: '200px'});

function renderTopics() {
  const tp = S.topics[topicKind], box = $('#topiclist');
  if (!tp) { box.innerHTML = `<div class="topic"><span class="mono faint">Not fitted yet — topics appear after the first ${topicKind === 'text' ? '1,500 documents' : '3,000 images'} are indexed.</span></div>`; $('#topic-ver').textContent = ''; return; }
  $('#topic-ver').textContent = `· fitted on ${fmt(tp.fitted_on)} ${topicKind === 'text' ? 'docs' : 'images'}, k=${tp.k}`;
  const max = Math.max(...tp.counts, 1);
  const order = tp.labels.map((_, i) => i).sort((a, b) => tp.counts[b] - tp.counts[a]);
  box.innerHTML = order.map(i => `
    <article class="topic" data-k="${i}" tabindex="0">
      <div class="tn mono"><span><i style="background:${tp.colors[i]}"></i>T${String(i).padStart(2, '0')}</span><span>${fmt(tp.counts[i])}</span></div>
      <div class="tw">${esc(tp.labels[i].slice(0, 3).join(' · '))} <span>${esc(tp.labels[i].slice(3, 6).join(' · '))}</span></div>
      ${topicKind === 'text' ? '<div class="ts"></div>' : '<div class="imgs"></div>'}
      <div class="bar"><i style="width:${tp.counts[i] / max * 100}%;background:${tp.colors[i]}"></i></div>
    </article>`).join('');
  $$('.topic', box).forEach(el => topicObs.observe(el));
}

async function loadTopicSamples(el) {
  const k = el.dataset.k;
  if (topicKind === 'text') {
    const ds = await api(`/api/topic/docs/${k}?n=3`).catch(() => []);
    $('.ts', el).innerHTML = ds.map(d => `<div title="${esc(d.title)}">${esc(d.title)}</div>`).join('');
  } else {
    const is = await api(`/api/topic/images/${k}?n=4`).catch(() => []);
    const box = $('.imgs', el);
    is.forEach(i => box.appendChild(lazyImg(new Image(), imgSrc(i.id, 120))));
  }
}

// ======================================================================= Fig. 1 map
const Fig = (() => {
  const cv = $('#map'), g = cv.getContext('2d');
  let kind = 'docs', colorBy = 'topic';
  let n = 0, ids, xs, ys, tops, colls, born = new Float32Array(0), known = new Set();
  let view = {s: 1, x: 0, y: 0}, W = 0, H = 0, grid = null, hl = null, hover = -1, raf = 0;
  // lasso: polygon in map (data) coordinates, so it follows zoom/pan; sel = indices of points inside it
  let lassoOn = false, poly = null, drawing = false, sel = null, lastPt = null;
  // lassoes already saved (by anyone) on this map layout: shown while lasso mode is on
  let saved = [], savedOther = 0, savedTimer = 0;
  const hue = s => { let h = 0; for (const c of s) h = (h * 31 + c.codePointAt(0)) >>> 0; return h % 360; };
  const PAD = 24;

  function resize() {
    const r = cv.getBoundingClientRect(); W = r.width; H = r.height;
    cv.width = W * DPR; cv.height = H * DPR; draw();
  }
  const sx = i => PAD + (xs[i] * (W - 2 * PAD) - view.x) * view.s + (W - 2 * PAD) * (1 - view.s) / 2;
  const sy = i => 70 + (ys[i] * (H - 70 - PAD) - view.y) * view.s + (H - 70 - PAD) * (1 - view.s) / 2;
  const toScreen = (x, y) => [PAD + (x * (W - 2 * PAD) - view.x) * view.s + (W - 2 * PAD) * (1 - view.s) / 2,
                              70 + (y * (H - 70 - PAD) - view.y) * view.s + (H - 70 - PAD) * (1 - view.s) / 2];
  const toData = (mx, my) => [((mx - PAD - (W - 2 * PAD) * (1 - view.s) / 2) / view.s + view.x) / (W - 2 * PAD),
                              ((my - 70 - (H - 70 - PAD) * (1 - view.s) / 2) / view.s + view.y) / (H - 70 - PAD)];

  async function reload(keepView) {
    let buf;
    try { buf = await api(`/api/map/${kind}.bin`); } catch { return; }
    const dv = new DataView(buf), m = dv.getUint32(0, true);
    let o = 4;
    const nids = new Int32Array(buf.slice(o, o + 4 * m)); o += 4 * m;
    const xy = new Uint16Array(buf.slice(o, o + 4 * m)); o += 4 * m;
    const tp = new Uint8Array(buf.slice(o, o + m)); o += m;
    const cl = new Uint8Array(buf.slice(o, o + m));
    const now = performance.now(), nb = new Float32Array(m);
    const fresh = known.size > 0;
    const nk = new Set();
    xs = new Float32Array(m); ys = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      xs[i] = xy[2 * i] / 65535; ys[i] = xy[2 * i + 1] / 65535;
      nk.add(nids[i]);
      nb[i] = fresh && !known.has(nids[i]) ? now : 0;   // new since last load -> red halo
    }
    known = nk; ids = nids; tops = tp; colls = cl; born = nb; n = m;
    if (!keepView) view = {s: 1, x: 0, y: 0};
    buildGrid();
    if (poly && !drawing) select();
    $('#map-n').textContent = fmt(n);
    $('#mapempty').hidden = n > 0;
    legend(); draw();
  }

  function buildGrid() {   // spatial hash in data space for O(1) hover lookup
    const G = 128; grid = new Map_(G);
    for (let i = 0; i < n; i++) grid.add(Math.min(G - 1, xs[i] * G | 0), Math.min(G - 1, ys[i] * G | 0), i);
  }
  function Map_(G) { this.G = G; this.b = new Array(G * G); }
  Map_.prototype.add = function (x, y, i) { const k = y * this.G + x; (this.b[k] ||= []).push(i); };

  function color(i) {
    const t = tops[i] - 1;
    return colorBy === 'topic' ? topicColor(TOPIC_KIND(kind), t) : (COLL_COLORS(S.colls[colls[i]]) || '#888');
  }

  function draw() {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      g.setTransform(DPR, 0, 0, DPR, 0, 0);
      g.clearRect(0, 0, W, H);
      if (!n) return;
      const r = Math.max(1, Math.min(3.2, 1.1 * Math.sqrt(view.s) * (n < 20000 ? 1.6 : 1)));
      const now = performance.now();
      let animating = false;
      // batch by color for speed
      const buckets = {};
      for (let i = 0; i < n; i++) { const c = color(i); (buckets[c] ||= []).push(i); }
      g.globalAlpha = hl ? 0.16 : (n > 60000 ? 0.55 : 0.8);
      for (const c in buckets) {
        g.fillStyle = c; g.beginPath();
        for (const i of buckets[c]) { const x = sx(i), y = sy(i); if (x < -5 || y < -5 || x > W + 5 || y > H + 5) continue; g.rect(x - r / 2, y - r / 2, r, r); }
        g.fill();
      }
      g.globalAlpha = 1;
      if (hl) {
        g.fillStyle = '#ff3b2f'; g.beginPath();
        for (let i = 0; i < n; i++) if (hl.has(ids[i])) { const x = sx(i), y = sy(i); g.rect(x - r, y - r, r * 2, r * 2); }
        g.fill();
      }
      // newborn halo (1.6 s, ease-out)
      for (let i = 0; i < n; i++) {
        if (!born[i]) continue;
        const t = (now - born[i]) / 1600;
        if (t >= 1) { born[i] = 0; continue; }
        animating = true;
        const e = 1 - Math.pow(1 - t, 3);
        g.strokeStyle = `rgba(255,59,47,${1 - e})`; g.lineWidth = 1;
        g.beginPath(); g.arc(sx(i), sy(i), 2 + 10 * e, 0, 6.283); g.stroke();
      }
      if (lassoOn && saved.length) {
        g.font = '500 10px "Martian Mono", monospace'; g.lineWidth = 1.2;
        for (const L of saved) {
          const h = hue(L.label);
          g.beginPath();
          L.polygon.forEach(([x, y], k) => { const [px, py] = toScreen(x, y); k ? g.lineTo(px, py) : g.moveTo(px, py); });
          g.closePath();
          g.fillStyle = `hsla(${h} 75% 60% / .10)`; g.fill();
          g.strokeStyle = `hsl(${h} 75% 62%)`; g.stroke();
          const [tx, ty] = toScreen(L.bx, L.by);
          const t = L.label.length > 28 ? L.label.slice(0, 27) + '…' : L.label, tw = g.measureText(t).width;
          g.fillStyle = 'rgba(10,10,10,.8)'; g.fillRect(tx, ty - 13, tw + 8, 14);
          g.fillStyle = `hsl(${h} 75% 70%)`; g.fillText(t, tx + 4, ty - 2);
        }
      }
      if (poly && poly.length > 1) {
        g.beginPath();
        poly.forEach(([x, y], k) => { const [px, py] = toScreen(x, y); k ? g.lineTo(px, py) : g.moveTo(px, py); });
        if (!drawing) g.closePath();
        g.fillStyle = 'rgba(255,59,47,.08)'; g.fill();
        g.strokeStyle = '#ff3b2f'; g.lineWidth = 1.5; g.setLineDash(drawing ? [4, 4] : []); g.stroke(); g.setLineDash([]);
        if (sel && sel.length) {
          g.fillStyle = '#ececea'; g.beginPath();
          for (const i of sel) { const x = sx(i), y = sy(i); g.rect(x - r / 2 - .5, y - r / 2 - .5, r + 1, r + 1); }
          g.fill();
        }
      }
      if (hover >= 0) {
        g.strokeStyle = '#ececea'; g.lineWidth = 1.5;
        g.beginPath(); g.arc(sx(hover), sy(hover), 6, 0, 6.283); g.stroke();
      }
      if (animating) draw();
    });
  }

  function pick(mx, my) {
    if (!grid) return -1;
    const G = grid.G;
    const dx = ((mx - PAD - (W - 2 * PAD) * (1 - view.s) / 2) / view.s + view.x) / (W - 2 * PAD);
    const dy = ((my - 70 - (H - 70 - PAD) * (1 - view.s) / 2) / view.s + view.y) / (H - 70 - PAD);
    const cx = dx * G | 0, cy = dy * G | 0, rad = Math.max(1, Math.ceil(2 / view.s));
    let best = -1, bd = 14 * 14;
    for (let y = cy - rad; y <= cy + rad; y++) for (let x = cx - rad; x <= cx + rad; x++) {
      if (x < 0 || y < 0 || x >= G || y >= G) continue;
      const b = grid.b[y * G + x]; if (!b) continue;
      for (const i of b) { const ddx = sx(i) - mx, ddy = sy(i) - my, d = ddx * ddx + ddy * ddy; if (d < bd) { bd = d; best = i; } }
    }
    return best;
  }

  // hover card: fetched lazily, cached, origin-aware placement
  const card = $('#card'), cache = new Map__();
  function Map__() { this.m = {}; }
  Map__.prototype.get = function (k) { return this.m[k]; };
  Map__.prototype.set = function (k, v) { this.m[k] = v; };
  let cardTimer = 0, cardFor = -1;
  async function showCard(i, mx, my) {
    const id = ids[i]; cardFor = id;
    const left = mx > W - 320, top = my > H - 300;
    card.style.transform = `translate(${left ? mx - 300 : mx + 14}px, ${top ? Math.max(0, my - 280) : my + 14}px)`;
    card.style.setProperty('--ox', left ? '100%' : '0%'); card.style.setProperty('--oy', top ? '100%' : '0%');
    let b = cache.get(kind + id);
    if (!b) {
      const r = await api(`/api/brief?kind=${kind}&ids=${id}`).catch(() => []);
      b = r[0]; if (!b) return; cache.set(kind + id, b);
    }
    if (cardFor !== id) return;
    const ph = kind === 'docs' ? pageSrc(b.doc_id, 0, 288) : kind === 'tasks'
      ? (b.n_figures ? `/api/task/${b.id}/crop/0.jpg` : pageSrc(b.doc_id, b.page, 288)) : imgSrc(b.id, 288);
    const t = tops[i] - 1;
    card.firstElementChild.innerHTML = `<div class="ph"><img alt=""></div><div class="tx"><span class="mono faint">${kind === 'tasks' ? esc(taskName(b.task)) + ' · ' : ''}${esc(b.collection)} · ${b.year ?? 's.d.'}${kind === 'docs' ? ` · ${fmt(b.n_pages)} pp` : ` · p.${b.page + 1}`}</span>
      <b>${esc(b.title)}</b>${kind === 'tasks' && b.target ? `<span class="faint">${esc(b.target.slice(0, 160))}</span>` : ''}<span class="mono" style="color:${color(i)}">${esc(topicName(TOPIC_KIND(kind), t))}</span></div>`;
    lazyImg($('img', card), ph);
    card.classList.add('on');
  }
  function hideCard() { cardFor = -1; card.classList.remove('on'); }

  // interaction: wheel zoom at cursor, drag pan, click open
  let drag = null, moved = false;
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015), ns = Math.min(40, Math.max(1, view.s * k));
    const fx = (mx - PAD - (W - 2 * PAD) * (1 - view.s) / 2) / view.s + view.x;
    const fy = (my - 70 - (H - 70 - PAD) * (1 - view.s) / 2) / view.s + view.y;
    view.s = ns;
    view.x = fx - (mx - PAD - (W - 2 * PAD) * (1 - ns) / 2) / ns;
    view.y = fy - (my - 70 - (H - 70 - PAD) * (1 - ns) / 2) / ns;
    if (ns === 1) view.x = view.y = 0;
    hideCard(); draw();
  }, {passive: false});
  cv.addEventListener('pointerdown', e => {
    cv.setPointerCapture(e.pointerId);
    if (lassoOn || e.shiftKey) {   // start a lasso (replaces any pending one)
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      drawing = true; poly = [toData(mx, my)]; lastPt = [mx, my]; sel = null;
      $('#lasso').hidden = true; hideCard(); draw();
      return;
    }
    drag = {x: e.clientX, y: e.clientY, vx: view.x, vy: view.y}; moved = false;
  });
  cv.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (drawing) {
      if (Math.hypot(mx - lastPt[0], my - lastPt[1]) >= 3) { poly.push(toData(mx, my)); lastPt = [mx, my]; draw(); }
      return;
    }
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) { moved = true; cv.classList.add('grab'); hideCard(); }
      if (moved) { view.x = drag.vx - dx / view.s; view.y = drag.vy - dy / view.s; draw(); }
      return;
    }
    if (e.pointerType !== 'mouse') return;
    const i = pick(mx, my);
    if (i !== hover) { hover = i; draw(); clearTimeout(cardTimer); if (i < 0) hideCard(); else cardTimer = setTimeout(() => showCard(i, mx, my), card.classList.contains('on') ? 0 : 60); }
  });
  cv.addEventListener('pointerup', e => {
    if (drawing) { drawing = false; finishLasso(); return; }
    cv.classList.remove('grab');
    const wasDrag = moved; drag = null;
    if (wasDrag) return;
    const r = cv.getBoundingClientRect(), i = pick(e.clientX - r.left, e.clientY - r.top);
    if (i < 0) return;
    if (e.pointerType !== 'mouse' && hover !== i) { hover = i; draw(); showCard(i, e.clientX - r.left, e.clientY - r.top); return; } // tap = preview, 2nd tap = open
    if (kind === 'docs') Viewer.open(ids[i]);
    else api(`/api/brief?kind=${kind}&ids=${ids[i]}`).then(b => b[0] && Viewer.open(b[0].doc_id, b[0].page, [], kind === 'tasks' ? b[0].id : null));
  });
  cv.addEventListener('pointerleave', () => { if (!drag && !drawing) { hover = -1; hideCard(); draw(); } });
  cv.addEventListener('pointercancel', () => { if (drawing) clearLasso(); });

  // ---- lasso -> label
  function inPoly(x, y, P) {   // even-odd ray casting
    let c = false;
    for (let a = 0, b = P.length - 1; a < P.length; b = a++) {
      const [xa, ya] = P[a], [xb, yb] = P[b];
      if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) c = !c;
    }
    return c;
  }
  function select() {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const [x, y] of poly) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = xs[i], y = ys[i];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && inPoly(x, y, poly)) out.push(i);
    }
    sel = out;
    // overlap with lassoes already saved: points of this selection that someone already put in a region
    const hit = new Set(); let shared = 0;
    for (const i of sel) {
      let any = false;
      for (const L of saved) {
        if (xs[i] < L.x0 || xs[i] > L.x1 || ys[i] < L.y0 || ys[i] > L.y1 || !inPoly(xs[i], ys[i], L.polygon)) continue;
        any = true; hit.add(L.label);
      }
      if (any) shared++;
    }
    const lb = [...hit];
    $('#lasso-n').innerHTML = `<b>${fmt(sel.length)}</b> ${kind === 'docs' ? 'documents' : kind === 'tasks' ? 'tasks (→ their documents)' : 'figures (→ their documents)'} selected` +
      (shared ? ` <span class="lasso-warn">· ${fmt(shared)} already in “${esc(lb.slice(0, 3).join('”, “'))}”${lb.length > 3 ? ` +${lb.length - 3}` : ''}</span>` : '');
    $('#lasso').querySelector('[type=submit]').disabled = !sel.length;
  }
  function finishLasso() {
    // simplify: at most ~400 vertices are kept (stored with the selection for the record)
    if (poly.length > 400) { const k = Math.ceil(poly.length / 400); poly = poly.filter((_, j) => j % k === 0); }
    if (poly.length < 3) { clearLasso(); return; }
    select(); draw();
    $('#lasso').hidden = false;
    const inp = $('#lasso-label'); inp.focus(); inp.select();
  }
  function clearLasso() { poly = null; sel = null; drawing = false; $('#lasso').hidden = true; draw(); }
  async function loadSaved() {
    let r;
    try { r = await api(`/api/labels/polygons?kind=${kind}`); } catch { return; }
    saved = r.items.map(L => {
      let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
      for (const [x, y] of L.polygon) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      const top = L.polygon.reduce((a, b) => b[1] < a[1] ? b : a);   // label sits on the topmost vertex
      return {...L, x0, y0, x1, y1, bx: top[0], by: top[1]};
    });
    savedOther = r.other_layout;
    if (poly && !drawing) select();
    $('#lasso-saved').textContent = lassoOn ? `${fmt(saved.length)} saved lasso${saved.length === 1 ? '' : 'es'} shown` +
      (savedOther ? ` · ${fmt(savedOther)} from an older map layout hidden` : '') : '';
    draw();
  }
  function setLasso(on) {
    lassoOn = on; $('#maplasso').setAttribute('aria-pressed', on); cv.classList.toggle('lassoing', on);
    clearInterval(savedTimer);
    if (on) { loadSaved(); savedTimer = setInterval(() => { if (!document.hidden) loadSaved(); }, 20000); }  // others keep drawing
    else { $('#lasso-saved').textContent = ''; if (!sel) clearLasso(); draw(); }
  }
  $('#maplasso').onclick = () => setLasso(!lassoOn);
  $('#lasso-cancel').onclick = clearLasso;
  $('#lasso').addEventListener('submit', async e => {
    e.preventDefault();
    const label = $('#lasso-label').value.trim();
    if (!label || !sel || !sel.length) return;
    const btn = $('#lasso [type=submit]'); btn.disabled = true;
    try {
      const r = await api('/api/labels', {method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({label, kind, ids: sel.map(i => ids[i]), polygon: poly})});
      toast(`“${r.label}” → ${fmt(r.n_docs)} documents (${fmt(r.n_new)} new)`);
      clearLasso(); Labels.refresh();
    } catch (err) { toast('Could not save: ' + err.message.slice(0, 120)); }
    finally { btn.disabled = false; }
  });

  function legend() {
    const tp = S.topics[TOPIC_KIND(kind)];
    const key = $('#map-key');
    if (colorBy === 'collection') { key.innerHTML = S.colls.map(c => `<span><i style="background:${COLL_COLORS(c)}"></i>${esc(c)}</span>`).join(''); return; }
    if (!tp) { key.innerHTML = '<span class="faint">topics not fitted yet</span>'; return; }
    const top = tp.counts.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
    key.innerHTML = top.map(([, i]) => `<span><i style="background:${tp.colors[i]}"></i>${esc(tp.labels[i].slice(0, 2).join(' · '))}</span>`).join('') + `<span class="faint">+${tp.k - top.length}</span>`;
  }

  function setKind(k) {
    $$('#mapkind button').forEach(x => x.setAttribute('aria-selected', x.dataset.k === k)); kind = k;
    known = new Set(); saved = []; if (lassoOn) setTimeout(loadSaved); $('#map-what').textContent = kind === 'docs' ? 'documents' : kind === 'tasks' ? 'tasks (1 in 32 sampled)' : 'figures (1 in 32 sampled)'; hl = null; clearLasso();
    return reload();
  }
  $$('#mapkind button').forEach(b => b.onclick = () => setKind(b.dataset.k));
  $$('#mapcolor button').forEach(b => b.onclick = () => {
    $$('#mapcolor button').forEach(x => x.setAttribute('aria-pressed', x === b)); colorBy = b.dataset.c; legend(); draw();
  });
  $('#mapreset').onclick = () => { view = {s: 1, x: 0, y: 0}; draw(); };
  new ResizeObserver(resize).observe(cv);
  return {
    reload, legend, draw, setKind, setLasso, clearLasso, loadSaved,
    highlight(docIds) { hl = docIds && docIds.size ? docIds : null; if (kind === 'docs') draw(); },
    get kind() { return kind; },
    get lassoPending() { return !!poly; },
    get lassoOn() { return lassoOn; },
  };
})();

// ======================================================================= search
const HINT = {
  semantic: 'Meaning, not words: passages ranked by multilingual-e5-large-instruct, grouped by document. Query in any language.',
  docs: 'Whole documents ranked by their mean passage vector. Good for “papers about…”.',
  image: 'Figures inside PDFs, ranked by SigLIP-2 text→image similarity. Try “a map”, “bar chart” or “group photo”.',
  keyword: 'Exact words (SQLite FTS5, BM25). Supports "quoted phrases", OR, NOT, prefix*.',
  tasks: 'Training tasks generated by DocMaestro from each document (OCR, layout, captions, ordering…). Leave the box empty to browse; words search their input and target. Click one to open it in its document.',
};
const taskName = t => String(t || '').replace(/_/g, ' ');

function params(offset) {
  const p = new URLSearchParams({mode: S.mode, offset, limit: S.mode === 'image' ? 36 : PAGE});
  if (S.q) p.set('q', S.q);
  if (S.like && S.mode !== 'tasks') p.set(S.like[0], S.like[1]);
  if (S.mode === 'tasks' && S.task) p.set('task', S.task);
  const f = S.filt;
  if (f.collection) p.set('collection', f.collection);
  if (f.topic !== '') p.set('topic', f.topic);
  if (f.yf) p.set('year_from', f.yf);
  if (f.yt) p.set('year_to', f.yt);
  return p;
}

function syncURL() {
  const p = new URLSearchParams();
  if (S.q) p.set('q', S.q);
  if (S.mode !== 'semantic') p.set('mode', S.mode);
  if (S.mode === 'tasks' && S.task) p.set('task', S.task);
  for (const [k, v] of Object.entries(S.filt)) if (v !== '') p.set(k, v);
  history.replaceState(null, '', (p.toString() ? '?' + p : location.pathname) + location.hash);
}

function newSearch() {
  S.gen++; S.next = 0; S.shown = 0; S.loading = false;  // a stale in-flight request must not block this one
  $('#results').innerHTML = ''; $('#igrid').innerHTML = '';
  $('#status').innerHTML = '';
  $('#modehint').textContent = HINT[S.mode];
  syncURL();
  if (!S.q && !S.like && S.mode !== 'tasks') { $('#sentinel').textContent = ''; Fig.highlight(null); return; }
  more();
}

async function more() {
  const tasks = S.mode === 'tasks';   // tasks can be browsed without a query
  if (S.loading || S.next === null || (!S.q && !S.like && !tasks)) return;
  S.loading = true;
  const gen = S.gen;
  $('#sentinel').innerHTML = '<span class="spin"></span>searching';
  try {
    const r = await api((tasks ? '/api/tasks?' : '/api/search?') + params(S.next));
    if (gen !== S.gen) return;
    S.next = r.next;
    if (tasks) renderTasks(r.items); else if (S.mode === 'image') renderTiles(r.items); else renderResults(r.items);
    S.shown += r.items.length;
    const what = tasks ? (S.q ? `“${esc(S.q)}”` : S.task ? esc(taskName(S.task)) : 'all types')
      : S.like ? `similar to ${S.like[2]}` : `“${esc(S.q)}”`;
    $('#status').innerHTML = `<span><b>${fmt(S.shown)}</b>${r.total ? ` of ${fmt(r.total)}` : ''} ${tasks ? 'tasks' : S.mode === 'image' ? 'figures' : 'documents'} for ${what}</span>` +
      (r.searched ? `<span>over ${fmt(r.searched)} ${S.mode === 'image' ? 'figures' : S.mode === 'docs' ? 'documents' : 'passages'} indexed so far</span>` : '') +
      `<span>${r.ms} ms</span>` + (S.like ? '<button class="ghost" id="unlike">✕ clear</button>' : '');
    if ($('#unlike')) $('#unlike').onclick = () => { S.like = null; newSearch(); };
    $('#sentinel').textContent = S.next === null ? (S.shown ? '— end —' : 'Nothing found (yet: the index is still growing).') : '';
    if (S.next !== null) {   // also loads on demand when auto-loading is off (see canAuto)
      $('#sentinel').innerHTML = '<button class="ghost mono" id="loadmore">Show more ↓</button>';
      $('#loadmore').onclick = () => more();
    }
    if (S.mode !== 'image') Fig.highlight(new Set($$('#results [data-doc]').map(e => +e.dataset.doc)));
  } catch (e) {
    if (gen === S.gen) $('#sentinel').textContent = 'Error: ' + e.message.slice(0, 200);
  } finally {
    if (gen === S.gen) S.loading = false;
    // the sentinel may still be visible after a short page -> keep filling
    if (gen === S.gen && S.next !== null && canAuto() && isVisible($('#sentinel'))) more();
  }
}
// Infinite scroll only up to AUTO_MAX results. Beyond that, a "Show more" button: otherwise the list keeps growing
// whenever its end is near the screen, and the sections under it (topics, collections…) can never be reached.
// It also pauses while the next section is on screen, i.e. after a jump with the §02, §03… links.
const AUTO_MAX = 60;
// A jump with the §0x links scrolls past the end of the list: loading more then would push the target section away
// mid-scroll, so loading waits until that scroll is over (and, after it, the target section is on screen).
let navUntil = 0;
document.addEventListener('click', e => { const a = e.target.closest('a[href^="#"]'); if (a && a.getAttribute('href') !== '#search') navUntil = Date.now() + 2000; }, true);
addEventListener('hashchange', () => { if (location.hash !== '#search') navUntil = Date.now() + 2000; });
const canAuto = () => S.shown < AUTO_MAX && Date.now() > navUntil && $('#topics').getBoundingClientRect().top > innerHeight;
const isVisible = el => { const r = el.getBoundingClientRect(); return r.top < innerHeight + 600 && r.bottom > 0; };
new IntersectionObserver(es => es.some(e => e.isIntersecting) && canAuto() && more(), {rootMargin: '600px'}).observe($('#sentinel'));

function highlight(text, q) {
  let t = esc(text).replace(/\u0002/g, '<mark>').replace(/\u0003/g, '</mark>');
  if (S.mode === 'keyword' || !q) return t;
  const words = q.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(w => w.length > 3);
  if (!words.length) return t;
  const re = new RegExp(`(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'giu');
  return t.replace(re, '<mark>$1</mark>');
}

function renderResults(items) {
  const ol = $('#results'), frag = document.createDocumentFragment();
  items.forEach((it, k) => {
    const d = it.doc, li = document.createElement('li');
    const firstPage = it.hits.length ? Math.max(0, it.hits[0].page) : 0;
    li.innerHTML = `<div class="res new" style="--i:${k}">
      <div class="thumb" data-open="${d.doc_id}" data-page="${firstPage}"><img alt=""></div>
      <div>
        <a class="res-doc" data-doc="${d.doc_id}" data-open="${d.doc_id}" data-page="${firstPage}">${esc(d.title)}</a>
        <div class="res-meta mono"><span>${esc(d.collection)}</span><span>${d.year ?? 's.d.'}</span><span>${fmt(d.n_pages)} pp</span>${d.n_images ? `<span>${fmt(d.n_images)} figs</span>` : ''}
          <span class="tp"><i style="background:${topicColor('text', d.topic)}"></i>${esc(topicName('text', d.topic, 2))}</span><span>${it.score}</span></div>
        <div class="hits">${it.hits.map(h => `<div class="hit"><span class="pg mono" data-open="${d.doc_id}" data-page="${Math.max(0, h.page)}">${h.kind === 1 ? 'meta' : 'p.' + (h.page + 1)}</span>
          <span class="tx">${highlight(h.text, S.q)}</span><button class="more mono" data-like="${h.chunk}">similar →</button></div>`).join('')}</div>
      </div></div>`;
    lazyImg($('img', li), pageSrc(d.doc_id, firstPage, 96));
    frag.appendChild(li);
  });
  ol.appendChild(frag);
}

function renderTiles(items) {
  const g = $('#igrid'), frag = document.createDocumentFragment();
  items.forEach((it, k) => {
    const m = it.image, el = document.createElement('div');
    el.className = 'tile new'; el.style.setProperty('--i', k % 36);
    el.dataset.open = it.doc.doc_id; el.dataset.page = m.page;
    el.innerHTML = `<img alt=""><span class="cap2 mono"><b>${esc(it.doc.title)}</b>${esc(it.doc.collection)} · p.${m.page + 1} · ${it.score}</span>
      <button class="ghost sim mono" data-likeimg="${m.id}">similar</button>`;
    lazyImg($('img', el), imgSrc(m.id, 200));
    frag.appendChild(el);
  });
  g.appendChild(frag);
}

// one row per task: where it comes from (page or crop), what the model sees, what it must answer.
// Clicking the row opens the document at that page, like a figure tile does.
function renderTasks(items) {
  const ol = $('#results'), frag = document.createDocumentFragment();
  items.forEach((it, k) => {
    const t = it.task, d = it.doc, li = document.createElement('li');
    const page = t.page >= 0 ? t.page : (t.pages[0] ?? 0);
    const where = t.page >= 0 ? `p.${t.page + 1}` : t.pages.length ? `${t.pages.length} pages` : 'document';
    li.innerHTML = `<div class="res task new" style="--i:${k}">
      <div class="thumb" data-open="${d.doc_id}" data-page="${page}" data-task="${t.id}"><img alt=""></div>
      <div>
        <a class="res-doc" data-doc="${d.doc_id}" data-open="${d.doc_id}" data-page="${page}" data-task="${t.id}">${esc(taskName(t.task))}<span class="faint"> · ${esc(d.title)}</span></a>
        <div class="res-meta mono"><span>${esc(d.collection)}</span><span>${d.year ?? 's.d.'}</span><span>${where}</span>${t.n_figures ? `<span>${t.n_figures} crop${t.n_figures > 1 ? 's' : ''}</span>` : ''}<span>#${t.id}</span></div>
        <div class="hits">${t.input ? `<div class="hit"><span class="pg mono" data-open="${d.doc_id}" data-page="${page}" data-task="${t.id}">input</span><span class="tx">${highlight(t.input, S.q)}</span></div>` : ''}
          <div class="hit"><span class="pg mono" data-open="${d.doc_id}" data-page="${page}" data-task="${t.id}">target</span><span class="tx">${highlight(t.target, S.q)}</span></div></div>
      </div></div>`;
    lazyImg($('img', li), t.n_figures ? `/api/task/${t.id}/crop/0.jpg` : pageSrc(d.doc_id, page, 96));
    frag.appendChild(li);
  });
  ol.appendChild(frag);
}

async function loadTaskTypes() {   // the Tasks mode only appears once tasks have been imported (pdfatlas.tasks)
  const r = await api('/api/tasks/types').catch(() => ({types: []}));
  if (!r.types.length) return;
  $('#modes [data-m="tasks"]').hidden = false;
  if (S.topics.tasks) $('#mapkind [data-k="tasks"]').hidden = false;   // after `pdfatlas.tasks --map`
  $('#f-task').innerHTML = '<option value="">All</option>' +
    r.types.map(x => `<option value="${esc(x.task)}">${esc(taskName(x.task))} (${fmt(x.n)})</option>`).join('');
  $('#f-task').value = S.task || '';
}

function filtersChanged() {
  S.filt = {collection: $('#f-coll').value, topic: $('#f-topic').value, yf: $('#f-y0').value.trim(), yt: $('#f-y1').value.trim()};
  $('#f-clear').hidden = !Object.values(S.filt).some(v => v !== '');
  newSearch(); Browse.reset();
}

function setMode(m) {
  S.mode = m;
  $$('#modes button').forEach(b => b.setAttribute('aria-selected', b.dataset.m === m));
  if ((m === 'image') !== (S.lastTopicKind === 'image')) { $('#f-topic').value = ''; S.filt.topic = ''; }
  S.lastTopicKind = m === 'image' ? 'image' : 'text';
  $('#f-taskwrap').hidden = m !== 'tasks';
  fillFilters();
  newSearch();
}

// ======================================================================= browse (keyset pagination)
const Browse = (() => {
  let cursor = -1, loading = false, gen = 0, done = false;
  async function more() {
    if (loading || done) return;
    loading = true; const g = gen;
    $('#bsentinel').innerHTML = '<span class="spin"></span>loading';
    const p = new URLSearchParams({cursor, limit: 40});
    const f = S.filt;
    if (f.collection) p.set('collection', f.collection);
    if (f.topic !== '' && S.mode !== 'image') p.set('topic', f.topic);
    if (f.yf) p.set('year_from', f.yf);
    if (f.yt) p.set('year_to', f.yt);
    try {
      const r = await api('/api/docs?' + p);
      if (g !== gen) return;
      $('#browsebody').insertAdjacentHTML('beforeend', r.items.map(d => `<tr data-open="${d.doc_id}">
        <td>${d.doc_id}</td><td>${esc(d.title)}</td><td class="mono">${esc(d.collection)}</td><td>${d.year ?? ''}</td>
        <td>${fmt(d.n_pages)}</td><td>${fmt(d.n_images)}</td>
        <td class="tp"><i style="background:${topicColor('text', d.topic)}"></i>${d.topic >= 0 ? 'T' + String(d.topic).padStart(2, '0') : '—'}</td></tr>`).join(''));
      cursor = r.next; done = r.next === null;
      $('#bsentinel').textContent = done ? '— end of what is indexed so far —' : '';
    } catch (e) { $('#bsentinel').textContent = 'Index not ready yet.'; }
    finally { if (g === gen) loading = false; if (!done && isVisible($('#bsentinel'))) more(); }
  }
  new IntersectionObserver(es => es.some(e => e.isIntersecting) && more(), {rootMargin: '600px'}).observe($('#bsentinel'));
  return {more, reset() { gen++; cursor = -1; done = false; loading = false; $('#browsebody').innerHTML = ''; more(); }};
})();

// ======================================================================= labels (§05)
const Labels = (() => {
  let shown = null;   // label currently lit up on the map
  const ago = t => { const m = (Date.now() / 1000 - t) / 60; return m < 1 ? 'just now' : m < 60 ? `${m | 0} min ago` : m < 1440 ? `${m / 60 | 0} h ago` : new Date(t * 1000).toLocaleDateString(); };
  async function refresh() {
    let r;
    if (Fig.lassoOn) Fig.loadSaved();   // undo / delete / new lasso: keep the outlines on the map in sync
    try { r = await api('/api/labels'); } catch { return; }
    $('#labelnames').innerHTML = r.labels.map(l => `<option value="${esc(l.label)}">`).join('');
    $('#labellist').innerHTML = r.labels.length ? r.labels.map(l => `<li data-label="${esc(l.label)}"${l.label === shown ? ' class="cur"' : ''}>
        <span class="nm" data-show>${esc(l.label)}</span><span class="n">${fmt(l.n_docs)} docs · ${l.n_selections} sel.</span>
        <button data-dellabel title="Remove this label from every document">✕</button></li>`).join('')
      : '<li><span class="faint mono">No labels yet. Draw on Fig. 1 with Lasso ✎ or Shift+drag.</span></li>';
    $('#sellist').innerHTML = r.selections.map(x => `<li data-sel="${x.id}">
        <span class="nm" data-show-label="${esc(x.label)}">${esc(x.label)}</span>
        <span class="n">${fmt(x.n_docs)} docs · ${x.kind === 'docs' ? 'doc map' : x.kind === 'images' ? 'figure map' : x.kind} · ${ago(x.t)}</span>
        <button data-undo title="Undo this selection">undo</button></li>`).join('');
  }
  async function show(label) {
    if (shown === label) { shown = null; Fig.highlight(null); refresh(); return; }
    const r = await api('/api/labels/docs?label=' + encodeURIComponent(label));
    shown = label;
    if (Fig.kind !== 'docs') await Fig.setKind('docs');
    Fig.highlight(new Set(r.doc_ids));
    location.hash = '#top';
    toast(`“${label}”: ${fmt(r.doc_ids.length)} documents lit up`);
    refresh();
  }
  $('#labels').addEventListener('click', async e => {
    const li = e.target.closest('li');
    if (!li) return;
    if (e.target.closest('[data-show]')) return show(li.dataset.label);
    if (e.target.closest('[data-show-label]')) return show(e.target.closest('[data-show-label]').dataset.showLabel);
    if (e.target.closest('[data-dellabel]')) {
      const l = li.dataset.label;
      if (!confirm(`Remove the label “${l}” from every document?`)) return;
      await api('/api/labels?label=' + encodeURIComponent(l), {method: 'DELETE'});
      if (shown === l) { shown = null; Fig.highlight(null); }
      toast(`Removed “${l}”`); return refresh();
    }
    if (e.target.closest('[data-undo]')) {
      const r = await api('/api/labels?selection=' + li.dataset.sel, {method: 'DELETE'});
      toast(`Undone (${fmt(r.removed)} document labels removed)`); return refresh();
    }
  });
  return {refresh};
})();

// ======================================================================= sources (§06)
const Sources = (() => {
  const msg = (t, bad) => { const m = $('#srcmsg'); m.textContent = t; m.classList.toggle('bad', !!bad); };
  async function refresh() {
    let r;
    try { r = await api('/api/sources'); } catch { return; }
    $('#srclist').innerHTML = r.sources.map(s => `<li data-path="${esc(s.path)}">
        <span class="nm" title="${esc(s.path)}">${s.added ? '<span class="red">+</span> ' : ''}${esc(s.name)}</span>
        <span class="n">${esc(s.path)}</span>${s.added ? '<button data-unwatch title="Stop watching (indexed documents stay)">✕</button>' : ''}</li>`).join('')
      || '<li><span class="faint mono">No sources configured.</span></li>';
    if (!r.admin) msg('Adding folders is disabled until a password is set on the server (python -m pdfatlas.admin set-password).', true);
  }
  async function send(url, body) {
    const r = await fetch(url, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    return j;
  }
  $('#srcform').addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#srcform [type=submit]'); btn.disabled = true; msg('checking…');
    try {
      const r = await send('/api/sources', {path: $('#src-path').value.trim(), name: $('#src-name').value.trim(), password: $('#src-pw').value});
      msg(`Added “${r.name}”: ${fmt(r.n_pdfs)}${r.n_complete ? '' : '+'} PDFs found. The watcher will index them within a few minutes.`);
      $('#src-path').value = ''; $('#src-name').value = ''; $('#src-pw').value = '';
      refresh();
    } catch (err) { msg(err.message, true); }
    finally { btn.disabled = false; }
  });
  $('#srclist').addEventListener('click', async e => {
    if (!e.target.closest('[data-unwatch]')) return;
    const path = e.target.closest('li').dataset.path;
    if (!$('#src-pw').value) { msg('Type the password above, then click ✕ again.', true); $('#src-pw').focus(); return; }
    if (!confirm(`Stop watching ${path}? Documents already indexed stay searchable.`)) return;
    try { await send('/api/sources/remove', {path, password: $('#src-pw').value}); $('#src-pw').value = ''; msg('Stopped watching ' + path); refresh(); }
    catch (err) { msg(err.message, true); }
  });
  return {refresh};
})();

// ======================================================================= viewer
const Viewer = (() => {
  const V = $('#viewer');
  let doc = null, page = 0, hitPages = new Set(), showText = false;
  const stripObs = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    stripObs.unobserve(e.target);
    lazyImg($('img', e.target), pageSrc(doc.doc_id, +e.target.dataset.p, 80));
  }), {root: $('#v-strip'), rootMargin: '400px'});

  let task = null;   // the DocMaestro task shown in the side panel (opened from the tasks map / search, or the list)
  async function open(id, p = 0, hits = [], taskId = null) {
    V.hidden = false; requestAnimationFrame(() => V.classList.add('on'));
    document.body.style.overflow = 'hidden';
    $('#v-title').textContent = '…'; $('#v-strip').innerHTML = ''; $('#v-imgs').innerHTML = ''; $('#v-sim').innerHTML = '';
    $('#v-tasks').innerHTML = ''; $('#v-taskcap').hidden = true; hideTask();
    try { doc = await api(`/api/doc/${id}`); } catch (e) { $('#v-title').textContent = 'Not indexed yet'; return; }
    loadTasks(doc.doc_id);
    if (taskId != null) showTask(taskId);
    hitPages = new Set(hits);
    $('#v-sub').textContent = `${doc.collection} · ${doc.year ?? 's.d.'} · ${fmt(doc.n_pages)} pages · ${fmt(doc.n_images)} figures · #${doc.doc_id}`;
    $('#v-title').textContent = doc.title;
    $('#v-links').innerHTML = `<a href="/api/pdf/${doc.doc_id}#page=${p + 1}" target="_blank" id="v-pdf"><span>↗</span> PDF</a>` +
      (doc.url ? `<a href="${esc(doc.url)}" target="_blank" rel="noopener"><span>↗</span> Source</a>` : '') +
      `<button id="v-likedoc"><span>→</span> Similar documents in search</button>` +
      `<span class="faint">${esc(topicName('text', doc.topic, 4))}</span>`;
    renderLabels();
    $('#v-likedoc').onclick = () => { close(); S.like = ['like_doc', doc.doc_id, esc(doc.title.slice(0, 40))]; setMode('docs'); location.hash = '#search'; };
    const np = Math.min(doc.n_pages, 600), frag = document.createDocumentFragment();
    for (let i = 0; i < np; i++) {
      const b = document.createElement('button'); b.dataset.p = i;
      b.innerHTML = `<img alt=""><span>${i + 1}</span>`;
      if (hitPages.has(i)) b.classList.add('hit');
      b.onclick = () => go(i);
      frag.appendChild(b); stripObs.observe(b);
    }
    $('#v-strip').appendChild(frag);
    $('#v-imgcap').textContent = doc.images.length ? `Figures in this document (${doc.images.length}${doc.images.length >= 400 ? '+' : ''})` : 'No embedded figures';
    doc.images.slice(0, 120).forEach(im => {
      const b = document.createElement('button'); b.title = `p.${im.page + 1} · ${topicName('image', im.topic)}`;
      b.appendChild(lazyImg(new Image(), imgSrc(im.id, 100)));
      b.onclick = () => go(im.page);
      b.ondblclick = () => { close(); S.like = ['like_image', im.id, 'a figure']; setMode('image'); location.hash = '#search'; };
      $('#v-imgs').appendChild(b);
    });
    go(Math.min(p, np - 1));
    api(`/api/search?mode=docs&like_doc=${doc.doc_id}&limit=10`).then(r => {
      $('#v-sim').innerHTML = r.items.map(it => `<li data-open="${it.doc.doc_id}">${esc(it.doc.title)}<span class="mono">${esc(it.doc.collection)} · ${it.doc.year ?? 's.d.'} · ${it.score}</span></li>`).join('');
    }).catch(() => {});
  }

  // ---- DocMaestro tasks of this document
  const pretty = s => { try { return JSON.stringify(JSON.parse(s), null, 1); } catch { return s; } };
  async function showTask(tid) {
    let t;
    try { t = await api(`/api/task/${tid}`); } catch { return; }
    if (!doc || t.doc_id !== doc.doc_id) return;
    task = t;
    const box = $('#v-task'), pg = t.page >= 0 ? t.page : (t.pages[0] ?? 0);
    const where = t.page >= 0 ? `p.${t.page + 1}` : t.pages.length ? `pages ${t.pages.map(x => x + 1).join(', ')}` : 'whole document';
    box.innerHTML = `<header><div><b>${esc(taskName(t.task))}</b> <span class="mono faint">${where} · #${t.id}</span></div>
        <button class="mono" id="v-taskclose" title="Close the task">✕</button></header>
      ${t.n_figures ? `<div class="mono lab">figure${t.n_figures > 1 ? 's' : ''} given to the model</div><div class="crops"></div>` : ''}
      <div class="mono lab">input</div><pre>${esc(pretty(t.input)) || '<span class="faint">(the page image only)</span>'}</pre>
      <div class="mono lab">target</div><pre>${esc(pretty(t.target))}</pre>`;
    for (let k = 0; k < Math.min(t.n_figures, 12); k++) $('.crops', box).appendChild(lazyImg(new Image(), `/api/task/${t.id}/crop/${k}.jpg`));
    $('#v-taskclose').onclick = hideTask;
    box.hidden = false; $('.v-side').scrollTop = 0;
    $$('#v-tasks button').forEach(b => b.classList.toggle('cur', +b.dataset.task === t.id));
    if (pg !== page) go(pg); else boxes();
  }
  function hideTask() { task = null; $('#v-task').hidden = true; $('#v-task').innerHTML = ''; $$('#v-tasks button.cur').forEach(b => b.classList.remove('cur')); if (doc) boxes(); }

  async function loadTasks(docId) {
    let r;
    try { r = await api(`/api/doc/${docId}/tasks`); } catch { return; }
    if (!doc || doc.doc_id !== docId || !r.total) return;
    const cap = $('#v-taskcap'); cap.hidden = false;
    cap.textContent = `Tasks from this document (${fmt(r.total)})`;
    const by = {};
    r.items.forEach(it => (by[it.task] ||= []).push(it));
    const box = $('#v-tasks');
    box.innerHTML = r.types.map(t => `<details data-type="${esc(t.task)}"><summary><span>${esc(taskName(t.task))}</span><span class="mono">${fmt(t.n)}</span></summary></details>`).join('');
    $$('details', box).forEach(d => d.addEventListener('toggle', () => {   // items built on first open: some documents have thousands
      if (!d.open || d.dataset.built) return;
      d.dataset.built = 1;
      d.insertAdjacentHTML('beforeend', (by[d.dataset.type] || []).map(it =>
        `<button data-task="${it.id}" class="${task && task.id === it.id ? 'cur' : ''}"><span class="mono">${it.page >= 0 ? 'p.' + (it.page + 1) : 'doc'}</span>${esc(it.preview || '(no text)')}</button>`).join(''));
    }));
    box.onclick = e => { const b = e.target.closest('[data-task]'); if (b) showTask(+b.dataset.task); };
  }

  function renderLabels() {
    const box = $('#v-labels');
    box.innerHTML = (doc.labels || []).map(l => `<span class="chip">${esc(l)}<button data-unlabel="${esc(l)}" title="Remove from this document">✕</button></span>`).join('') +
      '<input id="v-addlabel" list="labelnames" maxlength="80" placeholder="+ label" aria-label="Add a label to this document">';
    $$('[data-unlabel]', box).forEach(b => b.onclick = async () => {
      const l = b.dataset.unlabel;
      await api(`/api/labels?label=${encodeURIComponent(l)}&doc_id=${doc.doc_id}`, {method: 'DELETE'}).catch(() => {});
      doc.labels = doc.labels.filter(x => x !== l); renderLabels(); Labels.refresh();
    });
    $('#v-addlabel').onkeydown = async e => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      const l = e.target.value.trim();
      if (!l) return;
      try {
        const r = await api('/api/labels', {method: 'POST', headers: {'content-type': 'application/json'},
          body: JSON.stringify({label: l, kind: 'docs', ids: [doc.doc_id], note: 'viewer'})});
        if (!doc.labels.includes(r.label)) doc.labels.push(r.label);
        renderLabels(); Labels.refresh(); $('#v-addlabel').focus();
      } catch (err) { toast('Could not save: ' + err.message.slice(0, 120)); }
    };
  }

  function go(p) {
    if (!doc) return;
    page = Math.max(0, Math.min(p, Math.min(doc.n_pages, 600) - 1));
    $('#v-pno').textContent = `page ${page + 1} / ${fmt(doc.n_pages)}`;
    $$('#v-strip button').forEach(b => b.classList.toggle('cur', +b.dataset.p === page));
    const cur = $(`#v-strip button[data-p="${page}"]`); cur && cur.scrollIntoView({block: 'nearest'});
    $('#v-pdf') && ($('#v-pdf').href = `/api/pdf/${doc.doc_id}#page=${page + 1}`);
    const img = $('#v-img'), w = Math.min(1600, $('#v-canvas').clientWidth - 32);
    img.classList.add('loading');
    img.onload = () => { img.classList.remove('loading'); boxes(); };
    img.src = pageSrc(doc.doc_id, page, w);
    if (showText) text();
  }

  function boxes() {   // figure boxes from the PDF object tree, clickable -> similar figures
    const img = $('#v-img'), bx = $('#v-boxes');
    bx.style.left = img.offsetLeft + 'px'; bx.style.top = img.offsetTop + 'px';
    bx.style.width = img.clientWidth + 'px'; bx.style.height = img.clientHeight + 'px';
    bx.innerHTML = doc.images.filter(i => i.page === page).map(i =>
      `<i data-likeimg="${i.id}" title="Find similar figures" style="left:${i.x0 * 100}%;top:${i.y0 * 100}%;width:${(i.x1 - i.x0) * 100}%;height:${(i.y1 - i.y0) * 100}%"></i>`).join('') +
      (task && task.region && task.page === page ? (([x0, y0, x1, y1]) =>
        `<i class="treg" style="left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%"></i>`)(task.region) : '');
  }

  async function text() {
    const r = await api(`/api/doc/${doc.doc_id}/text?page=${page}`).catch(() => ({chunks: []}));
    $('#v-text').innerHTML = r.chunks.length ? r.chunks.map(c => `<div class="chunk">${esc(c.text)}<button class="ghost mono" data-like="${c.id}">similar passages →</button></div>`).join('')
      : '<p class="mono faint">No text layer on this page (scanned image or empty). OCR is deliberately not run.</p>';
  }

  function close() {
    V.classList.remove('on'); document.body.style.overflow = '';
    setTimeout(() => { V.hidden = true; $('#v-img').removeAttribute('src'); }, 260);
  }
  $('#v-prev').onclick = () => go(page - 1);
  $('#v-next').onclick = () => go(page + 1);
  $('#v-textbtn').onclick = () => {
    showText = !showText; $('#v-textbtn').setAttribute('aria-pressed', showText);
    $('#v-text').hidden = !showText; $('#v-canvas').hidden = showText; if (showText) text();
  };
  V.addEventListener('click', e => { if (e.target.closest('[data-close]')) close(); });
  document.addEventListener('keydown', e => {
    if (V.hidden) return;
    if (e.key === 'Escape') close();
    if (document.activeElement.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight' || e.key === 'j') go(page + 1);
    if (e.key === 'ArrowLeft' || e.key === 'k') go(page - 1);
  });
  return {open, close, get open_() { return !V.hidden; }};
})();

// ======================================================================= global wiring
document.addEventListener('click', e => {
  const like = e.target.closest('[data-like]');
  if (like) { e.stopPropagation(); Viewer.close(); S.like = ['like_chunk', +like.dataset.like, 'a passage']; setMode('semantic'); location.hash = '#search'; return; }
  const li = e.target.closest('[data-likeimg]');
  if (li) { e.stopPropagation(); Viewer.close(); S.like = ['like_image', +li.dataset.likeimg, 'a figure']; setMode('image'); location.hash = '#search'; return; }
  const o = e.target.closest('[data-open]');
  if (o) {
    const docId = +o.dataset.open, pg = +(o.dataset.page || 0);
    const res = o.closest('.res');
    const hits = res ? $$('.pg', res).map(x => +x.dataset.page) : [];
    Viewer.open(docId, pg, hits, o.dataset.task ? +o.dataset.task : null);
    return;
  }
  const tp = e.target.closest('.topic');
  if (tp) {
    const k = tp.dataset.k;
    if (topicKind === 'image') { setMode('image'); }
    else if (S.mode === 'image') setMode('semantic');
    fillFilters(); $('#f-topic').value = k; filtersChanged();
    location.hash = S.q || S.like ? '#search' : '#browse';
    toast(`Filtered to topic T${String(k).padStart(2, '0')}`);
    return;
  }
  const cb = e.target.closest('#collbars li');
  if (cb) { $('#f-coll').value = cb.dataset.c; filtersChanged(); location.hash = '#browse'; }
});

let qT = 0;
$('#q').addEventListener('input', () => {
  clearTimeout(qT);
  qT = setTimeout(() => { const v = $('#q').value.trim(); if (v !== S.q) { S.q = v; S.like = null; newSearch(); } }, 350);
});
$('#qform').addEventListener('submit', e => { e.preventDefault(); clearTimeout(qT); S.q = $('#q').value.trim(); S.like = null; newSearch(); $('#q').blur(); });
$$('#modes button').forEach(b => b.onclick = () => setMode(b.dataset.m));
['#f-coll', '#f-topic'].forEach(s => $(s).onchange = filtersChanged);
$('#f-task').onchange = () => { S.task = $('#f-task').value; newSearch(); };
['#f-y0', '#f-y1'].forEach(s => $(s).addEventListener('change', filtersChanged));
$('#f-clear').onclick = () => { $('#f-coll').value = ''; $('#f-topic').value = ''; $('#f-y0').value = ''; $('#f-y1').value = ''; filtersChanged(); };
$$('#topickind button').forEach(b => b.onclick = () => { $$('#topickind button').forEach(x => x.setAttribute('aria-pressed', x === b)); topicKind = b.dataset.k; renderTopics(); });
$$('.hero-links [data-mode]').forEach(a => a.addEventListener('click', () => setMode(a.dataset.mode)));
$('#idxchip').onclick = () => openProgress($('#progress').hidden);
$('[data-pclose]').onclick = () => openProgress(false);
document.addEventListener('keydown', e => {
  if (e.key === '/' && !Viewer.open_ && document.activeElement.tagName !== 'INPUT') { e.preventDefault(); location.hash = '#search'; $('#q').focus(); }
  if (e.key === 'Escape' && !$('#progress').hidden) openProgress(false);
  if (e.key === '0' && document.activeElement.tagName !== 'INPUT') $('#mapreset').click();
  if ((e.key === 'l' || e.key === 'L') && !Viewer.open_ && document.activeElement.tagName !== 'INPUT') Fig.setLasso(!Fig.lassoOn);
  if (e.key === 'Escape' && Viewer.open_ === false && Fig.lassoPending) Fig.clearLasso();
});
window.addEventListener('resize', () => { clearTimeout(window._rt); window._rt = setTimeout(() => S.stats && renderYears(), 150); });

// boot: restore URL state
const u = new URLSearchParams(location.search);
S.q = u.get('q') || ''; $('#q').value = S.q;
S.task = u.get('task') || '';
for (const k of ['collection', 'topic', 'yf', 'yt']) if (u.get(k)) S.filt[k] = u.get(k);
$('#modehint').textContent = HINT.semantic;
loadStats().then(loadTaskTypes).then(() => {
  $('#f-coll').value = S.filt.collection; $('#f-topic').value = S.filt.topic; $('#f-y0').value = S.filt.yf; $('#f-y1').value = S.filt.yt;
  setMode(u.get('mode') || 'semantic');
  Browse.reset();
  Fig.reload();
  Labels.refresh();
  Sources.refresh();
});
pollProgress();
})();
