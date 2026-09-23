import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => { const r = await fetch(path, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : (typeof v === 'number' ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d)) : String(v));
const PAL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#393b79', '#ad494a'];

// ---------------------------------------------------------------- navigation
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('active')); b.classList.add('active');
  document.querySelectorAll('section.panel').forEach(s => s.classList.remove('active')); $('p-' + b.dataset.p).classList.add('active');
  if (b.dataset.p === 'fracture') setTimeout(resizeViewer, 50);
  if (['results', 'fracture', 'database', 'top'].includes(b.dataset.p)) loadRuns().then(() => { if (b.dataset.p === 'top') api('/api/top').then(renderTop); });
});

// ---------------------------------------------------------------- status
async function refreshStatus() {
  try {
    const s = await api('/api/status');
    const e = s.force_engine;
    $('e-pot').textContent = e.potential; $('e-impl').textContent = e.implementation.split(' (')[0]; $('e-dev').textContent = e.device;
    $('e-prec').textContent = e.precision_fast + ' (fast) / ' + e.precision_reference + ' (reference)';
    const ok = e.status === 'ready'; $('e-status').textContent = ok ? 'engine ready · Atomistica ' + (e.reference_implementation.available ? 'available' : 'MISSING') : e.status;
    $('e-status').className = 'badge ' + (ok ? 'ok' : 'bad');
    $('e-val').textContent = `validation ${s.validation.n_tests - s.validation.n_fail}/${s.validation.n_tests} pass`; $('e-val').className = 'badge ' + (s.validation.essential_pass ? 'ok' : 'warn');
  } catch (err) { $('e-status').textContent = 'backend unreachable: ' + err.message; $('e-status').className = 'badge bad'; }
}
refreshStatus(); setInterval(refreshStatus, 15000);

// ---------------------------------------------------------------- canvas plotting helper
function plot(canvas, series, o = {}) {
  const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height; ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fbfbfb'; ctx.fillRect(0, 0, W, H);
  const m = { l: 70, r: 20, t: 20, b: 50 };
  let xs = [], ys = []; series.forEach(s => { xs = xs.concat(s.x); ys = ys.concat(s.y); });
  xs = xs.filter(Number.isFinite); ys = ys.filter(Number.isFinite); if (!xs.length) { ctx.fillStyle = '#666'; ctx.fillText('no data', W / 2 - 20, H / 2); return null; }
  let x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs), y0 = o.ymin ?? Math.min(0, Math.min(...ys)), y1 = o.ymax ?? Math.max(...ys);
  if (x1 <= x0) x1 = x0 + 1; if (y1 <= y0) y1 = y0 + 1; const py = (y1 - y0) * 0.05; y1 += py; const px = (x1 - x0) * 0.03; x1 += px;
  const X = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), Y = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.strokeRect(m.l, m.t, W - m.l - m.r, H - m.t - m.b);
  ctx.fillStyle = '#333'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  const ticks = (a, b, n) => { const step = niceStep((b - a) / n); const t = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-12; v += step) t.push(+v.toFixed(10)); return t; };
  ticks(x0, x1, 6).forEach(v => { ctx.fillText(fmtTick(v), X(v), H - m.b + 16); ctx.beginPath(); ctx.moveTo(X(v), H - m.b); ctx.lineTo(X(v), H - m.b + 4); ctx.stroke(); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(X(v), m.t); ctx.lineTo(X(v), H - m.b); ctx.stroke(); ctx.strokeStyle = '#333'; });
  ctx.textAlign = 'right'; ticks(y0, y1, 6).forEach(v => { ctx.fillText(fmtTick(v), m.l - 6, Y(v) + 4); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(W - m.r, Y(v)); ctx.stroke(); ctx.strokeStyle = '#333'; });
  ctx.textAlign = 'center'; ctx.fillText(o.xlabel || '', (m.l + W - m.r) / 2, H - 8); ctx.save(); ctx.translate(14, (m.t + H - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(o.ylabel || '', 0, 0); ctx.restore();
  series.forEach((s, i) => {
    ctx.strokeStyle = s.color || PAL[i % PAL.length]; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = s.lw || 1.5;
    if (s.line !== false) { ctx.beginPath(); let started = false; for (let k = 0; k < s.x.length; k++) { if (!Number.isFinite(s.y[k])) continue; if (!started) { ctx.moveTo(X(s.x[k]), Y(s.y[k])); started = true; } else ctx.lineTo(X(s.x[k]), Y(s.y[k])); } ctx.stroke(); }
    if (s.markers) for (let k = 0; k < s.x.length; k++) { if (!Number.isFinite(s.y[k])) continue; ctx.beginPath(); ctx.arc(X(s.x[k]), Y(s.y[k]), s.r || 3, 0, 6.283); ctx.fill(); }
    if (s.labels) { ctx.font = '10px sans-serif'; for (let k = 0; k < s.x.length; k++) ctx.fillText(s.labels[k], X(s.x[k]), Y(s.y[k]) - 6); ctx.font = '12px sans-serif'; }
  });
  if (o.vlines) o.vlines.forEach(v => { ctx.strokeStyle = v.color || '#999'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(X(v.x), m.t); ctx.lineTo(X(v.x), H - m.b); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = v.color || '#999'; ctx.fillText(v.label || '', X(v.x), m.t + 10); });
  if (o.marker) { ctx.fillStyle = '#e63946'; ctx.beginPath(); ctx.arc(X(o.marker[0]), Y(o.marker[1]), 6, 0, 6.283); ctx.fill(); }
  return { X, Y };
}
function niceStep(r) { const p = Math.pow(10, Math.floor(Math.log10(r || 1))); const f = r / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }
function fmtTick(v) { return Math.abs(v) < 1e-9 ? '0' : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2)); }

// ---------------------------------------------------------------- interpretation
let INTERP = null;
async function loadInterp() {
  INTERP = await api('/api/interpretation');
  $('ref-imgs').innerHTML = INTERP.images.map(im => `<div><img src="/files/${im.file}" title="${im.short}"><div class="muted">${im.id}: ${im.short}</div></div>`).join('');
  $('img-list').innerHTML = INTERP.images.map(im => `<div class="card"><b>${im.id} – ${im.short}</b><label>visual features</label><textarea data-img="${im.id}" data-k="visual_features">${im.visual_features}</textarea><div class="muted">principles: ${im.principles.join('; ')}</div></div>`).join('');
  $('principles').innerHTML = Object.entries(INTERP.principles).map(([k, p]) => `<div class="card"><b>${k}: ${p.name}</b>` + ['abstraction', 'atomistic_realisation', 'assumptions', 'alternatives'].map(f => `<label>${f.replace('_', ' ')}</label><textarea data-p="${k}" data-k="${f}">${p[f]}</textarea>`).join('') + `<div class="muted">families: ${Object.entries(INTERP.families).filter(([f, ps]) => ps.some(x => x.startsWith(k))).map(([f]) => `<a href="#" data-fam="${f}" class="famlink">${f}</a>`).join(', ') || '–'}</div></div>`).join('');
  document.querySelectorAll('.famlink').forEach(a => a.onclick = (ev) => { ev.preventDefault(); $('family').value = a.dataset.fam; buildParams(); document.querySelector('nav button[data-p=generate]').click(); });
}
$('save-interp').onclick = async () => {
  document.querySelectorAll('#img-list textarea').forEach(t => { INTERP.images.find(i => i.id === t.dataset.img)[t.dataset.k] = t.value; });
  document.querySelectorAll('#principles textarea').forEach(t => { INTERP.principles[t.dataset.p][t.dataset.k] = t.value; });
  await api('/api/interpretation', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(INTERP) });
  $('interp-msg').textContent = 'saved ' + new Date().toLocaleTimeString();
};
loadInterp();

// ---------------------------------------------------------------- generation
let FAMILIES = null, CURRENT = null;
async function loadFamilies() {
  FAMILIES = await api('/api/families');
  $('family').innerHTML = Object.keys(FAMILIES.families).map(f => `<option>${f}</option>`).join('');
  $('family').onchange = buildParams; buildParams();
}
function buildParams() {
  const f = $('family').value; const p = FAMILIES.families[f]; $('family-principles').textContent = 'principles: ' + (FAMILIES.principles[f] || []).join('; ');
  $('params').innerHTML = Object.entries(p).map(([k, v]) => `<label>${k}</label><input data-k="${k}" value='${v === null ? '' : (typeof v === 'object' ? JSON.stringify(v) : v)}'>`).join('');
}
$('gen-btn').onclick = async () => {
  const params = {}; document.querySelectorAll('#params input').forEach(i => { let v = i.value.trim(); if (v === '') return; try { v = JSON.parse(v); } catch (e) { } params[i.dataset.k] = v; });
  $('gen-msg').textContent = 'generating…';
  try {
    CURRENT = await api('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ family: $('family').value, params }) });
    $('gen-msg').textContent = `N = ${CURRENT.n_atoms} atoms, porosity ${fmt(CURRENT.porosity)}, cell ${fmt(CURRENT.cell[0][0], 1)} × ${fmt(CURRENT.cell[1][1], 1)} Å`;
    $('e-atoms').textContent = CURRENT.n_atoms;
    drawPreview(CURRENT); kv($('precheck'), CURRENT.precheck); kv($('descriptors'), CURRENT.descriptors);
    $('sim-struct').innerHTML = `<div>structure</div><div>${CURRENT.design.family} (${CURRENT.n_atoms} atoms)</div>`;
  } catch (e) { $('gen-msg').textContent = 'error: ' + e.message; }
};
function kv(el, obj) { el.innerHTML = Object.entries(obj || {}).map(([k, v]) => `<div>${k}</div><div>${typeof v === 'object' ? JSON.stringify(v) : fmt(v)}</div>`).join(''); }
function drawPreview(st) {
  const c = $('preview'), ctx = c.getContext('2d'); const Lx = st.cell[0][0], Ly = st.cell[1][1]; const sc = Math.min(c.width / Lx, c.height / Ly) * 0.96;
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height); const ox = (c.width - Lx * sc) / 2, oy = (c.height - Ly * sc) / 2;
  const X = x => ox + x * sc, Y = y => c.height - oy - y * sc; const P = st.positions;
  ctx.strokeStyle = '#888'; ctx.lineWidth = 1; ctx.beginPath();
  st.bonds.forEach(([i, j]) => { const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1]; if (Math.abs(dx) > Lx / 2 || Math.abs(dy) > Ly / 2) return; ctx.moveTo(X(P[i][0]), Y(P[i][1])); ctx.lineTo(X(P[j][0]), Y(P[j][1])); }); ctx.stroke();
  ctx.fillStyle = '#111'; P.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), Math.max(1.2, 0.45 * sc), 0, 6.283); ctx.fill(); });
  ctx.strokeStyle = '#4cc9f0'; ctx.strokeRect(X(0), Y(Ly), Lx * sc, Ly * sc);
}
$('to-sim').onclick = () => document.querySelector('nav button[data-p=simulate]').click();
loadFamilies();

// ---------------------------------------------------------------- simulation
$('sim-btn').onclick = async () => {
  if (!CURRENT) { $('sim-msg').textContent = 'generate a structure first'; return; }
  const [dev, dt] = $('s-dev').value.split('|');
  try {
    const r = await api('/api/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ structure_id: CURRENT.structure_id, name: $('s-name').value, d_strain: +$('s-ds').value, max_strain: +$('s-max').value, fmax: +$('s-fmax').value, transverse_relax: $('s-tr').value === 'true', device: dev, dtype: dt }) });
    $('sim-msg').textContent = 'launched job ' + r.job_id; refreshJobs();
  } catch (e) { $('sim-msg').textContent = 'error: ' + e.message; }
};
async function refreshJobs() {
  const j = await api('/api/jobs');
  $('jobs').innerHTML = Object.entries(j).map(([id, v]) => `<div class="card"><b>${v.name}</b> (${v.n_atoms} atoms) – ${v.status} ${v.run_id ? `→ <a href="#" class="runlink" data-run="${v.run_id}">${v.run_id}</a>` : ''}<pre>${(v.log_tail || []).join('\n')}</pre></div>`).join('') || '<div class="muted">no jobs</div>';
  document.querySelectorAll('.runlink').forEach(a => a.onclick = (ev) => { ev.preventDefault(); loadRuns().then(() => { document.querySelector('nav button[data-p=fracture]').click(); $('f-run').value = a.dataset.run; loadFracture(); }); });
}
setInterval(refreshJobs, 5000); refreshJobs();

// ---------------------------------------------------------------- physics / validation / performance
fetch('/files/analysis/physics_description.json').then(r => r.json()).then(d => { $('physics').innerHTML = d.sections.map(s => `<div class="card"><h3>${s.title}</h3>${s.html}</div>`).join(''); }).catch(() => { $('physics').innerHTML = '<div class="muted">description file missing</div>'; });
async function loadValidation() {
  const t = await api('/api/validation');
  $('val-table').innerHTML = '<tr><th>test</th><th>name</th><th>expected</th><th>observed</th><th>error</th><th>tolerance</th><th>status</th><th>note</th></tr>' + t.map(r => `<tr><td>${r.test}</td><td>${r.name}</td><td>${r.expected}</td><td>${r.observed}</td><td>${typeof r.error === 'number' ? r.error.toExponential(2) : r.error}</td><td>${typeof r.tolerance === 'number' ? r.tolerance.toExponential(1) : r.tolerance}</td><td class="${r.status}">${r.status}</td><td class="muted">${r.note || ''}</td></tr>`).join('');
  const f = await api('/api/figures'); $('val-figs').innerHTML = f.figures.filter(x => x.file.includes('validation')).map(x => `<div><img src="/files/${x.file}"><div class="muted">${x.name}: ${x.caption}</div></div>`).join('');
  const p = await api('/api/performance'); $('perf').innerHTML = p.html ? p.html : `<pre>${JSON.stringify(p, null, 1)}</pre>` + f.figures.filter(x => x.file.includes('performance')).map(x => `<img src="/files/${x.file}" style="max-width:900px">`).join('');
}
loadValidation();

// ---------------------------------------------------------------- results
let RUNS = [], SEL = new Set(), CURVES = {};
const METRICS = ['modulus_2d_Nm', 'strength_Nm', 'strain_at_peak', 'first_damage_strain', 'failure_strain', 'work_to_failure_J_m2', 'specific_work_eV_per_atom', 'post_peak_load_retention', 'damage_localization', 'n_broken_total', 'n_damage_steps', 'poisson_ratio'];
const XAXES = ['porosity', 'n_atoms', 'hierarchy_levels', 'desc.pore_size_mean_A', 'desc.ligament_width_mean_A', 'desc.min_solid_fraction_across_x', 'desc.anisotropy_index', 'desc.tortuosity_x', 'desc.cyclomatic_per_atom', 'desc.pore_scale_ratio', 'desc.fraction_undercoordinated'].concat(METRICS.map(m => 'm.' + m));
function val(r, key) { if (key.startsWith('m.')) return r.metrics?.[key.slice(2)]; if (key.startsWith('desc.')) return r.descriptors?.[key.slice(5)]; return r[key]; }
async function loadRuns() {
  RUNS = (await api('/api/runs')).filter(r => r.status === 'completed');
  const opts = (k) => ['all'].concat([...new Set(RUNS.map(r => String(r[k])))].sort()).map(v => `<option>${v}</option>`).join('');
  ['r-camp', 'r-stage', 'r-fam'].forEach((id, i) => { const k = ['campaign', 'stage', 'family'][i]; const cur = $(id).value; $(id).innerHTML = opts(k); if (cur) $(id).value = cur; $(id).onchange = renderRuns; });
  $('px').innerHTML = XAXES.map(x => `<option>${x}</option>`).join(''); $('py').innerHTML = METRICS.map(x => `<option>m.${x}</option>`).join('');
  $('px').value = 'porosity'; $('py').value = 'm.strength_Nm'; ['px', 'py', 'pc'].forEach(id => $(id).onchange = renderPareto);
  $('f-run').innerHTML = RUNS.map(r => `<option value="${r.run_id}">${r.name} (${r.family}, ${r.stage})</option>`).join('');
  $('d-run').innerHTML = $('f-run').innerHTML; $('d-run').onchange = loadRecord; if (RUNS.length) loadRecord();
  renderRuns();
}
function filtered() { return RUNS.filter(r => ['r-camp', 'r-stage', 'r-fam'].every((id, i) => { const v = $(id).value; return v === 'all' || String(r[['campaign', 'stage', 'family'][i]]) === v; })); }
let sortKey = 'name', sortDir = 1;
function renderRuns() {
  const rows = filtered().sort((a, b) => { const x = val(a, sortKey), y = val(b, sortKey); return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
  $('r-count').textContent = rows.length + ' runs'; const cols = [['name', 'name'], ['family', 'family'], ['stage', 'stage'], ['n_atoms', 'N'], ['porosity', 'φ'], ['hierarchy_levels', 'H'], ['m.modulus_2d_Nm', 'Y2D (N/m)'], ['m.strength_Nm', 'σmax (N/m)'], ['m.strain_at_peak', 'εpeak'], ['m.failure_strain', 'εfail'], ['m.work_to_failure_J_m2', 'W (J/m²)'], ['m.damage_localization', 'L'], ['m.fracture_mode', 'mode'], ['viability', 'viability']];
  $('runs').innerHTML = '<tr>' + cols.map(([k, l]) => `<th data-k="${k}">${l}</th>`).join('') + '</tr>' + rows.map(r => `<tr class="clickable ${SEL.has(r.run_id) ? 'sel' : ''}" data-id="${r.run_id}">` + cols.map(([k]) => `<td>${fmt(val(r, k), k.includes('strain') ? 4 : 2)}</td>`).join('') + '</tr>').join('');
  document.querySelectorAll('#runs th').forEach(th => th.onclick = () => { if (sortKey === th.dataset.k) sortDir *= -1; else { sortKey = th.dataset.k; sortDir = 1; } renderRuns(); });
  document.querySelectorAll('#runs tr.clickable').forEach(tr => tr.onclick = async () => { const id = tr.dataset.id; if (SEL.has(id)) SEL.delete(id); else SEL.add(id); await renderCurves(); renderRuns(); });
  renderPareto();
}
async function renderCurves() {
  const series = []; let i = 0; const leg = [];
  for (const id of SEL) {
    if (!CURVES[id]) CURVES[id] = await api(`/api/runs/${id}/trajectory?fields=0&max_frames=5`);
    const c = CURVES[id]; const r = RUNS.find(x => x.run_id === id); const col = PAL[i % PAL.length]; i++;
    series.push({ x: c.eps_x, y: c.sigma_xx_Nm, color: col, markers: true, r: 2 }); leg.push(`<span style="color:${col}">■</span> ${r.name}`);
    const rec = RUNS.find(x => x.run_id === id); kv($('run-metrics'), Object.assign({ name: rec.name, family: rec.family, viability: rec.viability, termination: rec.termination }, rec.metrics));
  }
  plot($('ss'), series, { xlabel: 'engineering strain (-)', ylabel: '2D stress σxx (N/m)' }); $('ss-legend').innerHTML = leg.join(' &nbsp; ');
}
function renderPareto() {
  const rows = filtered(); const kx = $('px').value, ky = $('py').value, kc = $('pc').value; const groups = {};
  rows.forEach(r => { const g = String(r[kc]); (groups[g] = groups[g] || []).push(r); });
  const series = []; const leg = []; Object.entries(groups).forEach(([g, rs], i) => { const col = PAL[i % PAL.length]; series.push({ x: rs.map(r => val(r, kx)), y: rs.map(r => val(r, ky)), color: col, markers: true, line: false, r: 5, labels: rs.map(r => r.name.replace(/^S\d_/, '')) }); leg.push(`<span style="color:${col}">●</span> ${g}`); });
  // Pareto front (maximise y, minimise x if x is a cost like porosity? show non-dominated for max-max on (x,y) when x is a metric)
  const pts = rows.map(r => [val(r, kx), val(r, ky), r]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  const maxX = kx.startsWith('m.');
  const front = pts.filter(p => !pts.some(q => q !== p && (maxX ? q[0] >= p[0] : q[0] <= p[0]) && q[1] >= p[1] && ((maxX ? q[0] > p[0] : q[0] < p[0]) || q[1] > p[1]))).sort((a, b) => a[0] - b[0]);
  if (front.length > 1) series.push({ x: front.map(p => p[0]), y: front.map(p => p[1]), color: '#000', lw: 1, markers: false });
  plot($('pareto'), series, { xlabel: kx, ylabel: ky, ymin: undefined }); $('pareto-legend').innerHTML = leg.join(' &nbsp; ') + (front.length > 1 ? ' &nbsp; — Pareto front (' + (maxX ? 'max x' : 'min x') + ', max y): ' + front.map(p => p[2].name).join(', ') : '');
}
loadRuns(); setInterval(() => { if (document.querySelector('#p-results.active')) loadRuns(); }, 60000);

// ---------------------------------------------------------------- top structures
function renderTop(t) { $('top').innerHTML = (t.selected || []).map(s => `<div class="card"><h3>${s.title}</h3><div class="muted">${s.name} · ${s.run_id} · ${s.reason}</div><div class="row"><div class="col"><img src="/files/${s.panel}" style="width:100%;background:#fff;border-radius:6px"></div><div class="col" style="max-width:420px"><p>${s.interpretation}</p>${s.movie ? `<video controls width="380" src="/files/${s.movie}"></video>` : ''}<div class="kv">${Object.entries(s.metrics || {}).filter(([k]) => ['strength_Nm','failure_strain','work_to_failure_J_m2','damage_localization','fracture_mode'].includes(k)).map(([k, v]) => `<div>${k}</div><div>${fmt(v)}</div>`).join('')}</div></div></div></div>`).join('') || `<div class="muted">${t.note || ''}</div>`; }
api('/api/top').then(renderTop);

// ---------------------------------------------------------------- fracture viewer (three.js)
let renderer, scene, camera, controls, points, lines, TRAJ = null, frameIdx = 0, playing = false, damageTime = null;
function initViewer() {
  const el = $('viewer'); renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(window.devicePixelRatio); el.appendChild(renderer.domElement);
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x0a0d10);
  camera = new THREE.PerspectiveCamera(40, 1, 1, 5000); camera.position.set(50, 50, 200); camera.up.set(0, 1, 0);
  controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  resizeViewer(); animate();
}
function resizeViewer() { if (!renderer) return; const el = $('viewer'); const w = el.clientWidth || 800, h = el.clientHeight || 520; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resizeViewer);
function animate() { requestAnimationFrame(animate); if (controls) controls.update(); if (renderer) renderer.render(scene, camera); }
const cmap = (t) => { t = Math.min(1, Math.max(0, t)); const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]; const i = Math.min(3, Math.floor(t * 4)); const f = t * 4 - i; const c = stops[i].map((a, k) => a + (stops[i + 1][k] - a) * f); return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255); };
function colorValues(k) {
  const T = TRAJ; const mode = $('f-mode').value; const n = T.n_atoms; let v;
  if (mode === 'energy') v = T.energy_per_atom[k]; else if (mode === 'energy_rel') v = T.energy_per_atom[k].map((e, i) => e - T.energy_per_atom[0][i]);
  else if (mode === 'virial') v = T.virial_per_atom[k].map(w => w[0]); else if (mode === 'disp') v = T.nonaffine_displacement[k]; else if (mode === 'coord') v = T.coordination[k];
  else if (mode === 'damage') { v = damageTime.map(t => (t <= T.eps_x[T.frames[k]] ? t : NaN)); }
  return v;
}
function updateFrame(k) {
  if (!TRAJ) return; frameIdx = k; const fi = TRAJ.frames[k]; const P = TRAJ.positions[k]; const cell = TRAJ.cells[k]; const n = TRAJ.n_atoms;
  const pos = points.geometry.attributes.position.array; const col = points.geometry.attributes.color.array;
  const v = colorValues(k); const fin = v.filter(Number.isFinite); let lo = 0, hi = 1;
  if (fin.length) { const s = [...fin].sort((a, b) => a - b); lo = s[Math.floor(s.length * 0.02)]; hi = s[Math.floor(s.length * 0.98)]; if (hi <= lo) hi = lo + 1e-6; }
  if ($('f-mode').value === 'energy_rel') { lo = Math.min(lo, 0); }
  for (let i = 0; i < n; i++) { pos[3 * i] = P[i][0] - cell[0][0] / 2; pos[3 * i + 1] = P[i][1] - cell[1][1] / 2; pos[3 * i + 2] = 0; const c = Number.isFinite(v[i]) ? cmap((v[i] - lo) / (hi - lo)) : new THREE.Color(0.35, 0.35, 0.35); col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b; }
  points.geometry.attributes.position.needsUpdate = true; points.geometry.attributes.color.needsUpdate = true;
  const b = TRAJ.bonds[k]; const lp = new Float32Array(b.length * 6); let m = 0; const Lx = cell[0][0], Ly = cell[1][1];
  b.forEach(([i, j]) => { let dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1]; dx -= Lx * Math.round(dx / Lx); dy -= Ly * Math.round(dy / Ly); lp[m++] = P[i][0] - Lx / 2; lp[m++] = P[i][1] - Ly / 2; lp[m++] = 0; lp[m++] = P[i][0] + dx - Lx / 2; lp[m++] = P[i][1] + dy - Ly / 2; lp[m++] = 0; });
  lines.geometry.setAttribute('position', new THREE.BufferAttribute(lp, 3)); lines.geometry.computeBoundingSphere(); lines.visible = $('f-bonds').checked;
  $('f-slider').value = k; const eps = TRAJ.eps_x[fi], s = TRAJ.sigma_xx_Nm[fi];
  kv($('f-frame'), { frame: fi, strain: eps.toFixed(4), 'σxx (N/m)': s.toFixed(2), 'σyy (N/m)': TRAJ.sigma_yy_Nm[fi].toFixed(2), 'εy': TRAJ.eps_y[fi].toFixed(4), 'bonds': TRAJ.n_bonds[fi], 'broken (cum.)': TRAJ.n_broken_cum[fi], 'colour range': `${lo.toPrecision(3)} … ${hi.toPrecision(3)}` });
  plot($('f-ss'), [{ x: TRAJ.eps_x, y: TRAJ.sigma_xx_Nm, color: '#222', markers: true, r: 2 }], { xlabel: 'strain', ylabel: 'σxx (N/m)', marker: [eps, s], vlines: EVENTS });
}
let EVENTS = [];
async function loadFracture() {
  const id = $('f-run').value; if (!id) return; $('f-info').textContent = 'loading…';
  TRAJ = await api(`/api/runs/${id}/trajectory?fields=1&max_frames=300`);
  const r = RUNS.find(x => x.run_id === id); const n = TRAJ.n_atoms; $('e-atoms').textContent = n;
  damageTime = new Array(n).fill(Infinity); (TRAJ.damage_events || []).forEach(e => { damageTime[e[1]] = Math.min(damageTime[e[1]], e[0]); damageTime[e[2]] = Math.min(damageTime[e[2]], e[0]); });
  if (points) { scene.remove(points); scene.remove(lines); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  points = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, vertexColors: true })); scene.add(points);
  lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x8899aa, transparent: true, opacity: 0.6 })); scene.add(lines);
  const m = r.metrics || {}; const ip = TRAJ.sigma_xx_Nm.indexOf(Math.max(...TRAJ.sigma_xx_Nm));
  EVENTS = [{ x: m.first_damage_strain, label: 'first damage', color: '#e76f51' }, { x: TRAJ.eps_x[ip], label: 'peak', color: '#2a9d8f' }, { x: m.failure_strain, label: 'failure', color: '#264653' }].filter(e => Number.isFinite(e.x));
  $('f-events').innerHTML = EVENTS.map(e => `${e.label}: ε = ${e.x.toFixed(4)}`).join('<br>') + `<br>mode: ${m.fracture_mode || ''}<br>localisation L = ${fmt(m.damage_localization)}`;
  $('f-slider').max = TRAJ.frames.length - 1; setTop(); updateFrame(0); $('f-info').textContent = `${r.name}: ${TRAJ.frames.length} frames shown of ${TRAJ.eps_x.length}`;
}
function setTop() { const c = TRAJ ? TRAJ.cells[0] : [[100], [0, 100]]; const L = Math.max(c[0][0], c[1][1]); camera.position.set(0, 0, L * 1.35); controls.target.set(0, 0, 0); camera.up.set(0, 1, 0); controls.update(); }
$('f-top').onclick = setTop; $('f-persp').onclick = () => { const c = TRAJ ? TRAJ.cells[0] : [[100], [0, 100]]; const L = Math.max(c[0][0], c[1][1]); camera.position.set(-L * 0.6, -L * 0.9, L * 0.9); controls.update(); };
$('f-slider').oninput = (e) => updateFrame(+e.target.value); $('f-mode').onchange = () => updateFrame(frameIdx); $('f-bonds').onchange = () => updateFrame(frameIdx); $('f-run').onchange = loadFracture;
$('f-play').onclick = () => { playing = !playing; $('f-play').textContent = playing ? '❚❚ pause' : '▶ play'; if (playing) step(); };
function step() { if (!playing || !TRAJ) return; updateFrame((frameIdx + 1) % TRAJ.frames.length); setTimeout(step, 150); }
initViewer();
document.querySelector('nav button[data-p=fracture]').addEventListener('click', () => { if (!TRAJ && $('f-run').value) loadFracture(); });

// ---------------------------------------------------------------- database
async function loadRecord() {
  const id = $('d-run').value; if (!id) return; const rec = await api('/api/runs/' + id); const slim = Object.assign({}, rec); delete slim.broken_bond_events; $('d-record').textContent = JSON.stringify(slim, null, 1);
  const fm = ['extxyz', 'xyz', 'traj', 'lammps-data', 'cif', 'vasp']; $('d-exports').innerHTML = ['initial', 'relaxed', 'final'].map(w => `<div><b>${w}</b>: ` + fm.map(f => `<a href="/api/runs/${id}/export/${w}/${f}">${f}</a>`).join(' · ') + '</div>').join('') + `<div><b>trajectory</b>: <a href="/api/runs/${id}/export/final/trajectory">extxyz frames</a> · <a href="/api/runs/${id}/export/final/npz">npz (raw)</a> · <a href="/api/runs/${id}/export/final/csv">stress-strain csv</a> · <a href="/api/runs/${id}/export/final/json">record json</a></div>`;
}
api('/api/hypotheses').then(h => { $('d-hyp').textContent = JSON.stringify(h, null, 1); });
