'use strict';
/* Qwen3-TTS Console — front end. No framework: state lives in S, render* functions update the DOM.
   Every visible string goes through t() (see i18n.js). */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pad2 = n => String(n).padStart(2, '0');
const errText = e => (e && e.code && t('err.' + e.code) !== 'err.' + e.code) ? t('err.' + e.code) : (e?.message || String(e));

async function api(method, path, body, raw = false) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (raw) opt.body = body;
    else { opt.body = JSON.stringify(body); opt.headers['Content-Type'] = 'application/json'; }
  }
  const r = await fetch('/api/' + path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.message || r.status); e.code = data.error; throw e; }
  return data;
}

const S = {
  config: { defaults: {}, features: {}, idle_unload_minutes: 10 },
  voices: [],
  project: null,
  draft: null,          // settings used while no script is open; become the next script's settings
  status: null,
  rev: -1,
  sel: null,            // selected paragraph id
  playing: null,        // { paraId, mode: 'one' | 'all' }
  textTimer: null,
  settingsTimer: null,
  seenErrors: new Set(),
  lib: { tab: 'system', lang: '', gender: '', q: '' },
};
const settings = () => S.project ? S.project.settings : S.draft;

/* ───────── LCD ───────── */
const lcd = {
  msgTimer: null,
  flash(msg, err = false, ms = 4500) {
    const el = $('#lcd-msg');
    el.textContent = msg; el.classList.toggle('err', err);
    clearTimeout(this.msgTimer);
    this.msgTimer = setTimeout(() => { this.msgTimer = null; renderLcd(); }, ms);
  },
};
const fmtClock = sec => {
  sec = Math.max(0, sec || 0);
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${pad2(m)}:${s.toFixed(1).padStart(4, '0')}`;
};
const chosenTake = para => para.takes.find(x => x.id === para.chosen) || null;
const paraIndex = id => S.project ? S.project.paragraphs.findIndex(p => p.id === id) : -1;
const voiceById = id => S.voices.find(v => v.id === id);
const voiceLabel = id => voiceById(id)?.name.replace(/_/g, ' ') || '—';
function totalLength() {
  if (!S.project) return 0;
  const st = S.project.settings, speed = st.speed || 1;
  const durs = S.project.paragraphs.map(chosenTake).filter(Boolean).map(x => x.duration / speed);
  return durs.reduce((a, b) => a + b, 0) + Math.max(0, durs.length - 1) * (st.gap || 0);
}
function jobsByPara() {
  const m = new Map();
  for (const j of S.status?.jobs || []) {
    if (!S.project || j.pid !== S.project.id) continue;
    if (!m.has(j.para_id) || j.state === 'running') m.set(j.para_id, j);
  }
  return m;
}

function renderLcd() {
  const p = S.project, st = S.status;
  const title = $('#lcd-title');
  if (document.activeElement !== title) title.textContent = p ? p.title : t('lcd.noScript');
  title.contentEditable = p ? 'true' : 'false';
  let mode = p ? 'ready' : 'standby', blink = false, msg = '', prog = 0;
  const cur = st?.current;
  if (st?.loading) { mode = 'loading'; blink = true; msg = t('msg.loading.' + st.loading); }
  if (cur) {
    const i = p && cur.pid === p.id ? paraIndex(cur.para_id) : -1;
    mode = 'rec'; blink = true; prog = cur.progress || 0;
    const pct = Math.round(prog * 100), rest = (st.jobs || []).length - 1;
    msg = (i >= 0 ? t('msg.recording', { n: pad2(i + 1), p: pct }) : t('msg.recOther', { p: pct }))
      + (rest > 0 ? t('msg.queueRest', { k: rest }) : '');
  } else if (S.playing && p) {
    mode = S.playing.mode === 'all' ? 'playall' : 'play';
    msg = t('msg.playing', { n: pad2(paraIndex(S.playing.paraId) + 1) });
  } else if (p) {
    const n = p.paragraphs.length, done = p.paragraphs.filter(x => x.chosen).length;
    msg = done === n && n ? t('msg.allDone', { n })
      : st?.idle_unload_in != null ? t('msg.idle', { m: Math.ceil(st.idle_unload_in / 60) })
      : t('msg.pending', { n: n - done });
  } else msg = t('msg.open');
  if (!st && S.rev === -2) mode = 'offline';
  const modeEl = $('#lcd-mode');
  modeEl.textContent = t('mode.' + mode); modeEl.classList.toggle('blink', blink);
  if (!lcd.msgTimer) { const m = $('#lcd-msg'); m.textContent = msg; m.classList.remove('err'); }
  $('#lcd-progress').style.width = (prog * 100).toFixed(1) + '%';
  if (!S.playing) $('#lcd-counter .val').textContent = fmtClock(totalLength());
  const i = paraIndex(S.sel);
  $('#lcd-trk').textContent = i >= 0 ? pad2(i + 1) : '--';
  $('#lcd-done').textContent = p ? `${p.paragraphs.filter(x => x.chosen).length}/${p.paragraphs.length}` : '0/0';
  $('#lcd-voice').textContent = voiceLabel(settings()?.voice);
  const led = (id, kind) => {
    const el = $(id);
    el.classList.toggle('on', !!st?.models?.[kind]);
    el.classList.toggle('loading', st?.loading === kind);
  };
  led('#led-preset', 'preset'); led('#led-clone', 'clone');
  $('#led-rec').classList.toggle('on', !!st?.busy);
}

/* ───────── knobs ───────── */
const KNOBS = {
  temperature: { group: 'gen', name: 'k.temp', sub: 'k.tempSub', min: .3, max: 1.3, step: .05, def: .9, fmt: v => v.toFixed(2) },
  speed: { group: 'post', name: 'k.speed', sub: 'k.speedSub', min: .7, max: 1.6, step: .01, def: 1, fmt: v => '×' + v.toFixed(2) },
  gate: { group: 'post', name: 'k.gate', sub: 'k.gateSub', min: -61, max: -25, step: 1, def: -40, off: -61, fmt: v => v <= -61 ? t('k.off') : `${v} dB` },
  keep: { group: 'post', name: 'k.keep', sub: 'k.keepSub', min: .02, max: .6, step: .01, def: .05, fmt: v => t('k.sec', { v: v.toFixed(2) }) },
  gap: { group: 'post', name: 'k.gap', sub: 'k.gapSub', min: 0, max: 2, step: .05, def: .3, fmt: v => t('k.sec', { v: v.toFixed(2) }) },
};
const knobEls = {};

function buildKnobs() {
  for (const [key, k] of Object.entries(KNOBS)) {
    const wrap = document.createElement('div');
    wrap.className = 'knob-wrap';
    wrap.innerHTML = `<div class="knob" tabindex="0" role="slider" aria-valuemin="${k.min}" aria-valuemax="${k.max}">
        <div class="scale"></div><div class="arc"></div><div class="cap"></div></div>
      <div class="knob-val"></div><div class="knob-name" data-i18n="${k.name}"></div><div class="knob-sub" data-i18n="${k.sub}"></div>`;
    $(k.group === 'gen' ? '#knobs-gen' : '#knobs-post').appendChild(wrap);
    const knob = $('.knob', wrap);
    knobEls[key] = { knob, val: $('.knob-val', wrap), value: k.def };
    wireKnob(key, knob);
  }
}
function knobValue(key) {
  const st = settings(), k = KNOBS[key];
  if (key === 'gate') return st.gate == null ? k.off : st.gate;
  return st[key] ?? k.def;
}
function paintKnob(key, v) {
  const k = KNOBS[key], e = knobEls[key];
  const f = (v - k.min) / (k.max - k.min);
  e.value = v;
  e.knob.style.setProperty('--rot', (-135 + f * 270) + 'deg');
  e.knob.style.setProperty('--deg', (f * 270) + 'deg');
  e.knob.classList.toggle('off', k.off !== undefined && v <= k.off);
  e.knob.setAttribute('aria-valuenow', v);
  e.knob.setAttribute('aria-label', t(k.name));
  e.val.textContent = k.fmt(v);
}
function renderKnobs() { for (const key in KNOBS) paintKnob(key, knobValue(key)); }
function snap(key, v) {
  const k = KNOBS[key];
  v = Math.min(k.max, Math.max(k.min, v));
  return +(Math.round((v - k.min) / k.step) * k.step + k.min).toFixed(4);
}
function wireKnob(key, knob) {
  const k = KNOBS[key];
  let drag = null;
  const set = (v, commit) => { paintKnob(key, snap(key, v)); if (commit) commitKnob(key); };
  // Drag up / right to increase. Shift = fine. Pointer capture keeps the drag alive outside the knob.
  knob.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    e.preventDefault();
    knob.setPointerCapture(e.pointerId);
    drag = { id: e.pointerId, y: e.clientY, x: e.clientX, v: knobEls[key].value };
    knob.classList.add('dragging');
    knob.focus({ preventScroll: true });
  });
  knob.addEventListener('pointermove', e => {
    if (!drag || e.pointerId !== drag.id) return;
    const d = (drag.y - e.clientY) + (e.clientX - drag.x) * .5;
    set(drag.v + d / (e.shiftKey ? 600 : 160) * (k.max - k.min), false);
  });
  const end = e => {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null; knob.classList.remove('dragging'); commitKnob(key);
  };
  knob.addEventListener('pointerup', end);
  knob.addEventListener('pointercancel', end);
  knob.addEventListener('lostpointercapture', end);
  knob.addEventListener('dblclick', () => set(k.def, true));
  knob.addEventListener('wheel', e => {
    e.preventDefault();
    set(knobEls[key].value + (e.deltaY < 0 ? 1 : -1) * k.step, false);
    clearTimeout(knob._wt); knob._wt = setTimeout(() => commitKnob(key), 300);
  }, { passive: false });
  knob.addEventListener('keydown', e => {
    const dir = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (!dir) return;
    e.preventDefault(); e.stopPropagation();
    set(knobEls[key].value + dir * k.step * (e.shiftKey ? 5 : 1), true);
  });
}
function commitKnob(key) {
  const k = KNOBS[key], v = knobEls[key].value;
  const val = key === 'gate' && v <= k.off ? null : v;
  if (settings()[key] === val) return;
  setSetting(key, val);
  if (k.group === 'post') { renderLcd(); renderTracks(); if (S.playing) reloadPlaying(); }
}

/* ───────── settings ───────── */
function setSetting(key, val) {
  settings()[key] = val;
  if (!S.project) { localStorage.setItem('tts.draft', JSON.stringify(S.draft)); return; }
  clearTimeout(S.settingsTimer);
  S.settingsTimer = setTimeout(saveSettings, 300);
}
async function saveSettings() {
  clearTimeout(S.settingsTimer); S.settingsTimer = null;
  if (!S.project) return;
  try { await api('PUT', 'projects/' + S.project.id, { settings: S.project.settings }); }
  catch (e) { lcd.flash(t('msg.saveFailed', { err: errText(e) }), true); }
}

function buildMoods() {
  const box = $('#mood-chips');
  box.innerHTML = MOODS[LANG].map(([label, v]) => `<button type="button" data-v="${esc(v)}" title="${esc(v)}">${esc(label)}</button>`).join('');
}
$('#mood-chips').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b || isCustom()) return;
  const v = settings().instruct === b.dataset.v ? '' : b.dataset.v;
  $('#mood').value = v; setSetting('instruct', v); renderMood();
});
$('#mood').addEventListener('input', e => { setSetting('instruct', e.target.value); renderMood(); });
const isCustom = () => settings().voice.startsWith('custom:');
function renderMood() {
  const st = settings(), custom = isCustom();
  const input = $('#mood');
  if (document.activeElement !== input) input.value = st.instruct || '';
  input.disabled = custom;
  $('#dymo').classList.toggle('disabled', custom);
  $('#mood-note').textContent = t(custom ? 'mood.cloneNote' : 'mood.note');
  for (const b of $$('#mood-chips button')) { b.classList.toggle('on', st.instruct === b.dataset.v); b.disabled = custom; }
  const btns = $$('#lang button');
  btns.forEach((b, i) => { const on = b.dataset.v === (st.lang || 'auto'); b.classList.toggle('on', on); if (on) $('#lang').style.setProperty('--i', i); });
}
$('#lang').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) { setSetting('lang', b.dataset.v); renderMood(); }
});

/* ───────── voice library ───────── */
async function loadVoices() { S.voices = await api('GET', 'voices'); renderLibrary(); }

function voiceTags(v) {
  const tags = [];
  if (v.gender) tags.push(t('gs.' + v.gender));
  tags.push(v.dialect ? t('dialect.' + v.dialect) : v.langs.map(l => t('vs.' + l)).join(LANG === 'zh' ? '' : '/'));
  return tags.join(' · ');
}
function voiceDesc(v) {
  if (v.kind === 'system') return t('desc.' + v.name);
  return v.source === 'take' ? t('src.take', { title: v.source_title }) : t('src.upload');
}
function matches(v) {
  const f = S.lib;
  if (v.kind !== f.tab) return false;
  if (f.gender && v.gender !== f.gender) return false;
  if (f.lang === 'dialect' ? !v.dialect : f.lang && !v.langs.includes(f.lang)) return false;
  if (f.q) {
    const hay = `${v.name} ${voiceDesc(v)} ${voiceTags(v)} ${v.ref_text || ''}`.toLowerCase();
    if (!hay.includes(f.q.toLowerCase())) return false;
  }
  return true;
}
function renderLibrary() {
  const cur = settings()?.voice;
  const spinning = new Set();
  if (S.status?.current) spinning.add(S.status.current.voice);
  if (S.playing && S.project) {
    const para = S.project.paragraphs.find(p => p.id === S.playing.paraId);
    const tk = para && chosenTake(para);
    if (tk) spinning.add(tk.voice);
  }
  $('#count-system').textContent = S.voices.filter(v => v.kind === 'system').length;
  $('#count-custom').textContent = S.voices.filter(v => v.kind === 'custom').length;
  for (const b of $$('.lib-tabs button')) {
    b.classList.toggle('on', b.dataset.tab === S.lib.tab);
    b.classList.toggle('has-sel', !!cur && cur.startsWith(b.dataset.tab + ':'));
  }
  for (const b of $$('#f-lang button')) b.classList.toggle('on', b.dataset.v === S.lib.lang);
  for (const b of $$('#f-gender button')) b.classList.toggle('on', b.dataset.v === S.lib.gender);
  const list = S.voices.filter(matches);
  let html = list.map(v => `
    <button class="cart ${v.kind}${v.id === cur ? ' sel' : ''}${spinning.has(v.id) ? ' spin' : ''}" data-id="${esc(v.id)}"
        style="--c:${esc(v.color)}" title="${esc(v.kind === 'custom' ? v.ref_text : voiceDesc(v))}">
      <div class="sticker"><div class="nm"><span>${esc(v.name.replace(/_/g, ' '))}</span><small>${esc(voiceTags(v))}</small></div><div class="ds">${esc(voiceDesc(v))}</div></div>
      <div class="window"><i class="reel"></i><i class="reel"></i></div>
      ${v.kind === 'custom' ? `<div class="tools"><span data-act="hear" title="${esc(t('cart.hear'))}">▶</span><span data-act="edit" title="${esc(t('cart.edit'))}">✎</span></div>` : ''}
    </button>`).join('');
  if (S.lib.tab === 'custom') html += `<button class="cart add" id="cart-add"><div><b>＋</b>${esc(t('cart.add'))}</div></button>`;
  const shelf = $('#shelf');
  shelf.innerHTML = html;
  if (!list.length) {
    const p = document.createElement('p');
    p.className = 'shelf-empty';
    p.textContent = S.lib.tab === 'custom' && !S.voices.some(v => v.kind === 'custom') ? t('lib.emptyCustom') : t('lib.noMatch');
    shelf.prepend(p);
  }
}
$('.lib-tabs').addEventListener('click', e => {
  const b = e.target.closest('[data-tab]');
  if (b) { S.lib.tab = b.dataset.tab; renderLibrary(); }
});
for (const id of ['#f-lang', '#f-gender']) {
  $(id).addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    S.lib[id === '#f-lang' ? 'lang' : 'gender'] = b.dataset.v;
    renderLibrary();
  });
}
$('#f-search').addEventListener('input', e => { S.lib.q = e.target.value.trim(); renderLibrary(); });
$('#shelf').addEventListener('click', e => {
  const cart = e.target.closest('.cart');
  if (!cart) return;
  if (cart.id === 'cart-add') return openVoiceDialog({ mode: 'new' });
  const id = cart.dataset.id, act = e.target.closest('[data-act]')?.dataset.act;
  if (act === 'hear') { e.stopPropagation(); return playUrl('/media/voice/' + id.split(':')[1]); }
  if (act === 'edit') { e.stopPropagation(); return openVoiceDialog({ mode: 'edit', voice: voiceById(id) }); }
  setSetting('voice', id);
  renderLibrary(); renderMood(); renderLcd();
  lcd.flash(t('msg.voiceSet', { name: voiceLabel(id) }));
});
function showVoiceTab() {
  const v = settings()?.voice;
  if (v) S.lib.tab = v.startsWith('custom:') ? 'custom' : 'system';
}

/* ───────── paragraphs ───────── */
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function trackTemplate() {
  const el = document.createElement('div');
  el.className = 'trk';
  el.innerHTML = `
    <div class="trk-no"><b></b><i></i></div>
    <div class="trk-main">
      <textarea rows="1" spellcheck="false" data-i18n-ph="trk.ph"></textarea>
      <div class="wave"><canvas></canvas><div class="prog"></div><div class="scan"></div><div class="empty-wave" data-i18n="trk.noTake"></div></div>
      <div class="trk-meta"></div>
    </div>
    <div class="trk-side">
      <div class="takes"></div>
      <div class="trk-btns">
        <button class="ibtn rec" data-act="rec" data-i18n-title="trk.recTip"><svg viewBox="0 0 24 24"><path d="M12 3.5 13.9 9l5.6 1.1-4.3 3.7 1.1 5.7L12 16.7l-4.3 2.8 1.1-5.7-4.3-3.7L10.1 9z"/></svg><span data-i18n="trk.rec"></span></button>
        <button class="ibtn" data-act="play" data-i18n-title="trk.playTip"><svg viewBox="0 0 24 24"><path d="M7 4.5v15l12.5-7.5z"/></svg></button>
        <button class="ibtn" data-act="lock" data-i18n-title="trk.lockTip"><svg viewBox="0 0 24 24"><path d="M7 10V7a5 5 0 0 1 10 0v3h1.5v11h-13V10zm2.5 0h5V7a2.5 2.5 0 0 0-5 0z"/></svg><span data-i18n="trk.lock"></span></button>
        <button class="ibtn danger" data-act="deltake" data-i18n-title="trk.delTakeTip"><svg viewBox="0 0 24 24"><path d="M9 3h6l1 2h4v2H4V5h4zm-3 6h12l-1 12H7z"/></svg></button>
        <button class="ibtn danger" data-act="delpara" data-i18n-title="trk.delParaTip"><svg viewBox="0 0 24 24"><path d="m6.4 5 5.6 5.6L17.6 5 19 6.4 13.4 12l5.6 5.6-1.4 1.4-5.6-5.6L6.4 19 5 17.6l5.6-5.6L5 6.4z"/></svg></button>
      </div>
    </div>`;
  applyI18n(el);
  return el;
}

function renderTracks() {
  const box = $('#tracks');
  const p = S.project;
  $('#empty').hidden = !!p;
  box.hidden = !p;
  $('.tape-actions').style.visibility = p ? '' : 'hidden';
  if (!p) { box.innerHTML = ''; return; }
  const existing = new Map($$('.trk', box).map(el => [el.dataset.id, el]));
  const jobs = jobsByPara();
  p.paragraphs.forEach((para, i) => {
    let el = existing.get(para.id);
    if (!el) { el = trackTemplate(); el.dataset.id = para.id; }
    existing.delete(para.id);
    if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
    updateTrack(el, para, i, jobs.get(para.id));
  });
  for (const el of existing.values()) el.remove();
}

function updateTrack(el, para, i, job) {
  const take = chosenTake(para);
  el.classList.toggle('sel', para.id === S.sel);
  el.classList.toggle('has', !!take);
  el.classList.toggle('busy', job?.state === 'running');
  el.classList.toggle('queued', job?.state === 'queued');
  $('.trk-no b', el).textContent = pad2(i + 1);
  const ta = $('textarea', el);
  if (document.activeElement !== ta && ta.value !== para.text) ta.value = para.text;
  autosize(ta);
  const takes = $('.takes', el);
  const sig = para.takes.map(x => x.id).join() + '|' + para.chosen + '|' + LANG;
  if (takes.dataset.sig !== sig) {
    takes.dataset.sig = sig;
    takes.innerHTML = para.takes.map((x, n) => {
      const tip = t('trk.takeTip', { voice: x.voice_name.replace(/_/g, ' '), temp: x.temperature, mood: x.instruct ? ` · ${x.instruct}` : '', dur: x.duration });
      return `<button class="take${x.id === para.chosen ? ' on' : ''}" data-tid="${x.id}" title="${esc(tip)}">${n + 1}</button>`;
    }).join('');
  }
  for (const a of ['lock', 'deltake', 'play']) $(`[data-act=${a}]`, el).disabled = !take;
  const meta = $('.trk-meta', el);
  let html = '';
  if (job?.state === 'running') html = `<span class="tag warn">${esc(t('trk.running', { p: Math.round((job.progress || 0) * 100) }))}</span><span>${esc(voiceLabel(job.voice))}</span>`;
  else if (job?.state === 'queued') html = `<span class="tag">${esc(t('trk.queued'))}</span><span>${esc(voiceLabel(job.voice))}</span>`;
  else if (take) {
    const sp = S.project.settings.speed || 1;
    html = `<span class="tag">${esc(take.voice_name.replace(/_/g, ' '))}</span><span>${esc(t('k.sec', { v: (take.duration / sp).toFixed(1) }))}</span>`
      + `<span>${esc(t('k.temp'))} ${take.temperature}</span>`
      + (take.instruct ? `<span>${esc(take.instruct)}</span>` : '')
      + `<span>${esc(t(para.takes.length === 1 && LANG === 'en' ? 'trk.takes1' : 'trk.takes', { n: para.takes.length }))}</span>`
      + (take.text.trim() !== para.text.trim() ? `<span class="tag warn">${esc(t('trk.textChanged'))}</span>` : '');
  }
  meta.innerHTML = html;
  $('.wave .prog', el).style.width = job?.state === 'running' ? ((job.progress || 0) * 100) + '%' : '0';
  $('.wave .empty-wave', el).hidden = !!take || job?.state === 'running';
  drawWave(el, take, S.playing?.paraId === para.id ? playFrac() : 0);
}

function drawWave(el, take, frac) {
  const canvas = $('canvas', el);
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w) return;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (!take) return;
  const peaks = take.peaks, n = peaks.length, bw = w / n;
  const on = css('--orange'), off = css('--ink-3');
  for (let i = 0; i < n; i++) {
    const a = Math.max(.05, peaks[i]) * (h * .4), played = (i + .5) / n <= frac;
    g.fillStyle = played ? on : off;
    g.globalAlpha = played ? 1 : .7;
    g.fillRect(i * bw + bw * .18, h / 2 - a, Math.max(1, bw * .64), a * 2);
  }
  g.globalAlpha = 1;
  if (frac > 0) { g.fillStyle = on; g.fillRect(frac * w - 1, 0, 2, h); }
}
const autosize = ta => { ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; };

function selectTrack(id, scroll = false) {
  if (S.sel === id) return;
  S.sel = id;
  for (const el of $$('.trk')) el.classList.toggle('sel', el.dataset.id === id);
  if (scroll) $(`.trk[data-id="${id}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  renderLcd();
}

const tracks = $('#tracks');
tracks.addEventListener('focusin', e => { const x = e.target.closest('.trk'); if (x) selectTrack(x.dataset.id); });
tracks.addEventListener('pointerdown', e => { const x = e.target.closest('.trk'); if (x) selectTrack(x.dataset.id); });
tracks.addEventListener('input', e => {
  if (e.target.tagName !== 'TEXTAREA') return;
  const para = S.project.paragraphs.find(p => p.id === e.target.closest('.trk').dataset.id);
  para.text = e.target.value;
  autosize(e.target);
  clearTimeout(S.textTimer);
  S.textTimer = setTimeout(saveParagraphs, 700);
});
tracks.addEventListener('click', async e => {
  const trk = e.target.closest('.trk');
  if (!trk) return;
  const para = S.project.paragraphs.find(p => p.id === trk.dataset.id);
  const takeBtn = e.target.closest('.take');
  if (takeBtn) return choose(para, takeBtn.dataset.tid, true);
  const wave = e.target.closest('.wave');
  if (wave && chosenTake(para)) {
    const r = wave.getBoundingClientRect();
    return playPara(para.id, (e.clientX - r.left) / r.width, 'one');
  }
  const act = e.target.closest('[data-act]')?.dataset.act;
  const n = pad2(paraIndex(para.id) + 1);
  if (act === 'rec') rec([para.id]);
  if (act === 'play') togglePlay(para.id);
  if (act === 'lock') openVoiceDialog({ mode: 'lock', para });
  if (act === 'deltake') {
    const tk = chosenTake(para);
    if (!tk || !confirm(t('c.delTake', { n, k: para.takes.indexOf(tk) + 1 }))) return;
    if (S.playing?.paraId === para.id) stop();
    S.project = await api('DELETE', `projects/${S.project.id}/takes/${tk.id}`);
    renderAll();
  }
  if (act === 'delpara') {
    if (para.takes.length && !confirm(t('c.delPara', { n, k: para.takes.length }))) return;
    const i = paraIndex(para.id);
    S.project.paragraphs.splice(i, 1);
    if (S.sel === para.id) S.sel = S.project.paragraphs[Math.min(i, S.project.paragraphs.length - 1)]?.id || null;
    await saveParagraphs(); renderAll();
  }
});
tracks.addEventListener('keydown', e => {
  if (e.target.tagName !== 'TEXTAREA') return;
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); rec([e.target.closest('.trk').dataset.id]); }
  if (e.key === 'Escape') e.target.blur();
});

async function saveParagraphs() {
  clearTimeout(S.textTimer); S.textTimer = null;
  if (!S.project) return;
  try {
    const p = await api('PUT', 'projects/' + S.project.id, { paragraphs: S.project.paragraphs.map(x => ({ id: x.id, text: x.text })) });
    const local = new Map(S.project.paragraphs.map(x => [x.id, x.text]));   // text typed while the request was in flight wins
    for (const x of p.paragraphs) if (local.has(x.id)) x.text = local.get(x.id);
    S.project = p;
    renderTracks(); renderLcd();
  } catch (e) { lcd.flash(t('msg.saveFailed', { err: errText(e) }), true); }
}

async function choose(para, tid, play) {
  if (para.chosen !== tid) {
    para.chosen = tid;
    renderTracks();
    api('POST', `projects/${S.project.id}/choose`, { para_id: para.id, tid }).catch(e => lcd.flash(errText(e), true));
  }
  renderLcd();
  if (play) playPara(para.id, 0, 'one');
}

$('#btn-add').onclick = async () => {
  if (!S.project) return;
  S.project.paragraphs.push({ text: '', takes: [], chosen: null });
  await saveParagraphs();
  const last = S.project.paragraphs.at(-1);
  selectTrack(last.id);
  $(`.trk[data-id="${last.id}"] textarea`)?.focus();
};

/* ───────── recording (generation) ───────── */
async function rec(ids) {
  if (!S.project) return lcd.flash(t('msg.needScript'), true);
  ids = ids.filter(id => S.project.paragraphs.find(p => p.id === id)?.text.trim());
  if (!ids.length) return lcd.flash(t('msg.emptyPara'), true);
  await Promise.all([S.textTimer && saveParagraphs(), S.settingsTimer && saveSettings()]);
  try {
    const r = await api('POST', `projects/${S.project.id}/generate`, { para_ids: ids });
    const voice = voiceLabel(S.project.settings.voice);
    const cold = !S.status?.models?.[isCustom() ? 'clone' : 'preset'];
    lcd.flash((ids.length > 1 ? t('msg.queuedMany', { n: ids.length, voice }) : t('msg.recStart', { n: pad2(paraIndex(ids[0]) + 1), voice }))
      + (cold ? t('msg.firstLoad') : ''));
    S.status = S.status || {};
    S.status.jobs = [...(S.status.jobs || []), ...r.jobs];
    renderTracks();
    poll(true);
  } catch (e) { lcd.flash(errText(e), true); }
}
function recPending() {
  if (!S.project) return lcd.flash(t('msg.needScript'), true);
  const ids = S.project.paragraphs.filter(p => !p.chosen && p.text.trim()).map(p => p.id);
  const busy = jobsByPara();
  const todo = ids.filter(id => !busy.has(id));
  if (!todo.length) return lcd.flash(t(ids.length ? 'msg.allQueued' : 'msg.nothingPending'));
  rec(todo);
}
$('#btn-rec-all').onclick = recPending;

/* ───────── status polling ───────── */
let pollTimer = null;
async function poll(soon = false) {
  clearTimeout(pollTimer);
  if (soon) { pollTimer = setTimeout(poll, 400); return; }
  try {
    const st = await api('GET', 'status');
    S.status = st;
    for (const e of st.errors || []) {
      if (S.seenErrors.has(e.id)) continue;
      S.seenErrors.add(e.id);
      lcd.flash(t('msg.genFailed', { err: errText({ code: e.error }) }), true, 8000);
    }
    if (st.rev !== S.rev) {
      const first = S.rev < 0;
      S.rev = st.rev;
      if (!first && S.project) await reloadProject();
    }
    renderTracks(); renderLcd(); renderLibrary();
    vu.generating = !!st.busy;
    vu.kick();
  } catch {
    S.status = null; S.rev = -2; renderLcd();
  }
  pollTimer = setTimeout(poll, S.status?.busy || S.status?.loading ? 900 : 4000);
}
async function reloadProject() {
  const p = await api('GET', 'projects/' + S.project.id).catch(() => null);
  if (!p || p.id !== S.project?.id) return;
  if (S.textTimer) {
    const local = new Map(S.project.paragraphs.map(x => [x.id, x.text]));
    for (const x of p.paragraphs) if (local.has(x.id)) x.text = local.get(x.id);
  }
  const prevChosen = new Map(S.project.paragraphs.map(x => [x.id, x.chosen]));
  S.project = p;
  // a fresh take on the selected paragraph: play it right away
  const fresh = p.paragraphs.find(x => x.id === S.sel && x.chosen && x.chosen !== prevChosen.get(x.id));
  if (fresh && !S.playing) playPara(fresh.id, 0, 'one');
}

/* ───────── playback (Web Audio) ─────────
   An AudioContext opened on the first click stays running, so the output device is already awake when
   playback starts (Bluetooth / HDMI outputs otherwise swallow the first few hundred ms), and every clip
   is scheduled 60 ms ahead instead of starting at once. */
const audio = {
  ctx: null, analyser: null, src: null, buf: null, url: null, cache: new Map(),
  startAt: 0, offset: 0, playing: false, token: 0,
  async ensure() {
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: 'interactive' });
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state !== 'running') await this.ctx.resume().catch(() => {});
  },
  load(url) {
    if (!this.cache.has(url)) {
      const p = fetch(url).then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
        .then(b => this.ctx.decodeAudioData(b));
      p.catch(() => this.cache.delete(url));
      this.cache.set(url, p);
      if (this.cache.size > 40) this.cache.delete(this.cache.keys().next().value);
    }
    return this.cache.get(url);
  },
  async play(url, offset, onEnded) {
    await this.ensure();
    const token = ++this.token;
    const buf = await this.load(url).catch(() => null);
    if (!buf || token !== this.token) return false;          // failed, or a newer play() took over
    this.halt();
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser);
    offset = Math.min(Math.max(0, offset || 0), Math.max(0, buf.duration - 0.02));
    const when = this.ctx.currentTime + 0.06;
    src.start(when, offset);
    Object.assign(this, { src, buf, url, startAt: when - offset, offset, playing: true });
    src.onended = () => {
      if (this.src !== src) return;
      this.src = null; this.playing = false; this.offset = buf.duration;
      onEnded?.();
    };
    return true;
  },
  halt() {
    if (!this.src) return;
    this.src.onended = null;
    try { this.src.stop(); } catch { /* not started yet */ }
    this.src.disconnect();
    this.src = null;
  },
  pause() { if (this.playing) { this.offset = this.time(); this.playing = false; this.halt(); } },
  stop() { this.token++; this.playing = false; this.halt(); this.offset = 0; this.url = null; },
  time() { return this.playing ? Math.min(this.buf.duration, Math.max(0, this.ctx.currentTime - this.startAt)) : this.offset; },
  duration() { return this.buf ? this.buf.duration : 0; },
};
document.addEventListener('pointerdown', () => audio.ensure(), { once: true, capture: true });
document.addEventListener('keydown', () => audio.ensure(), { once: true, capture: true });

function takeUrl(para) {
  const st = S.project.settings;
  const q = new URLSearchParams({ speed: st.speed ?? 1, gate: st.gate == null ? 'off' : st.gate, keep: st.keep ?? .05 });
  return `/media/take/${S.project.id}/${chosenTake(para).id}.wav?${q}`;
}
const playFrac = () => S.playing && audio.duration() ? audio.time() / audio.duration() : 0;
async function playPara(id, frac = 0, mode = 'one') {
  const para = S.project?.paragraphs.find(p => p.id === id);
  if (!para || !chosenTake(para)) return lcd.flash(t('msg.noTake'), true);
  clearTimeout(S.nextTimer);
  S.playing = { paraId: id, mode };
  selectTrack(id, mode === 'all');
  renderLcd(); renderLibrary();
  await audio.ensure();
  const url = takeUrl(para);
  const buf = await audio.load(url).catch(() => null);
  if (S.playing?.paraId !== id) return;
  if (!buf || !await audio.play(url, frac * buf.duration, clipEnded)) return;
  vu.kick();
  if (mode === 'all') {                      // decode the next one while this one plays
    const next = nextChosen(id);
    if (next) audio.load(takeUrl(next)).catch(() => {});
  }
}
function nextChosen(id) {
  const ps = S.project.paragraphs;
  return ps.slice(paraIndex(id) + 1).find(chosenTake) || null;
}
function clipEnded() {
  if (S.playing?.mode !== 'all') return stop();
  const next = nextChosen(S.playing.paraId);
  if (!next) { stop(); lcd.flash(t('msg.done')); return; }
  S.nextTimer = setTimeout(() => playPara(next.id, 0, 'all'), (S.project.settings.gap || 0) * 1000);
}
async function playUrl(url) {
  stop();
  if (await audio.play(url, 0)) vu.kick();
}
function reloadPlaying() { if (S.playing) playPara(S.playing.paraId, playFrac(), S.playing.mode); }
function togglePlay(id) {
  id = id || S.sel;
  if (S.playing?.paraId === id && audio.url) {
    if (audio.playing) { audio.pause(); renderLcd(); }
    else audio.play(audio.url, audio.offset, clipEnded).then(ok => ok && vu.kick());
    return;
  }
  playPara(id, 0, 'one');
}
function playAll() {
  if (!S.project) return lcd.flash(t('msg.needScript'), true);
  const ps = S.project.paragraphs;
  const first = ps.slice(Math.max(0, paraIndex(S.sel))).find(chosenTake) || ps.find(chosenTake);
  if (!first) return lcd.flash(t('msg.noTakesAll'), true);
  playPara(first.id, 0, 'all');
}
function stop() {
  clearTimeout(S.nextTimer);
  audio.stop();
  const was = S.playing; S.playing = null;
  if (was) { renderTracks(); renderLcd(); renderLibrary(); }
}

const vu = {
  dots: [], cols: 9, rows: 6, raf: 0, generating: false, buf: null, level: 0,
  build() {
    const g = $('#grille');
    g.innerHTML = '<i></i>'.repeat(this.cols * this.rows);
    this.dots = $$('i', g).map((el, n) => {
      const c = n % this.cols, r = Math.floor(n / this.cols);
      const dx = (c - (this.cols - 1) / 2) / ((this.cols - 1) / 2), dy = (r - (this.rows - 1) / 2) / ((this.rows - 1) / 2);
      return { el, c, d: Math.hypot(dx, dy * .75) / 1.25 };
    });
  },
  kick() { if (!this.raf) this.raf = requestAnimationFrame(ts => this.frame(ts)); },
  frame(ts) {
    this.raf = 0;
    const playing = audio.playing;
    if (playing && audio.analyser) {
      this.buf = this.buf || new Float32Array(audio.analyser.fftSize);
      audio.analyser.getFloatTimeDomainData(this.buf);
      let s = 0; for (const x of this.buf) s += x * x;
      this.level = Math.max(Math.min(1, Math.sqrt(s / this.buf.length) * 4.5), this.level * .82);
    } else this.level = 0;
    const sweep = this.generating && !playing ? ((ts / 90) % (this.cols + 6)) - 3 : null;
    for (const d of this.dots) {
      let lit = false, dim = false;
      if (playing) { lit = d.d < this.level; dim = !lit && d.d < this.level + .12; }
      else if (sweep !== null) { const k = Math.abs(d.c - sweep); lit = k < .6; dim = !lit && k < 2; }
      d.el.classList.toggle('lit', lit); d.el.classList.toggle('dim', dim);
    }
    if (playing && S.playing && S.project) {
      const el = $(`.trk[data-id="${S.playing.paraId}"]`);
      const para = S.project.paragraphs.find(p => p.id === S.playing.paraId);
      if (el && para) drawWave(el, chosenTake(para), playFrac());
      const ps = S.project.paragraphs, i = paraIndex(S.playing.paraId), sp = S.project.settings.speed || 1;
      let before = 0, n = 0;
      for (const p of ps.slice(0, i)) { const tk = chosenTake(p); if (tk) { before += tk.duration / sp; n++; } }
      $('#lcd-counter .val').textContent = fmtClock(S.playing.mode === 'all' ? before + n * (S.project.settings.gap || 0) + audio.time() : audio.time());
    }
    if (playing || this.generating) this.kick();
  },
};

/* ───────── keyboard ───────── */
document.addEventListener('keydown', e => {
  const el = e.target;
  if (el.closest?.('input, textarea, select, [contenteditable=true], dialog') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (el.classList?.contains('knob') && e.key.startsWith('Arrow')) return;
  const ps = S.project?.paragraphs || [];
  const i = paraIndex(S.sel);
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const go = d => { const n = ps[Math.min(ps.length - 1, Math.max(0, i + d))]; if (n) selectTrack(n.id, true); };
  switch (key) {
    case ' ': e.preventDefault(); e.shiftKey ? playAll() : S.sel && togglePlay(); break;
    case 'g': S.sel && rec([S.sel]); break;
    case 'a': recPending(); break;
    case 'b': openExport(); break;
    case 'ArrowDown': case 'j': e.preventDefault(); go(1); break;
    case 'ArrowUp': case 'k': e.preventDefault(); go(-1); break;
    case 'Escape': stop(); break;
    case 'Enter': if (S.sel) { e.preventDefault(); $(`.trk[data-id="${S.sel}"] textarea`)?.focus(); } break;
    default:
      if (/^[1-9]$/.test(key) && S.sel) {
        const para = ps[i], tk = para?.takes[+key - 1];
        if (tk) choose(para, tk.id, true);
      }
  }
});
$('#t-rec').onclick = () => S.sel ? rec([S.sel]) : lcd.flash(t(S.project ? 'msg.pickPara' : 'msg.needScript'), true);
$('#t-play').onclick = () => S.sel ? togglePlay() : lcd.flash(t(S.project ? 'msg.pickPara' : 'msg.needScript'), true);
$('#t-all').onclick = playAll;
$('#t-stop').onclick = stop;
$('#t-bounce').onclick = () => openExport();

/* ───────── title ───────── */
const titleEl = $('#lcd-title');
titleEl.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); titleEl.blur(); } });
titleEl.addEventListener('blur', async () => {
  const v = titleEl.textContent.trim();
  if (!S.project || !v || v === S.project.title) { renderLcd(); return; }
  S.project.title = v;
  await api('PUT', 'projects/' + S.project.id, { title: v }).catch(e => lcd.flash(errText(e), true));
});

/* ───────── scripts ───────── */
async function openProject(pid) {
  stop();
  const p = await api('GET', 'projects/' + pid).catch(() => null);
  if (!p) return false;
  S.project = p;
  S.sel = p.paragraphs.find(x => !x.chosen)?.id || p.paragraphs[0]?.id || null;
  localStorage.setItem('tts.last', pid);
  history.replaceState(null, '', '#' + pid);
  $$('.trk').forEach(el => el.remove());
  showVoiceTab();
  renderAll();
  return true;
}
function closeProject() {
  // keep the closed script's settings as the draft, so the console stays as it was
  if (S.project) { S.draft = { ...S.project.settings }; localStorage.setItem('tts.draft', JSON.stringify(S.draft)); }
  S.project = null; S.sel = null; stop();
  localStorage.removeItem('tts.last');
  history.replaceState(null, '', location.pathname);
  renderAll();
}
function renderAll() { renderLcd(); renderLibrary(); renderKnobs(); renderMood(); renderTracks(); }

const splitParas = text => {
  text = text.replace(/\r\n/g, '\n').trim();
  let parts = text.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
  if (parts.length <= 1) parts = text.split('\n').map(s => s.trim()).filter(Boolean);
  return parts;
};
function openNew() {
  $('#new-title').value = ''; $('#new-text').value = ''; $('#new-count').textContent = '';
  $('#dlg-new').showModal(); $('#new-text').focus();
}
$('#btn-new').onclick = openNew;
$('#btn-new2').onclick = openNew;
$('#new-text').addEventListener('input', e => {
  const n = splitParas(e.target.value).length;
  $('#new-count').textContent = n ? t('new.count', { n }) : '';
});
$('#dlg-new').addEventListener('dragover', e => e.preventDefault());
$('#dlg-new').addEventListener('drop', async e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (!f) return;
  $('#new-text').value = await f.text();
  if (!$('#new-title').value) $('#new-title').value = f.name.replace(/\.[^.]+$/, '');
  $('#new-text').dispatchEvent(new Event('input'));
});
$('#new-ok').onclick = async e => {
  e.preventDefault();
  const text = $('#new-text').value;
  if (!text.trim()) return $('#new-text').focus();
  try {
    await saveSettings();
    const p = await api('POST', 'projects', { title: $('#new-title').value, text, settings: settings() });
    $('#dlg-new').close();
    await openProject(p.id);
    lcd.flash(t('msg.newScript', { n: p.paragraphs.length }), false, 7000);
  } catch (err) { $('#new-count').textContent = errText(err); }
};

$('#btn-scripts').onclick = async () => {
  const list = await api('GET', 'projects').catch(() => []);
  const fmt = new Intl.DateTimeFormat(LANG === 'zh' ? 'zh-CN' : 'en', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  $('#script-list').innerHTML = list.length ? list.map(p => `
    <div class="script-item${p.id === S.project?.id ? ' cur' : ''}" data-id="${p.id}" role="button" tabindex="0">
      <span class="spine"></span>
      <div><div class="st">${esc(p.title)}</div><div class="sm">${esc(t('scripts.meta', { done: p.chosen, n: p.paragraphs, date: fmt.format(p.updated * 1000) }))}</div></div>
      <button type="button" class="ibtn danger del" data-del="${p.id}" title="${esc(t('d.delete'))}"><svg viewBox="0 0 24 24"><path d="M9 3h6l1 2h4v2H4V5h4zm-3 6h12l-1 12H7z"/></svg></button>
    </div>`).join('') : `<p class="sub">${esc(t('scripts.empty'))}</p>`;
  $('#dlg-scripts').showModal();
};
$('#script-list').addEventListener('click', async e => {
  const del = e.target.closest('[data-del]');
  if (del) {
    const item = del.closest('.script-item');
    if (!confirm(t('c.delScript', { title: $('.st', item).textContent }))) return;
    await api('DELETE', 'projects/' + del.dataset.del).catch(err => lcd.flash(errText(err), true));
    item.remove();
    if (S.project?.id === del.dataset.del) closeProject();
    return;
  }
  const item = e.target.closest('.script-item');
  if (item) { $('#dlg-scripts').close(); openProject(item.dataset.id); }
});

/* ───────── voice dialog: new (upload) / lock (from a take) / edit ───────── */
const vd = { mode: 'new', upload: null, from: null, voice: null };
function openVoiceDialog({ mode, para, voice }) {
  Object.assign(vd, { mode, upload: null, from: null, voice: voice || null });
  const audio = $('#voice-audio');
  audio.pause(); audio.removeAttribute('src'); audio.hidden = true;
  $('#voice-status').textContent = '';
  $('#voice-drop').hidden = mode !== 'new';
  $('#voice-drop-text').textContent = t('v.drop');
  $('#voice-asr').hidden = mode !== 'new';
  $('#voice-asr').disabled = true;
  $('#voice-del').hidden = mode !== 'edit';
  $('#voice-ok').textContent = t(mode === 'edit' ? 'v.save' : 'v.create');
  $('#voice-sub').textContent = t({ new: 'v.sub', lock: 'v.lockSub', edit: 'v.editSub' }[mode]);
  if (mode === 'new') {
    $('#voice-title').textContent = t('v.newTitle');
    $('#voice-name').value = ''; $('#voice-text').value = '';
    $('#voice-gender').value = ''; $('#voice-lang').value = LANG === 'zh' ? 'zh' : 'en';
  } else if (mode === 'lock') {
    const tk = chosenTake(para), src = voiceById(tk.voice);
    vd.from = { pid: S.project.id, tid: tk.id };
    $('#voice-title').textContent = t('v.lockTitle', { n: pad2(paraIndex(para.id) + 1) });
    $('#voice-name').value = t('v.defaultName', { voice: tk.voice_name.replace(/_/g, ' '), title: S.project.title.slice(0, 8) }).slice(0, 24);
    $('#voice-text').value = tk.text;
    $('#voice-gender').value = src?.gender || '';
    $('#voice-lang').value = src?.langs?.[0] || 'zh';
    audio.src = takeUrl(para); audio.hidden = false;
  } else {
    $('#voice-title').textContent = t('v.editTitle');
    $('#voice-name').value = voice.name; $('#voice-text').value = voice.ref_text;
    $('#voice-gender').value = voice.gender || ''; $('#voice-lang').value = voice.langs[0] || 'zh';
    audio.src = '/media/voice/' + voice.id.split(':')[1]; audio.hidden = false;
  }
  $('#dlg-voice').showModal();
  checkVoice();
}
function checkVoice() {
  const hasRef = vd.mode !== 'new' || vd.upload;
  $('#voice-ok').disabled = !(hasRef && $('#voice-name').value.trim() && $('#voice-text').value.trim());
}
$('#voice-name').addEventListener('input', checkVoice);
$('#voice-text').addEventListener('input', checkVoice);
async function uploadFile(f) {
  $('#voice-drop-text').textContent = t('v.uploading', { name: f.name });
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: f });
    const d = await r.json();
    if (!r.ok) { const e = new Error(d.message); e.code = d.error; throw e; }
    vd.upload = d.upload_id;
    $('#voice-drop-text').textContent = t('v.uploaded', { name: f.name, d: d.duration.toFixed(1) })
      + (d.duration > 20 ? t('v.tooLong') : d.duration < 3 ? t('v.tooShort') : '');
    const a = $('#voice-audio'); a.src = '/media/upload/' + d.upload_id; a.hidden = false;
    if (!$('#voice-name').value) $('#voice-name').value = f.name.replace(/\.[^.]+$/, '').slice(0, 24);
    checkVoice();
    if (S.config.features.transcribe) transcribeUpload();
    else $('#voice-status').textContent = t('v.asrOff');
  } catch (e) { $('#voice-drop-text').textContent = errText(e); }
}
async function transcribeUpload() {
  if (!vd.upload) return;
  $('#voice-status').textContent = t('v.asrRunning');
  $('#voice-asr').disabled = true;
  try {
    const d = await api('POST', 'transcribe', { upload_id: vd.upload, lang: $('#voice-lang').value });
    $('#voice-text').value = d.text;
    $('#voice-status').textContent = t('v.asrDone');
  } catch (e) { $('#voice-status').textContent = errText(e); }
  $('#voice-asr').disabled = false;
  checkVoice();
}
$('#voice-file').addEventListener('change', e => e.target.files[0] && uploadFile(e.target.files[0]));
$('#voice-asr').onclick = transcribeUpload;
const drop = $('#voice-drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]); });
$('#voice-ok').onclick = async e => {
  e.preventDefault();
  const body = { name: $('#voice-name').value.trim(), ref_text: $('#voice-text').value.trim(),
                 gender: $('#voice-gender').value, lang: $('#voice-lang').value };
  try {
    if (vd.mode === 'edit') {
      await api('PUT', 'voices/' + encodeURIComponent(vd.voice.id), body);
      $('#dlg-voice').close();
      await loadVoices(); renderLcd();
      lcd.flash(t('msg.voiceSaved'));
      return;
    }
    Object.assign(body, vd.from || { upload_id: vd.upload });
    const r = await api('POST', 'voices', body);
    $('#dlg-voice').close();
    await loadVoices();
    setSetting('voice', r.id);
    S.lib = { ...S.lib, tab: 'custom', lang: '', gender: '', q: '' };
    $('#f-search').value = '';
    renderAll();
    lcd.flash(t(vd.mode === 'lock' ? 'msg.locked' : 'msg.voiceAdded', { name: body.name }), false, 7000);
  } catch (err) { $('#voice-status').textContent = errText(err); }
};
$('#voice-del').onclick = async () => {
  const v = vd.voice;
  if (!confirm(t('c.delVoice', { name: v.name }))) return;
  await api('DELETE', 'voices/' + encodeURIComponent(v.id)).catch(err => lcd.flash(errText(err), true));
  $('#dlg-voice').close();
  if (settings().voice === v.id) setSetting('voice', S.config.defaults.voice || 'system:Vivian');
  await loadVoices(); renderAll();
};
$('#dlg-voice').addEventListener('close', () => $('#voice-audio').pause());

/* ───────── export ───────── */
function openExport() {
  if (!S.project) return lcd.flash(t('msg.needScript'), true);
  const ps = S.project.paragraphs, done = ps.filter(p => p.chosen).length;
  $('#export-sum').textContent = t('x.sum', { n: ps.length, done, len: fmtClock(totalLength()).replace(/\.\d$/, '') })
    + (done < ps.length ? t('x.skip', { k: ps.length - done }) : '') + t('x.note');
  $('#export-status').textContent = '';
  $('#dlg-export').showModal();
}
$('#export-ok').onclick = async e => {
  e.preventDefault();
  const fmt = $('#export-fmt input:checked').value;
  $('#export-status').textContent = t('x.running');
  try {
    await saveSettings();
    const r = await api('POST', `projects/${S.project.id}/export`, { format: fmt });
    const a = document.createElement('a');
    a.href = r.url; a.download = r.name; document.body.appendChild(a); a.click(); a.remove();
    $('#export-status').innerHTML = `${esc(t('x.done'))}<a href="${r.url}" download>${esc(r.name)}</a>`;
  } catch (err) { $('#export-status').textContent = errText(err); }
};

/* ───────── UI language ───────── */
function renderChrome() {
  applyI18n();
  const c = S.config;
  $('#model-line').textContent = [t('model'), c.device_name].filter(Boolean).join(' · ');
  const home = $('#home-link');
  home.hidden = !c.home_url;
  if (c.home_url) { home.href = c.home_url; home.textContent = t('home'); home.title = t('home'); }
  document.title = t('brand');
  buildMoods();
}
$('.ui-lang').addEventListener('click', e => {
  const b = e.target.closest('[data-ui-lang]');
  if (!b || b.dataset.uiLang === LANG) return;
  LANG = b.dataset.uiLang;
  localStorage.setItem('tts.lang', LANG);
  renderChrome();
  renderAll();
});

/* ───────── start ───────── */
window.addEventListener('resize', () => renderTracks());
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => renderTracks());
(async function init() {
  S.config = await api('GET', 'config').catch(() => S.config);
  const saved = JSON.parse(localStorage.getItem('tts.draft') || 'null');
  S.draft = { ...S.config.defaults, ...(saved || {}) };
  if (!S.draft.voice) S.draft.voice = 'system:Vivian';
  vu.build(); buildKnobs(); renderChrome();
  await loadVoices().catch(() => lcd.flash(t('msg.offline'), true));
  if (!voiceById(S.draft.voice)) S.draft.voice = S.config.defaults.voice || 'system:Vivian';
  showVoiceTab();
  const want = location.hash.slice(1) || localStorage.getItem('tts.last');
  let ok = want && await openProject(want);
  if (!ok) {
    const list = await api('GET', 'projects').catch(() => []);
    if (list[0]) ok = await openProject(list[0].id);
  }
  if (!ok) renderAll();
  poll();
})();
