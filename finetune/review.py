"""A small web page for checking clips before training: listen, fix the transcript, keep or drop.

Edits go to WORK/<speaker>/edits.json (keyed by clip id), so re-running prep_data.py never loses them.
Usage:  .venv/bin/python review.py [--config voiceprep.json] [--port 18410]   →  http://127.0.0.1:18410
        listening comparison of fine-tuned epochs   →  http://127.0.0.1:18410/compare/<speaker>
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

from common import add_config_arg, load_config

WORK = None   # set from the config in __main__


def speakers():
    return sorted(d for d in os.listdir(WORK) if os.path.isfile(os.path.join(WORK, d, "manifest.jsonl")))


def load(spk):
    rows = [json.loads(l) for l in open(os.path.join(WORK, spk, "manifest.jsonl"), encoding="utf-8")]
    ep = os.path.join(WORK, spk, "edits.json")
    edits = json.load(open(ep, encoding="utf-8")) if os.path.exists(ep) else {}
    for r in rows:
        r.update(edits.get(r["id"], {}))
        r["edited"] = r["id"] in edits
    return [r for r in rows if os.path.exists(os.path.join(WORK, r["audio"]))]


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Clip review</title>
<style>
body{font:14px -apple-system,system-ui,sans-serif;margin:0;background:#f6f6f4;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:15px} select,button{font:inherit}
.sum{color:#555} .sum em{font-style:normal;font-weight:600;color:#1a7f37}
.help{color:#888;font-size:12px}
table{border-collapse:collapse;width:100%} td{padding:6px 8px;border-bottom:1px solid #e6e6e6;vertical-align:middle}
tr.cur{background:#fff8d6} tr.drop td:not(.st){opacity:.45} tr.drop input{text-decoration:line-through}
td.t{width:100%} td.t input{width:100%;box-sizing:border-box;font:15px/1.5 inherit;padding:4px 6px;border:1px solid #ccc;border-radius:4px}
td.t input.ed{border-color:#d29922;background:#fffbea}
.m{font:12px ui-monospace,monospace;color:#666;white-space:nowrap}
.low{color:#cf222e;font-weight:600}
.st button{width:64px;border:1px solid #ccc;border-radius:4px;padding:3px 0;cursor:pointer;background:#fff}
.st button.keep{background:#dafbe1;border-color:#1a7f37} .st button.drop{background:#ffebe9;border-color:#cf222e}
.st button.review{background:#fff8c5;border-color:#bf8700} .st button.dup{background:#eee}
.flag{display:inline-block;font-size:11px;padding:0 5px;margin-left:3px;border-radius:8px;background:#eef;color:#447}
.play{cursor:pointer;border:none;background:none;font-size:18px}
</style></head><body>
<header><b>Clip review</b>
<select id="spk"></select>
<select id="filter"><option value="">全部</option><option value="review">待定</option><option value="keep">保留</option>
<option value="drop">丢弃</option><option value="dup">重复</option><option value="flag">有标记</option><option value="edited">已改</option></select>
<span class="sum" id="sum"></span>
<span class="help">↑↓ 选择 · 空格 播放 · K 保留 · D 丢弃 · Enter 编辑文字 · Esc 退出编辑（改动自动保存）</span>
</header>
<table><tbody id="rows"></tbody></table>
<audio id="au"></audio>
<script>
const $ = s => document.querySelector(s), au = $('#au');
let rows = [], view = [], cur = 0, spk = '';
const FLAG = {english:'英文', filler:'语气词', repeat:'重复/口误', number:'数字'};
async function init() {
  const list = await (await fetch('/api/speakers')).json();
  $('#spk').innerHTML = list.map(s => `<option>${s}</option>`).join('');
  $('#spk').onchange = () => loadSpk($('#spk').value);
  $('#filter').onchange = render;
  loadSpk(list[0]);
}
async function loadSpk(s) { spk = s; rows = await (await fetch('/api/rows/' + encodeURIComponent(s))).json(); cur = 0; render(); }
function summary() {
  const m = st => (rows.filter(r => r.status === st).reduce((a, r) => a + r.dur, 0) / 60).toFixed(1);
  $('#sum').innerHTML = `保留 <em>${m('keep')}</em> 分钟 · 待定 ${m('review')} · 重复 ${m('dup')} · 丢弃 ${m('drop')} · 共 ${rows.length} 段`;
}
function render() {
  const f = $('#filter').value;
  view = rows.filter(r => !f || (f === 'flag' ? r.flags.length : f === 'edited' ? r.edited : r.status === f));
  $('#rows').innerHTML = view.map((r, i) => `<tr data-i="${i}" class="${r.status}">
    <td><button class="play">▶</button></td>
    <td class="st"><button class="${r.status}">${{keep:'保留',drop:'丢弃',review:'待定',dup:'重复'}[r.status]}</button></td>
    <td class="t"><input value="${r.text.replace(/"/g, '&quot;')}" class="${r.edited ? 'ed' : ''}">
      <div class="m">${r.src} @${r.start}s${r.flags.map(x => `<span class="flag">${FLAG[x] || x}</span>`).join('')}${r.dup_of ? ' 同 ' + r.dup_of : ''}</div></td>
    <td class="m">${r.dur.toFixed(1)}s<br>相似 <span class="${r.sim < .55 ? 'low' : ''}">${r.sim.toFixed(2)}</span><br>信噪 <span class="${r.snr < 25 ? 'low' : ''}">${r.snr}</span></td></tr>`).join('');
  summary(); select(Math.min(cur, view.length - 1), false);
}
function select(i, scroll = true) {
  if (i < 0 || i >= view.length) return;
  document.querySelectorAll('tr.cur').forEach(t => t.classList.remove('cur'));
  cur = i; const tr = $(`tr[data-i="${i}"]`); tr.classList.add('cur');
  if (scroll) tr.scrollIntoView({block: 'center'});
}
function play(i) {
  select(i, false);
  const url = '/audio/' + view[i].audio.split('/').map(encodeURIComponent).join('/');
  if (au.dataset.i == i && !au.paused) return au.pause();
  au.src = url; au.dataset.i = i; au.play();
}
async function save(r, patch) {
  Object.assign(r, patch, {edited: true});
  await fetch('/api/edit/' + encodeURIComponent(spk), {method: 'POST', body: JSON.stringify({id: r.id, ...patch})});
}
function setStatus(i, st) {
  const r = view[i]; if (!r) return;
  save(r, {status: r.status === st ? 'review' : st});
  const tr = $(`tr[data-i="${i}"]`), b = tr.querySelector('.st button');
  tr.className = r.status + ' cur'; b.className = r.status;
  b.textContent = {keep:'保留',drop:'丢弃',review:'待定',dup:'重复'}[r.status]; summary();
}
document.addEventListener('click', e => {
  const tr = e.target.closest('tr'); if (!tr) return; const i = +tr.dataset.i;
  if (e.target.classList.contains('play')) play(i);
  else if (e.target.closest('.st')) setStatus(i, view[i].status === 'keep' ? 'drop' : 'keep');
  else select(i, false);
});
document.addEventListener('change', e => {
  if (e.target.tagName !== 'INPUT') return;
  const i = +e.target.closest('tr').dataset.i; save(view[i], {text: e.target.value.trim()}); e.target.classList.add('ed');
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') { if (e.key === 'Escape' || e.key === 'Enter') { e.target.blur(); } return; }
  if (e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === 'arrowdown' || k === 'j') { select(cur + 1); e.preventDefault(); }
  else if (k === 'arrowup') { select(cur - 1); e.preventDefault(); }
  else if (k === ' ') { play(cur); e.preventDefault(); }
  else if (k === 'k') { setStatus(cur, 'keep'); select(cur + 1); }
  else if (k === 'd') { setStatus(cur, 'drop'); select(cur + 1); }
  else if (k === 'enter') { $(`tr[data-i="${cur}"] input`).focus(); e.preventDefault(); }
});
init();
</script></body></html>"""


def compare_page(spk):
    """Listening page for gen_samples.py output: per sentence, one row per model."""
    idx = json.load(open(os.path.join(WORK, spk, "samples", "index.json"), encoding="utf-8"))
    scores = idx.get("scores", {})
    au = lambda rel: f'<audio controls preload="none" src="/audio/{quote(spk)}/{quote(rel)}"></audio>'
    real = [r for r in load(spk) if r["status"] == "keep"][:: max(1, len(load(spk)) // 4)][:3]
    h = [f"<!doctype html><meta charset=utf-8><title>{spk} 对比</title><style>"
         "body{font:14px -apple-system,sans-serif;margin:20px;color:#222}td,th{padding:4px 10px;text-align:left}"
         "h3{margin:28px 0 6px}.m{font:12px ui-monospace,monospace;color:#666}audio{height:30px;width:250px}"
         "tr:nth-child(even){background:#f4f4f2}</style>",
         f"<h2>{spk} · 微调版本对比</h2><p class=m>相似=与真人声纹的相似度（越高越像）· 错字=转写错字率 · "
         f"语速=字/秒（真人 {idx.get('real_rate', 0):.2f}）· 同一列用同一随机种子</p>",
         "<h3>真人录音（参照）</h3>" + "".join(f"<div>{au(r['audio'].split('/', 1)[1])} {r['text']}</div>" for r in real)]
    for i, text in enumerate(idx["sentences"]):
        h.append(f"<h3>{i + 1}. {text}</h3><table><tr><th>版本</th>"
                 + "".join(f"<th>种子 {s}</th>" for s in idx["seeds"]) + "<th>平均</th></tr>")
        for m in idx["models"]:
            rs = [r for r in scores.get(m, []) if r["s"] == i]
            avg = (f"相似 {sum(r['sim'] for r in rs) / len(rs):.3f}<br>错字 {sum(r['cer'] for r in rs) / len(rs):.0%}"
                   f"<br>语速 {sum(r['rate'] for r in rs) / len(rs):.2f}") if rs else ""
            h.append(f"<tr><td><b>{m}</b></td>" + "".join(f"<td>{au(f'samples/{m}/s{i}_{s}.wav')}</td>" for s in idx["seeds"])
                     + f"<td class=m>{avg}</td></tr>")
        h.append("</table>")
    return "".join(h)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
        if not parts:
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if parts == ["api", "speakers"]:
            return self.send(json.dumps(speakers()).encode(), "application/json")
        if parts[:2] == ["api", "rows"] and len(parts) == 3 and parts[2] in speakers():
            return self.send(json.dumps(load(parts[2]), ensure_ascii=False).encode(), "application/json")
        if parts[0] == "compare" and len(parts) == 2 and parts[1] in speakers() and \
                os.path.exists(os.path.join(WORK, parts[1], "samples", "index.json")):
            return self.send(compare_page(parts[1]).encode(), "text/html; charset=utf-8")
        if parts[0] == "audio":
            path = os.path.realpath(os.path.join(WORK, *parts[1:]))
            if path.startswith(WORK + os.sep) and path.endswith(".wav") and os.path.isfile(path):
                return self.send(open(path, "rb").read(), "audio/wav")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
        if parts[:2] != ["api", "edit"] or len(parts) != 3 or parts[2] not in speakers():
            return self.send(b"not found", "text/plain", 404)
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        ep = os.path.join(WORK, parts[2], "edits.json")
        edits = json.load(open(ep, encoding="utf-8")) if os.path.exists(ep) else {}
        e = edits.setdefault(b["id"], {})
        e.update({k: b[k] for k in ("text", "status") if k in b})
        tmp = ep + ".tmp"
        json.dump(edits, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, ep)
        self.send(b"{}", "application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--host", default="127.0.0.1", help="no auth: keep it on localhost (use ssh -L to reach it)")
    ap.add_argument("--port", type=int, default=18410)
    a = ap.parse_args()
    WORK = os.path.realpath(load_config(a.config)["work_dir"])
    print(f"review: http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
