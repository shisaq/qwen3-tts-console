#!/usr/bin/env python3
"""ttsd — the backend of Qwen3-TTS Console.

One process serves the web UI (web/), the JSON API (/api/*), audio files (/media/*) and a
generation queue.

- Models load on demand (system voices use the CustomVoice model, custom voices use the Base
  model for cloning) and unload after `idle_unload_minutes` so the memory goes back to other apps.
- All MLX calls happen on a single worker thread: one generation at a time.
- A script is a list of paragraphs; each paragraph can hold several takes, one of which is chosen.
  Speed and breath removal are rendered with ffmpeg at playback / export time, so they never
  require re-generating.

Settings live in a JSON file (see config.example.json); every key is optional.
Errors are returned as {"error": <code>, "message": <text>} so the UI can translate them.
"""
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
import soundfile as sf

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(APP_DIR, "web")
SR = 24000
MAX_UPLOAD = 60 * 1024 * 1024


def load_config():
    path = os.environ.get("TTSD_CONFIG") or os.path.expanduser("~/.config/ttsd/config.json")
    cfg = {
        "host": "127.0.0.1",
        "port": 18400,
        "models_dir": "models",                 # relative paths are relative to this file
        "data_dir": "~/.local/share/ttsd",
        "preset_model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "clone_model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "ffmpeg": None,                         # None = look up in PATH
        "transcribe_python": None,              # a Python that has faster-whisper; None = feature off
        "whisper_model": "base",
        "idle_unload_minutes": 10,
        "device_name": "",                      # shown on the nameplate, e.g. the host name
        "home_url": "",                         # optional "back" link in the footer
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))

    def resolve(p):
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.join(APP_DIR, p)

    cfg["models_dir"] = resolve(cfg["models_dir"])
    cfg["data_dir"] = resolve(cfg["data_dir"])
    cfg["ffmpeg"] = cfg["ffmpeg"] or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    tp = cfg.get("transcribe_python")
    cfg["transcribe_python"] = os.path.expanduser(tp) if tp else None
    return cfg


CFG = load_config()
DATA = CFG["data_dir"]
IDLE_UNLOAD = max(60, int(CFG["idle_unload_minutes"] * 60))

# The 9 speakers built into the CustomVoice model (config.json → spk_id).
# gender: f / m;  langs: what the voice speaks natively;  dialect: Chinese dialect, if any.
SYSTEM_VOICES = [
    {"name": "Vivian",   "gender": "f", "langs": ["zh", "en"], "dialect": None,      "color": "#ff5a1f"},
    {"name": "Serena",   "gender": "f", "langs": ["zh", "en"], "dialect": None,      "color": "#e9b44c"},
    {"name": "Uncle_Fu", "gender": "m", "langs": ["zh"],       "dialect": None,      "color": "#7a5c3e"},
    {"name": "Dylan",    "gender": "m", "langs": ["zh"],       "dialect": "beijing", "color": "#3d6fd8"},
    {"name": "Eric",     "gender": "m", "langs": ["zh"],       "dialect": "sichuan", "color": "#c83f3f"},
    {"name": "Ryan",     "gender": "m", "langs": ["en"],       "dialect": None,      "color": "#2f8f6b"},
    {"name": "Aiden",    "gender": "m", "langs": ["en"],       "dialect": None,      "color": "#4aa3c7"},
    {"name": "Ono_Anna", "gender": "f", "langs": ["ja"],       "dialect": None,      "color": "#d96aa7"},
    {"name": "Sohee",    "gender": "f", "langs": ["ko"],       "dialect": None,      "color": "#8a6fd1"},
]
SYSTEM_NAMES = {v["name"] for v in SYSTEM_VOICES}
GEN_LANGS = {"auto", "chinese", "english", "japanese", "korean"}
VOICE_LANGS = {"zh", "en", "ja", "ko", "other"}
GENDERS = {"f", "m", ""}

DEFAULT_SETTINGS = {
    "voice": "system:Vivian",
    "instruct": "",
    "lang": "auto",
    "temperature": 0.9,
    "speed": 1.0,       # post: atempo
    "gate": -40,        # post: breath-removal threshold in dB; None = off
    "keep": 0.05,       # post: silence kept after breath removal (s)
    "gap": 0.3,         # export: gap between paragraphs (s)
}

io_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status, code, message=""):
        super().__init__(message or code)
        self.status, self.code, self.message = status, code, message or code


def now():
    return time.time()


def new_id(n=10):
    return uuid.uuid4().hex[:n]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def safe_id(s):
    return isinstance(s, str) and re.fullmatch(r"[0-9a-f]{6,32}", s) is not None


def write_json(path, obj):
    tmp = f"{path}.{new_id(6)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def peaks_of(audio, bins=160):
    a = np.abs(np.asarray(audio, dtype=np.float32))
    if a.size == 0:
        return [0] * bins
    p = np.array([c.max() if c.size else 0 for c in np.array_split(a, bins)])
    top = max(float(p.max()), 1e-4)
    return [round(float(x / top), 3) for x in p]


def split_paragraphs(text):
    text = text.replace("\r\n", "\n").strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


# ───────────────────────── voices ─────────────────────────

def vdir(vid):
    return os.path.join(DATA, "voices", vid)


def custom_meta(vid):
    return read_json(os.path.join(vdir(vid), "meta.json")) if safe_id(vid) else None


def list_voices():
    out = [{"id": f"system:{v['name']}", "kind": "system", **v} for v in SYSTEM_VOICES]
    root = os.path.join(DATA, "voices")
    metas = [custom_meta(vid) for vid in (os.listdir(root) if os.path.isdir(root) else [])]
    for m in sorted(filter(None, metas), key=lambda m: m["created"]):
        out.append({"id": f"custom:{m['id']}", "kind": "custom", "name": m["name"],
                    "gender": m.get("gender", ""), "langs": [m.get("lang") or "zh"], "dialect": None,
                    "color": "#2b2a27", "ref_text": m["ref_text"], "source": m.get("source", "upload"),
                    "source_title": m.get("source_title", ""), "duration": m.get("duration")})
    return out


def valid_voice(v):
    if not isinstance(v, str) or ":" not in v:
        return False
    kind, name = v.split(":", 1)
    if kind == "system":
        return name in SYSTEM_NAMES
    return kind == "custom" and custom_meta(name) is not None


def voice_name(v):
    kind, name = v.split(":", 1)
    if kind == "system":
        return name
    return (custom_meta(name) or {}).get("name", "?")


def clean_voice_fields(b, meta):
    if "name" in b:
        name = str(b["name"] or "").strip()[:24]
        if not name:
            raise ApiError(400, "need_name")
        meta["name"] = name
    if "ref_text" in b:
        text = str(b["ref_text"] or "").strip()[:1000]
        if not text:
            raise ApiError(400, "need_ref_text")
        meta["ref_text"] = text
    if b.get("gender") in GENDERS:
        meta["gender"] = b["gender"]
    if b.get("lang") in VOICE_LANGS:
        meta["lang"] = b["lang"]
    return meta


# ───────────────────────── scripts (projects) ─────────────────────────

def pdir(pid):
    return os.path.join(DATA, "projects", pid)


def load_project(pid):
    return read_json(os.path.join(pdir(pid), "project.json")) if safe_id(pid) else None


def save_project(p):
    p["updated"] = now()
    os.makedirs(pdir(p["id"]), exist_ok=True)
    write_json(os.path.join(pdir(p["id"]), "project.json"), p)


def list_projects():
    root = os.path.join(DATA, "projects")
    out = []
    for pid in os.listdir(root) if os.path.isdir(root) else []:
        p = load_project(pid)
        if p:
            paras = p["paragraphs"]
            out.append({"id": p["id"], "title": p["title"], "updated": p["updated"],
                        "paragraphs": len(paras), "chosen": sum(1 for x in paras if x.get("chosen"))})
    return sorted(out, key=lambda x: -x["updated"])


def clean_settings(src, base):
    """Merge the valid keys of `src` into a copy of `base`."""
    s = dict(base)
    for k, v in (src or {}).items():
        if k not in DEFAULT_SETTINGS:
            continue
        try:
            if k == "voice":
                if not valid_voice(v):
                    continue
            elif k == "lang":
                if v not in GEN_LANGS:
                    continue
            elif k == "instruct":
                v = str(v or "")[:200]
            elif k == "gate":
                v = None if v is None else min(-10.0, max(-70.0, float(v)))
            elif k == "temperature":
                v = min(1.5, max(0.1, float(v)))
            elif k == "speed":
                v = min(2.0, max(0.5, float(v)))
            elif k in ("keep", "gap"):
                v = min(5.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            continue
        s[k] = v
    return s


def remove_take_files(pid, tid):
    raw = os.path.join(pdir(pid), "takes", f"{tid}.wav")
    if os.path.exists(raw):
        os.remove(raw)
    cache = os.path.join(pdir(pid), "cache")
    if os.path.isdir(cache):
        for fn in os.listdir(cache):
            if fn.startswith(tid + "_"):
                os.remove(os.path.join(cache, fn))


# ───────────────────────── audio post-processing ─────────────────────────

def ffmpeg(args, timeout=300):
    r = subprocess.run([CFG["ffmpeg"], "-y", "-loglevel", "error", *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr.strip().splitlines() or ["failed"])[-1])


def post_params(q, settings):
    def num(name, lo, hi):
        v = q.get(name, settings.get(name))
        if v in (None, "", "off", "null"):
            return None
        return min(hi, max(lo, float(v)))
    speed = num("speed", 0.5, 2.0) or 1.0
    gate = num("gate", -70, -10)
    keep = num("keep", 0.0, 5.0)
    return speed, gate, 0.05 if keep is None else keep


def render_take(pid, tid, speed, gate, keep):
    """Apply speed + breath removal and cache the result; defaults return the raw take."""
    raw = os.path.join(pdir(pid), "takes", f"{tid}.wav")
    if not safe_id(tid) or not os.path.exists(raw):
        return None
    filters = []
    if abs(speed - 1.0) > 0.005:
        filters.append(f"atempo={speed:.3f}")
    if gate is not None:
        # start_*: also trim leading silence (cloned takes often start with 0.3–0.5 s),
        # otherwise the gap between paragraphs ends up longer than the GAP knob says.
        # start_silence keeps 60 ms so the first syllable never starts at sample 0.
        filters.append(
            f"silenceremove=start_periods=1:start_threshold={gate:.0f}dB:start_silence=0.06:"
            f"stop_periods=-1:stop_duration={max(keep, 0.02) + 0.1:.2f}:"
            f"stop_silence={keep:.2f}:stop_threshold={gate:.0f}dB:detection=rms:window=0.005")
    if not filters:
        return raw
    cache = os.path.join(pdir(pid), "cache")
    os.makedirs(cache, exist_ok=True)
    g = "off" if gate is None else f"{int(gate)}"
    out = os.path.join(cache, f"{tid}_s{speed:.2f}_g{g}_k{keep:.2f}.wav")
    if not os.path.exists(out):
        tmp = out + f".{new_id(6)}.wav"
        ffmpeg(["-i", raw, "-af", ",".join(filters), "-ar", str(SR), "-ac", "1", tmp])
        os.replace(tmp, out)
    return out


# ───────────────────────── generation engine ─────────────────────────

class Engine:
    def __init__(self):
        self.models = {}
        self.loading = None
        self.last_used = 0.0
        self.jobs = {}
        self.queue = deque()
        self.cv = threading.Condition()
        self.rev = 0                   # +1 whenever a job ends; the UI reloads on change
        self.unload_req = False
        # seconds of compute per second of audio, used to estimate progress (rolling average)
        self.cost = {"preset": 0.6, "clone": 0.9}
        self.load_cost = {"preset": 6.0, "clone": 6.0}
        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, pid, para_id, text, settings):
        job = {"id": new_id(), "pid": pid, "para_id": para_id, "text": text,
               "settings": dict(settings), "state": "queued", "created": now(),
               "progress": 0.0, "error": None}
        with self.cv:
            self.jobs[job["id"]] = job
            self.queue.append(job["id"])
            self.cv.notify()
        return job

    def cancel(self, jid):
        with self.cv:
            j = self.jobs.get(jid)
            if not j:
                return False
            if j["state"] == "queued":
                j["state"] = "canceled"
                try:
                    self.queue.remove(jid)
                except ValueError:
                    pass
                self.rev += 1
                return True
            if j["state"] == "running":
                j["cancel"] = True          # discarded when it finishes
                return True
        return False

    def request_unload(self):
        with self.cv:
            self.unload_req = True
            self.cv.notify()

    def status(self):
        with self.cv:
            cur = next((j for j in self.jobs.values() if j["state"] == "running"), None)
            active = [self.public(j) for j in self.jobs.values() if j["state"] in ("queued", "running")]
            idle_in = None
            if self.models and not active:
                idle_in = max(0, int(IDLE_UNLOAD - (now() - self.last_used)))
            return {
                "models": {"preset": "preset" in self.models, "clone": "clone" in self.models},
                "loading": self.loading,
                "busy": bool(active),
                "current": self.public(cur) if cur else None,
                "jobs": sorted(active, key=lambda j: j["created"]),
                "rev": self.rev,
                "idle_unload_in": idle_in,
                "errors": [self.public(j) for j in self.jobs.values()
                           if j["state"] == "error" and now() - j.get("ended", 0) < 120],
            }

    def public(self, j):
        out = {k: j.get(k) for k in ("id", "pid", "para_id", "state", "progress", "error", "created")}
        out["voice"] = j["settings"].get("voice")
        if j["state"] == "running" and j.get("started"):
            out["progress"] = round(min(0.97, (now() - j["started"]) / max(j.get("est", 5), 1)), 3)
        return out

    def loop(self):
        while True:
            with self.cv:
                while not self.queue and not self.unload_req:
                    # with a model in memory, wake up when the idle period is over
                    timeout = max(1.0, IDLE_UNLOAD - (now() - self.last_used)) if self.models else None
                    self.cv.wait(timeout=timeout)
                    if self.models and not self.queue and now() - self.last_used >= IDLE_UNLOAD:
                        self.unload_req = True
                self.unload_req = False
                jid = self.queue.popleft() if self.queue else None
                if jid:
                    self.jobs[jid]["state"] = "running"
            if jid is None:
                self.unload()
                continue
            self.run(self.jobs[jid])
            self.prune()

    def prune(self):
        with self.cv:
            done = [j for j in self.jobs.values() if j["state"] in ("done", "error", "canceled")]
            for j in sorted(done, key=lambda j: j["created"])[:-60]:
                self.jobs.pop(j["id"], None)

    def unload(self):
        if not self.models:
            return
        import mlx.core as mx
        self.models.clear()
        gc.collect()
        mx.clear_cache()
        log("models unloaded")

    def model(self, kind):
        if kind in self.models:
            return self.models[kind]
        from mlx_audio.tts.utils import load_model
        self.loading = kind
        t0 = now()
        try:
            folder = CFG["preset_model"] if kind == "preset" else CFG["clone_model"]
            m = load_model(os.path.join(CFG["models_dir"], folder))
        finally:
            self.loading = None
        self.load_cost[kind] = now() - t0
        self.models[kind] = m
        log(f"loaded {kind} model ({now() - t0:.1f}s)")
        return m

    @staticmethod
    def estimate_seconds(text):
        cjk = len(re.findall(r"[぀-ヿ㐀-鿿가-힯]", text))
        words = len(re.findall(r"[A-Za-z0-9]+", text))
        return 0.6 + cjk * 0.24 + words * 0.38

    def run(self, job):
        s = job["settings"]
        voice = s["voice"]
        kind = "clone" if voice.startswith("custom:") else "preset"
        job["started"] = now()
        est = self.estimate_seconds(job["text"]) * self.cost[kind]
        if kind not in self.models:
            est += self.load_cost[kind]
        job["est"] = est
        p = None
        try:
            import mlx.core as mx
            kw = dict(text=job["text"], temperature=float(s["temperature"]),
                      lang_code=s.get("lang") or "auto", top_k=50, top_p=1.0,
                      repetition_penalty=1.05, split_pattern="")
            if kind == "preset":
                name = voice.split(":", 1)[1]
                if name not in SYSTEM_NAMES:
                    raise ApiError(400, "voice_missing")
                kw.update(voice=name, instruct=(s.get("instruct") or "").strip() or None)
            else:
                vid = voice.split(":", 1)[1]
                meta = custom_meta(vid)
                if not meta:
                    raise ApiError(400, "voice_missing")
                kw.update(ref_audio=os.path.join(vdir(vid), "ref.wav"), ref_text=meta["ref_text"])
            m = self.model(kind)
            seed = random.randint(1, 2**31 - 1)
            mx.random.seed(seed)
            t_gen = now()
            parts = [np.array(r.audio, dtype=np.float32) for r in m.generate(**kw)]
            audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            gen_time = now() - t_gen
            dur = len(audio) / SR
            if dur < 0.2:
                raise ApiError(500, "no_audio")
            if dur > 1:
                self.cost[kind] = 0.7 * self.cost[kind] + 0.3 * (gen_time / dur)
            if job.get("cancel"):
                job["state"] = "canceled"
                return
            tid = new_id()
            take = {
                "id": tid, "created": now(), "voice": voice, "voice_name": voice_name(voice),
                "instruct": kw.get("instruct") or "", "lang": kw["lang_code"],
                "temperature": kw["temperature"], "seed": seed, "text": job["text"],
                "duration": round(dur, 2), "gen_time": round(gen_time, 1), "peaks": peaks_of(audio),
            }
            with io_lock:
                p = load_project(job["pid"])
                para = p and next((x for x in p["paragraphs"] if x["id"] == job["para_id"]), None)
                if not para:
                    job["state"] = "canceled"     # script or paragraph deleted meanwhile
                    return
                os.makedirs(os.path.join(pdir(p["id"]), "takes"), exist_ok=True)
                sf.write(os.path.join(pdir(p["id"]), "takes", f"{tid}.wav"), audio, SR, subtype="PCM_16")
                para.setdefault("takes", []).append(take)
                para["chosen"] = tid              # select the new take so it can be heard right away
                save_project(p)
            job["state"] = "done"
            log(f"take done {p['id']}/{job['para_id']} {dur:.1f}s audio in {gen_time:.1f}s")
        except ApiError as e:
            job["state"], job["error"] = "error", e.code
            log("generation failed:", e.code)
        except Exception as e:  # noqa: BLE001
            job["state"], job["error"] = "error", "generate_failed"
            log("generation failed:", repr(e))
        finally:
            job["ended"] = now()
            self.last_used = now()
            with self.cv:
                self.rev += 1


ENGINE = None


# ───────────────────────── transcription / export ─────────────────────────

WHISPER_SCRIPT = r"""
import sys
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[2], device="cpu", compute_type="int8")
prompt = "以下是普通话的句子，使用简体中文。" if sys.argv[3] == "zh" else None
lang = None if sys.argv[3] in ("", "other") else sys.argv[3]
segs, info = m.transcribe(sys.argv[1], language=lang, initial_prompt=prompt)
print("".join(s.text for s in segs).strip())
"""


def transcribe(path, lang):
    py = CFG["transcribe_python"]
    if not py or not os.path.exists(py):
        raise ApiError(501, "transcribe_unavailable")
    r = subprocess.run([py, "-c", WHISPER_SCRIPT, path, CFG["whisper_model"], lang or ""],
                       capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, HF_HUB_OFFLINE="1"))
    if r.returncode != 0:
        log("transcribe failed:", r.stderr.strip()[-300:])
        raise ApiError(500, "transcribe_failed")
    return r.stdout.strip()


def export_project(p, fmt):
    s = p["settings"]
    speed, gate, keep = post_params({}, s)
    gap = float(s.get("gap", 0.3))
    clips = []
    for i, para in enumerate(p["paragraphs"], 1):
        if para.get("chosen"):
            path = render_take(p["id"], para["chosen"], speed, gate, keep)
            if path:
                clips.append((i, path))
    if not clips:
        raise ApiError(400, "nothing_to_export")
    out_dir = os.path.join(pdir(p["id"]), "exports")
    os.makedirs(out_dir, exist_ok=True)
    base = re.sub(r"[\\/:*?\"<>|\s]+", "_", p["title"]).strip("_")[:40] or "tts"
    stamp = time.strftime("%m%d-%H%M")
    if fmt == "zip":
        name = f"{base}-{stamp}.zip"
        with zipfile.ZipFile(os.path.join(out_dir, name), "w", zipfile.ZIP_DEFLATED) as z:
            for i, path in clips:
                z.write(path, f"{i:02d}.wav")
        return name
    silence = np.zeros(int(gap * SR), np.float32)
    pieces = []
    for n, (_, path) in enumerate(clips):
        a, _ = sf.read(path, dtype="float32")
        if n:
            pieces.append(silence)
        pieces.append(a)
    wav_name = f"{base}-{stamp}.wav"
    sf.write(os.path.join(out_dir, wav_name), np.concatenate(pieces), SR, subtype="PCM_16")
    if fmt == "mp3":
        mp3_name = f"{base}-{stamp}.mp3"
        ffmpeg(["-i", os.path.join(out_dir, wav_name), "-b:a", "192k", os.path.join(out_dir, mp3_name)])
        os.remove(os.path.join(out_dir, wav_name))
        return mp3_name
    return wav_name


# ───────────────────────── HTTP ─────────────────────────

MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".zip": "application/zip", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    server_version = "ttsd"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body_bytes(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, "file_too_large")
        return self.rfile.read(n) if n else b""

    def body_json(self):
        try:
            return json.loads(self.body_bytes() or b"{}")
        except ValueError:
            raise ApiError(400, "bad_request")

    def send_file(self, path, download_name=None, cache=False):
        if not path or not os.path.isfile(path):
            raise ApiError(404, "not_found")
        size = os.path.getsize(path)
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):           # Safari needs ranges to seek
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-cache")
        if download_name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(download_name)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def handle_any(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        parts = [unquote(x) for x in u.path.split("/") if x]
        try:
            if parts[:1] == ["api"]:
                return self.api(self.command, parts[1:], q)
            if parts[:1] == ["media"] and self.command in ("GET", "HEAD"):
                return self.media(parts[1:], q)
            if self.command in ("GET", "HEAD"):
                path = os.path.realpath(os.path.join(WEB_DIR, "/".join(parts) or "index.html"))
                if not path.startswith(os.path.realpath(WEB_DIR) + os.sep):
                    raise ApiError(404, "not_found")
                return self.send_file(path)
            raise ApiError(405, "not_allowed")
        except ApiError as e:
            self.send_json({"error": e.code, "message": e.message}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            log("api error:", self.command, self.path, repr(e))
            try:
                self.send_json({"error": "internal", "message": str(e)[:300]}, 500)
            except OSError:
                pass

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = handle_any

    def media(self, parts, q):
        if len(parts) == 3 and parts[0] == "take":
            p = load_project(parts[1])
            if not p:
                raise ApiError(404, "not_found")
            tid = parts[2].removesuffix(".wav")
            path = render_take(p["id"], tid, *post_params(q, p["settings"]))
            # ?download=<name> saves the file instead of streaming it
            name = os.path.basename(q.get("download") or "")
            return self.send_file(path, download_name=name or None, cache=not name)
        if len(parts) == 2 and parts[0] == "voice" and safe_id(parts[1]):
            return self.send_file(os.path.join(vdir(parts[1]), "ref.wav"))
        if len(parts) == 2 and parts[0] == "upload" and safe_id(parts[1]):
            return self.send_file(os.path.join(DATA, "uploads", f"{parts[1]}.wav"))
        if len(parts) == 3 and parts[0] == "export" and safe_id(parts[1]):
            name = os.path.basename(parts[2])
            return self.send_file(os.path.join(pdir(parts[1]), "exports", name), download_name=name)
        raise ApiError(404, "not_found")

    def api(self, method, parts, q):
        head = parts[0] if parts else ""
        if (method, head) == ("GET", "config"):
            return self.send_json({
                "device_name": CFG["device_name"], "home_url": CFG["home_url"],
                "idle_unload_minutes": IDLE_UNLOAD // 60, "defaults": DEFAULT_SETTINGS,
                "features": {"transcribe": bool(CFG["transcribe_python"] and os.path.exists(CFG["transcribe_python"]))},
            })
        if (method, head) == ("GET", "status"):
            return self.send_json(ENGINE.status())
        if (method, head) == ("POST", "unload"):
            ENGINE.request_unload()
            return self.send_json({"ok": True})
        if head == "voices":
            return self.api_voices(method, parts[1:])
        if (method, head) == ("POST", "upload"):
            return self.api_upload()
        if (method, head) == ("POST", "transcribe"):
            b = self.body_json()
            uid = b.get("upload_id", "")
            path = os.path.join(DATA, "uploads", f"{uid}.wav")
            if not safe_id(uid) or not os.path.exists(path):
                raise ApiError(404, "upload_missing")
            lang = b.get("lang") if b.get("lang") in VOICE_LANGS else ""
            return self.send_json({"text": transcribe(path, lang)})
        if head == "projects":
            return self.api_projects(method, parts[1:])
        if head == "jobs" and method == "DELETE" and len(parts) == 2:
            return self.send_json({"ok": ENGINE.cancel(parts[1])})
        raise ApiError(404, "not_found")

    def api_upload(self):
        data = self.body_bytes()
        if len(data) < 1000:
            raise ApiError(400, "empty_file")
        up = os.path.join(DATA, "uploads")
        os.makedirs(up, exist_ok=True)
        for fn in os.listdir(up):                      # drop uploads older than a day
            fp = os.path.join(up, fn)
            if now() - os.path.getmtime(fp) > 86400:
                os.remove(fp)
        uid = new_id()
        src, out = os.path.join(up, f"{uid}.src"), os.path.join(up, f"{uid}.wav")
        with open(src, "wb") as f:
            f.write(data)
        try:
            ffmpeg(["-i", src, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", out])
        except RuntimeError:
            raise ApiError(400, "bad_audio")
        finally:
            os.remove(src)
        return self.send_json({"upload_id": uid, "duration": round(sf.info(out).duration, 2)})

    def api_voices(self, method, parts):
        if method == "GET" and not parts:
            return self.send_json(list_voices())
        if method == "POST" and not parts:
            b = self.body_json()
            meta = clean_voice_fields(b, {"gender": "", "lang": "zh"})
            if "name" not in meta:
                raise ApiError(400, "need_name")
            if "ref_text" not in meta:
                raise ApiError(400, "need_ref_text")
            vid = new_id()
            dst = os.path.join(vdir(vid), "ref.wav")
            if b.get("upload_id"):
                src = os.path.join(DATA, "uploads", f"{b['upload_id']}.wav")
                if not safe_id(b["upload_id"]) or not os.path.exists(src):
                    raise ApiError(404, "upload_missing")
                source, title = "upload", ""
            elif b.get("pid") and b.get("tid"):
                p = load_project(b["pid"])
                if not p:
                    raise ApiError(404, "script_missing")
                src = render_take(p["id"], b["tid"], *post_params({}, p["settings"]))
                if not src:
                    raise ApiError(404, "take_missing")
                source, title = "take", p["title"]
            else:
                raise ApiError(400, "no_reference")
            os.makedirs(vdir(vid))
            shutil.copy(src, dst)
            meta.update({"id": vid, "created": now(), "source": source, "source_title": title[:30],
                         "duration": round(sf.info(dst).duration, 2)})
            write_json(os.path.join(vdir(vid), "meta.json"), meta)
            return self.send_json({"id": f"custom:{vid}"})
        if len(parts) == 1 and parts[0].startswith("custom:"):
            vid = parts[0].split(":", 1)[1]
            meta = custom_meta(vid)
            if not meta:
                raise ApiError(404, "voice_missing")
            if method == "PUT":
                write_json(os.path.join(vdir(vid), "meta.json"), clean_voice_fields(self.body_json(), meta))
                return self.send_json({"ok": True})
            if method == "DELETE":
                shutil.rmtree(vdir(vid))
                return self.send_json({"ok": True})
        raise ApiError(404, "not_found")

    def api_projects(self, method, parts):
        if not parts:
            if method == "GET":
                return self.send_json(list_projects())
            if method == "POST":
                b = self.body_json()
                paras = split_paragraphs(b.get("text") or "")
                title = (b.get("title") or "").strip()[:60] or (paras[0][:16] if paras else "untitled")
                p = {"id": new_id(), "title": title, "created": now(), "updated": now(),
                     "settings": clean_settings(b.get("settings"), DEFAULT_SETTINGS),
                     "paragraphs": [{"id": new_id(8), "text": t, "takes": [], "chosen": None} for t in paras]
                     or [{"id": new_id(8), "text": "", "takes": [], "chosen": None}]}
                with io_lock:
                    save_project(p)
                return self.send_json(p)
            raise ApiError(405, "not_allowed")

        pid, rest = parts[0], parts[1:]
        with io_lock:
            p = load_project(pid)
        if not p:
            raise ApiError(404, "script_missing")

        if method == "GET" and not rest:
            return self.send_json(p)
        if method == "DELETE" and not rest:
            for jid, j in list(ENGINE.jobs.items()):
                if j["pid"] == pid:
                    ENGINE.cancel(jid)
            shutil.rmtree(pdir(pid), ignore_errors=True)
            return self.send_json({"ok": True})
        if method == "PUT" and not rest:
            return self.update_project(pid, self.body_json())
        if method == "POST" and rest == ["generate"]:
            b = self.body_json()
            by_id = {x["id"]: x for x in p["paragraphs"]}
            jobs = []
            for para_id in b.get("para_ids") or []:
                para = by_id.get(para_id)
                if para and para["text"].strip():
                    jobs.append(ENGINE.public(ENGINE.submit(pid, para_id, para["text"].strip(), p["settings"])))
            if not jobs:
                raise ApiError(400, "empty_paragraph")
            return self.send_json({"jobs": jobs})
        if method == "POST" and rest == ["choose"]:
            b = self.body_json()
            with io_lock:
                p = load_project(pid)
                para = next((x for x in p["paragraphs"] if x["id"] == b.get("para_id")), None)
                if not para or (b.get("tid") and not any(t["id"] == b["tid"] for t in para["takes"])):
                    raise ApiError(404, "take_missing")
                para["chosen"] = b.get("tid")
                save_project(p)
            return self.send_json(p)
        if method == "DELETE" and len(rest) == 2 and rest[0] == "takes":
            tid = rest[1]
            with io_lock:
                p = load_project(pid)
                for para in p["paragraphs"]:
                    if any(t["id"] == tid for t in para["takes"]):
                        para["takes"] = [t for t in para["takes"] if t["id"] != tid]
                        if para["chosen"] == tid:
                            para["chosen"] = para["takes"][-1]["id"] if para["takes"] else None
                        remove_take_files(pid, tid)
                save_project(p)
            return self.send_json(p)
        if method == "POST" and rest == ["export"]:
            fmt = self.body_json().get("format")
            name = export_project(p, fmt if fmt in ("wav", "mp3", "zip") else "wav")
            return self.send_json({"name": name, "url": f"/media/export/{pid}/{quote(name)}"})
        raise ApiError(404, "not_found")

    def update_project(self, pid, b):
        with io_lock:
            p = load_project(pid)
            if "title" in b:
                p["title"] = str(b["title"] or "").strip()[:60] or p["title"]
            if isinstance(b.get("settings"), dict):
                p["settings"] = clean_settings(b["settings"], p["settings"])
            if isinstance(b.get("paragraphs"), list):
                old = {x["id"]: x for x in p["paragraphs"]}
                new = []
                for item in b["paragraphs"]:
                    text = str(item.get("text", ""))[:5000]
                    o = old.pop(item.get("id"), None)
                    if o:
                        o["text"] = text
                        new.append(o)
                    else:
                        new.append({"id": new_id(8), "text": text, "takes": [], "chosen": None})
                for gone in old.values():
                    for t in gone["takes"]:
                        remove_take_files(pid, t["id"])
                p["paragraphs"] = new
            save_project(p)
        return self.send_json(p)


def main():
    global ENGINE
    for d in ("projects", "voices", "uploads"):
        os.makedirs(os.path.join(DATA, d), exist_ok=True)
    ENGINE = Engine()
    srv = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler)
    srv.daemon_threads = True
    log(f"ttsd listening on http://{CFG['host']}:{CFG['port']}  (Python {sys.version.split()[0]})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
