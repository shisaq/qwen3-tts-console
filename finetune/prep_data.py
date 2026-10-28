"""Turn long recordings into a Qwen3-TTS fine-tuning set.

    source audio → FunASR (Paraformer-zh + VAD + punctuation + CAM++ diarization)
                 → sentences merged into 2–10 s clips
                 → CAM++ speaker check against each speaker's reference clip
                 → 24 kHz mono clips + <work_dir>/<speaker>/manifest.jsonl

A recording listed under several speakers (an interview, a dialogue) is transcribed once and every
clip goes to whichever of those speakers it sounds like. ASR results are cached in <work_dir>/asr,
so re-running only redoes the cheap steps. Edits made in review.py live in edits.json and survive re-runs.

Usage:  .venv/bin/python prep_data.py [--config voiceprep.json]
"""
import argparse
import glob
import json
import os
import re
from difflib import SequenceMatcher

import librosa
import numpy as np
import soundfile as sf

from common import add_config_arg, load_config

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".mp4", ".mov")
SR = 24000
MIN_DUR, MAX_DUR = 2.0, 10.0
MAX_GAP = 0.5            # s of silence allowed inside one clip
KEEP_SIM, REVIEW_SIM = 0.55, 0.45   # CAM++ cosine: same person ~0.7, different people <0.4 in our tests
PAD_BEFORE, PAD_AFTER = 0.08, 0.15
TARGET_RMS_DB = -20.0    # level of the voiced part after normalisation

# Paraformer writes English in lower case. Only unambiguous fixes; add your own names in config "terms".
# Keys are matched as whole words, case-insensitively.
TERMS = {
    "ai": "AI", "gpt": "GPT", "api": "API", "cpu": "CPU", "gpu": "GPU", "ok": "OK", "app": "App",
    "agent": "Agent", "agents": "Agents", "open ai": "OpenAI", "iphone": "iPhone", "mac": "Mac",
}


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)


def frame_db(x, sr, hop=0.05):
    n = int(sr * hop)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


class Models:
    def __init__(self):
        from funasr import AutoModel
        kw = dict(disable_update=True, log_level="ERROR")
        self.asr = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
                             spk_model="cam++", **kw)
        self.sv = AutoModel(model="cam++", **kw)

    def embed(self, wav16):
        e = np.asarray(self.sv.generate(input=wav16)[0]["spk_embedding"], dtype=np.float32).ravel()
        return e / np.linalg.norm(e)


PUNC = "，。？！、,.?!；;：:"
TOKEN = re.compile(rf"[A-Za-z']+|[{PUNC}]|[^\sA-Za-z'{PUNC}]")


def transcribe(models, cache, wav16):
    if os.path.exists(cache):
        raw = json.load(open(cache, encoding="utf-8"))
    else:
        r = models.asr.generate(input=wav16, batch_size_s=300, sentence_timestamp=True)[0]
        raw = {"text": r["text"], "timestamp": r["timestamp"],
               "spk": [[s["start"], s["end"], s.get("spk", 0)] for s in r["sentence_info"]]}
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(raw, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return split_sentences(raw)


def split_sentences(raw):
    """Re-split the punctuated text using per-token timestamps.

    FunASR 1.4.16's own sentence_info places every punctuation mark one token late
    ("开发者大。家好很，高兴"), so its sentence boundaries cut words in half. The full
    text and the token timestamps are right, so rebuild the sentences from those.
    """
    toks, ts = TOKEN.findall(raw["text"]), raw["timestamp"]
    words = [t for t in toks if t not in PUNC]
    if len(words) != len(ts):
        raise ValueError(f"{len(words)} tokens vs {len(ts)} timestamps")
    sents, cur, k = [], [], 0
    for t in toks + ["。"]:
        if t in PUNC:
            if cur:
                text = "".join(w if not w.isascii() else f" {w} " for w, _ in cur)
                text = re.sub(r"\s+", " ", text).strip() + t
                sents.append({"start": cur[0][1][0] / 1000, "end": cur[-1][1][1] / 1000, "text": text})
                cur = []
            continue
        cur.append((t, ts[k]))
        k += 1
    for s in sents:              # speaker label: diarized span that overlaps this sentence most
        ov = lambda sp: min(s["end"], sp[1] / 1000) - max(s["start"], sp[0] / 1000)
        s["spk"] = max(raw["spk"], key=ov)[2] if raw["spk"] else 0
    return sents


def group(sents):
    """Merge consecutive same-speaker sentences into clips of MIN_DUR..MAX_DUR seconds."""
    clips, cur = [], []

    def flush():
        if cur and cur[-1]["end"] - cur[0]["start"] >= MIN_DUR:
            clips.append(list(cur))
        cur.clear()

    for s in sents:
        if cur:
            gap = s["start"] - cur[-1]["end"]
            if s["spk"] != cur[-1]["spk"] or gap > MAX_GAP or s["end"] - cur[0]["start"] > MAX_DUR:
                flush()
            elif cur[-1]["text"][-1:] in "。？！?!" and cur[-1]["end"] - cur[0]["start"] >= 4:
                flush()                        # prefer ending on a full stop once long enough
        if s["end"] - s["start"] <= MAX_DUR:
            cur.append(s)
        else:
            flush()
    flush()
    return clips


def term_fixer(extra):
    terms = {**TERMS, **{k.lower(): v for k, v in extra.items()}}
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, terms), key=len, reverse=True)) + r")(?![A-Za-z])", re.I)
    return lambda text: rx.sub(lambda m: terms[m.group(0).lower()], text)


def flags_of(text):
    f = []
    if re.search(r"[A-Za-z]", text):
        f.append("english")                    # Paraformer often misspells English words
    if re.search(r"[呃嗯额]", text):
        f.append("filler")
    if re.search(r"(.{2,4})\1", re.sub(r"[，。？！、,.?!\s]", "", text)):
        f.append("repeat")                     # stutter / self-correction
    if re.search(r"[一二三四五六七八九零〇]{3,}", text):
        f.append("number")                     # spelled-out digits are often wrong
    return f


def list_sources(cfg):
    sources = {}
    for name, s in cfg["speakers"].items():
        for p in s["inputs"]:
            files = sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(AUDIO_EXT)) \
                if os.path.isdir(p) else [p]
            for f in files:
                sources.setdefault(f, []).append(name)
    return sources


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    cfg = load_config(ap.parse_args().config)
    work, speakers = cfg["work_dir"], cfg["speakers"]
    fix_terms = term_fixer(cfg["terms"])
    models = Models()
    load16 = lambda p: librosa.load(p, sr=16000, mono=True)[0]
    refs = {name: models.embed(load16(s["ref_audio"])) for name, s in speakers.items()}

    out = {name: [] for name in speakers}
    sources = list_sources(cfg)
    root = os.path.commonpath([os.path.dirname(p) for p in sources])   # clip ids are relative to this
    for path, candidates in sources.items():
        wav16, wav = load16(path), librosa.load(path, sr=SR, mono=True)[0]
        floor = max(np.percentile(frame_db(wav, SR), 10), -90.0)   # digital silence would make SNR meaningless
        src = os.path.relpath(path, root)
        sents = transcribe(models, os.path.join(work, "asr", src.replace(os.sep, "__") + ".json"), wav16)
        bounds = [(s["start"], s["end"]) for s in sents]
        for ci, clip in enumerate(group(sents)):
            start, end = clip[0]["start"], clip[-1]["end"]
            i0, i1 = sents.index(clip[0]), sents.index(clip[-1])
            lo = max(start - PAD_BEFORE, (bounds[i0 - 1][1] + start) / 2 if i0 > 0 else 0)
            hi = min(end + PAD_AFTER, (bounds[i1 + 1][0] + end) / 2 if i1 + 1 < len(bounds) else len(wav) / SR)
            e = models.embed(wav16[int(start * 16000):int(end * 16000)])
            sims = {n: float(e @ refs[n]) for n in speakers}
            who = max(candidates, key=lambda n: sims[n])
            other = max([v for n, v in sims.items() if n != who] or [0])
            seg = wav[int(lo * SR):int(hi * SR)]
            fdb = frame_db(seg, SR)
            speech_db = np.percentile(fdb, 75)
            mask = np.repeat(fdb > floor + 10, int(SR * .05))     # voiced frames only
            level = db(seg[:len(mask)][mask]) if mask.any() else db(seg)
            seg = seg * min(10 ** ((TARGET_RMS_DB - level) / 20), 0.89 / (np.abs(seg).max() + 1e-9))
            text = fix_terms("".join(s["text"] for s in clip))
            sim = sims[who]
            status = "keep" if sim >= KEEP_SIM else "review" if sim >= REVIEW_SIM else "drop"
            flags = flags_of(text)
            if len(re.findall(r"[A-Za-z]", text)) > len(re.findall(r"[一-龥]", text)):
                status = "drop"                    # English speech: Paraformer-zh can't transcribe it
                flags.append("english_speech")
            cid = f"{re.sub(r'[^0-9A-Za-z一-龥]+', '_', os.path.splitext(src)[0])}_{ci:03d}"
            rel = f"{who}/clips/{cid}.wav"
            os.makedirs(os.path.join(work, who, "clips"), exist_ok=True)
            if status != "drop":
                sf.write(os.path.join(work, rel), seg, SR, subtype="PCM_16")
            out[who].append({
                "id": cid, "audio": rel, "text": text, "dur": round(hi - lo, 2), "src": src,
                "start": round(lo, 2), "end": round(hi, 2), "sim": round(sim, 3), "sim_other": round(other, 3),
                "snr": round(float(speech_db - floor), 1), "status": status, "flags": flags,
            })
        print(f"{src}: {len(sents)} sentences", flush=True)

    norm = lambda t: re.sub(r"[，。？！、,.?!\s]", "", t)
    for who, rows in out.items():
        # overlapping recordings (two edits of one event): keep the cleaner take of each sentence
        live = sorted([r for r in rows if r["status"] != "drop"], key=lambda r: -r["snr"])
        for i, r in enumerate(live):
            for q in live[:i]:
                if q["src"] != r["src"] and q["status"] != "dup" and \
                        SequenceMatcher(None, norm(q["text"]), norm(r["text"])).ratio() > 0.8:
                    r["status"], r["dup_of"] = "dup", q["id"]
                    break
        os.makedirs(os.path.join(work, who), exist_ok=True)
        with open(os.path.join(work, who, "manifest.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mins = lambda s: sum(r["dur"] for r in rows if r["status"] == s) / 60
        print(f"{who}: keep {mins('keep'):.1f} min | review {mins('review'):.1f} | dup {mins('dup'):.1f} | "
              f"drop {mins('drop'):.1f}  ({len(rows)} clips)")


if __name__ == "__main__":
    main()
