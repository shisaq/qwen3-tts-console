"""Reviewed clips → training jsonl with speech-tokenizer codes.

Takes <work_dir>/<speaker>/manifest.jsonl + edits.json, keeps status == "keep",
and encodes every clip with Qwen3-TTS-Tokenizer-12Hz (same step as the official prepare_data.py).
Every sample gets the same reference clip, as the official guide recommends.

Usage:  .venv-train/bin/python build_train.py <speaker> [--config voiceprep.json]
"""
import argparse
import json
import os

import torch
from qwen_tts import Qwen3TTSTokenizer

from common import add_config_arg, load_config

BATCH = 16


def reviewed_rows(work, speaker):
    d = os.path.join(work, speaker)
    rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"), encoding="utf-8")]
    ep = os.path.join(d, "edits.json")
    edits = json.load(open(ep, encoding="utf-8")) if os.path.exists(ep) else {}
    for r in rows:
        r.update(edits.get(r["id"], {}))
    return [r for r in rows if r["status"] == "keep" and r["text"].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    add_config_arg(ap)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-TTS-Tokenizer-12Hz", help="HF id or local folder")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work, ref = cfg["work_dir"], cfg["speakers"][a.speaker]["ref_audio"]
    rows = reviewed_rows(work, a.speaker)
    print(f"{a.speaker}: {len(rows)} clips, {sum(r['dur'] for r in rows) / 60:.1f} min, ref {ref}")

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=device)
    out = []
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok.encode([os.path.join(work, r["audio"]) for r in batch])
        for r, codes in zip(batch, enc.audio_codes):
            out.append({"audio": os.path.join(work, r["audio"]), "text": r["text"].strip(),
                        "ref_audio": ref, "audio_codes": codes.cpu().tolist()})
        print(f"  encoded {len(out)}/{len(rows)}", flush=True)
    path = os.path.join(work, a.speaker, "train_with_codes.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
