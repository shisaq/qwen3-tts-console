"""Score the samples from gen_samples.py: speaker similarity (CAM++, against the mean of real clips),
character error rate (Paraformer transcript vs. input) and speaking rate vs. the real speaker.

  .venv/bin/python score_samples.py <speaker> [--config voiceprep.json]   → adds "scores" to samples/index.json
"""
import argparse
import json
import os
import random
import re

import librosa
import numpy as np
from funasr import AutoModel

from common import add_config_arg, load_config

norm = lambda t: re.sub(r"[^\w]", "", t).lower()


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    add_config_arg(ap)
    a = ap.parse_args()
    spk = a.speaker
    d = os.path.join(load_config(a.config)["work_dir"], spk)
    idx = json.load(open(os.path.join(d, "samples", "index.json"), encoding="utf-8"))
    kw = dict(disable_update=True, log_level="ERROR")
    sv, asr = AutoModel(model="cam++", **kw), AutoModel(model="paraformer-zh", punc_model="ct-punc", **kw)
    load = lambda p: librosa.load(p, sr=16000)[0]
    emb = lambda a: (lambda e: e / np.linalg.norm(e))(np.asarray(sv.generate(input=a)[0]["spk_embedding"]).ravel())

    train = [json.loads(l) for l in open(os.path.join(d, "train_with_codes.jsonl"), encoding="utf-8")]
    random.seed(0)
    real = random.sample(train, min(40, len(train)))
    centroid = np.mean([emb(load(r["audio"])) for r in real], 0)
    centroid /= np.linalg.norm(centroid)
    real_rate = sum(len(norm(r["text"])) for r in train) / sum(len(load(r["audio"])) / 16000 for r in train)

    scores = {}
    for m in idx["models"]:
        rows = []
        for i, text in enumerate(idx["sentences"]):
            for s in idx["seeds"]:
                a = load(os.path.join(d, "samples", m, f"s{i}_{s}.wav"))
                hyp = asr.generate(input=a)[0]["text"]
                rows.append({"s": i, "seed": s, "sim": float(emb(a) @ centroid), "cer": cer(text, hyp),
                             "rate": len(norm(text)) / (len(a) / 16000), "hyp": hyp})
        scores[m] = rows
        print(f"{m:10s} sim {np.mean([r['sim'] for r in rows]):.3f}  cer {np.mean([r['cer'] for r in rows]):.3f}  "
              f"rate {np.mean([r['rate'] for r in rows]):.2f} chars/s (real {real_rate:.2f})", flush=True)
    idx["scores"], idx["real_rate"] = scores, real_rate
    json.dump(idx, open(os.path.join(d, "samples", "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
