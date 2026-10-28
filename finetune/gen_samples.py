"""Generate comparison samples: each fine-tuned epoch (converted to MLX 8-bit, as ttsd runs it) plus a
zero-shot clone from the speaker's reference clip, on sentences that are not in the training set.

Run with the ttsd venv (mlx-audio):
  ../.venv/bin/python gen_samples.py <speaker> [--config voiceprep.json] [--sentences my_lines.txt]
Writes <work_dir>/<speaker>/samples/<model>/s<i>_<seed>.wav and samples/index.json;
MLX conversions go to <work_dir>/<speaker>/mlx/epoch-N.
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

from common import add_config_arg, load_config

SENTENCES = [
    "说实话，我第一次看到这个结果的时候，也挺意外的。",
    "今天我们正式发布新一代的产品，它可以帮大家把重复的工作交给机器来完成。",
    "很多人问我，AI Agent 和传统的自动化工具到底有什么区别？我的回答是：一个靠规则，一个靠理解。",
    "这件事情，其实没有大家想象的那么复杂。",
    "坚持了七年，我们终于等到了这个时刻。谢谢每一位一路同行的伙伴！",
]
SEEDS = [11, 22]
GEN = dict(temperature=0.65, top_k=50, top_p=1.0, repetition_penalty=1.05, split_pattern="")


def convert(src, dst, bits):
    if not os.path.exists(os.path.join(dst, "config.json")):
        subprocess.run([sys.executable, "-m", "mlx_audio.convert", "--hf-path", src, "--mlx-path", dst,
                        "-q", "--q-bits", str(bits)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(model, out_dir, sentences, **kw):
    m = load_model(model)
    os.makedirs(out_dir, exist_ok=True)
    for i, text in enumerate(sentences):
        for s in SEEDS:
            f = os.path.join(out_dir, f"s{i}_{s}.wav")
            if os.path.exists(f):
                continue
            mx.random.seed(s)
            a = np.concatenate([np.array(r.audio, np.float32) for r in m.generate(text=text, **GEN, **kw)])
            sf.write(f, a, 24000, subtype="PCM_16")
    del m
    gc.collect()
    mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    add_config_arg(ap)
    ap.add_argument("--sentences", help="text file, one test sentence per line (default: built-in Chinese set)")
    ap.add_argument("--base_model", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    help="MLX Base model for the zero-shot baseline")
    ap.add_argument("--q_bits", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    spk = cfg["speakers"][a.speaker]
    d = os.path.join(cfg["work_dir"], a.speaker)
    out = os.path.join(d, "samples")
    sentences = [l.strip() for l in open(a.sentences, encoding="utf-8") if l.strip()] if a.sentences else SENTENCES

    run(a.base_model, os.path.join(out, "zero-shot"), sentences,
        ref_audio=spk["ref_audio"], ref_text=spk["ref_text"], lang_code=spk.get("lang_code", "chinese"))
    print("zero-shot done", flush=True)
    models = ["zero-shot"]
    for ck in sorted(glob.glob(os.path.join(d, "ckpt", "epoch-*")), key=lambda p: int(p.rsplit("-", 1)[1])):
        ep = os.path.basename(ck)
        mlx_dir = os.path.join(d, "mlx", ep)
        convert(ck, mlx_dir, a.q_bits)
        # fine-tuned with the no-language prefix: generate with "auto"
        run(mlx_dir, os.path.join(out, ep), sentences, voice=a.speaker, lang_code="auto")
        models.append(ep)
        print(ep, "done", flush=True)
    json.dump({"sentences": sentences, "seeds": SEEDS, "models": models},
              open(os.path.join(out, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
