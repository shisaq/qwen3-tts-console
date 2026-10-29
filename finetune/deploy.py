"""Put a fine-tuned epoch into the console: MLX 8-bit model → <models_dir>/tuned/, entry in tuned_voices.

Run with the console's venv (it has mlx-audio, needed only if the epoch wasn't converted yet):
  ../.venv/bin/python deploy.py <speaker> <epoch> [--name "显示名"] [--gender m|f] [--lang zh] [--restart <launchd label>]

Reads the console config from $TTSD_CONFIG or ~/.config/ttsd/config.json (same rules as ttsd.py),
backs it up next to itself, and never touches other keys. The speaker id is taken from the checkpoint.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from common import add_config_arg, load_config

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def console_config():
    path = os.environ.get("TTSD_CONFIG") or os.path.expanduser("~/.config/ttsd/config.json")
    cfg = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    models = os.path.expanduser(cfg.get("models_dir", "models"))
    return path, cfg, models if os.path.isabs(models) else os.path.join(APP_DIR, models)


def speaker_id(model_dir):
    spk = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))["talker_config"].get("spk_id", {})
    if len(spk) != 1:
        sys.exit(f"{model_dir}: expected one speaker in spk_id, found {list(spk)}")
    return next(iter(spk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    ap.add_argument("epoch", type=int)
    add_config_arg(ap)
    ap.add_argument("--name", help="name shown in the console (default: the speaker key)")
    ap.add_argument("--gender", default="", choices=["", "m", "f"])
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--q_bits", type=int, default=8)
    ap.add_argument("--restart", metavar="LABEL", help="launchd label of ttsd to restart, e.g. com.example.ttsd")
    a = ap.parse_args()

    d = os.path.join(load_config(a.config)["work_dir"], a.speaker)
    src = os.path.join(d, "mlx", f"epoch-{a.epoch}")                 # made by gen_samples.py
    if not os.path.exists(os.path.join(src, "config.json")):
        ckpt = os.path.join(d, "ckpt", f"epoch-{a.epoch}")
        if not os.path.exists(os.path.join(ckpt, "config.json")):
            sys.exit(f"no checkpoint {ckpt} (train with --save_epochs including {a.epoch})")
        print(f"converting {ckpt} → MLX {a.q_bits}-bit …", flush=True)
        subprocess.run([sys.executable, "-m", "mlx_audio.convert", "--hf-path", ckpt, "--mlx-path", src,
                        "-q", "--q-bits", str(a.q_bits)], check=True, stdout=subprocess.DEVNULL)

    cfg_path, cfg, models_dir = console_config()
    rel = os.path.join("tuned", f"{a.speaker}-e{a.epoch}-{a.q_bits}bit")
    dst = os.path.join(models_dir, rel)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    if os.path.exists(cfg_path):
        shutil.copy2(cfg_path, f"{cfg_path}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    old = cfg.setdefault("tuned_voices", {}).get(a.speaker, {})
    cfg["tuned_voices"][a.speaker] = {
        "name": a.name or old.get("name") or a.speaker, "model": rel, "speaker": speaker_id(dst),
        "gender": a.gender or old.get("gender", ""), "lang": a.lang}
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"model  → {dst}\nconfig → {cfg_path} (tuned_voices.{a.speaker})")

    if a.restart:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{a.restart}"], check=True)
        print(f"restarted {a.restart}")
    else:
        print("restart ttsd to load it (or pass --restart <launchd label>)")


if __name__ == "__main__":
    main()
