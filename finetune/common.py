"""Shared config for the fine-tuning scripts.

All scripts read one JSON file (default: ./voiceprep.json, or --config). See voiceprep.example.json.
Speaker keys double as folder names and as the speaker id inside the trained model, so use ASCII.
"""
import json
import os


def expand(p):
    return os.path.abspath(os.path.expanduser(p))


def load_config(path):
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["work_dir"] = expand(cfg.get("work_dir", "~/Datasets/voiceprep"))
    for key, s in cfg["speakers"].items():
        if not key.isascii() or not key.replace("_", "").isalnum():
            raise ValueError(f"speaker key {key!r}: use ASCII letters, digits and _ (it becomes the model's speaker id)")
        s["ref_audio"] = expand(s["ref_audio"])
        s["inputs"] = [expand(p) for p in s.get("inputs", [])]
    cfg.setdefault("terms", {})
    return cfg


def add_config_arg(ap):
    ap.add_argument("--config", default="voiceprep.json", help="pipeline config (default: ./voiceprep.json)")
