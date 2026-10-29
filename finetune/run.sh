#!/usr/bin/env bash
# One entry point for the whole pipeline. Paths are relative to this folder; the config is
# ./voiceprep.json unless CONFIG=... is set.
#
#   ./run.sh prep                      prepare clips for every speaker, then open the review page
#   ./run.sh review                    (re)open the review / compare page  → http://127.0.0.1:18410
#   ./run.sh train <speaker> [opts]    encode → train → samples → scores; opts go to sft.py
#                                      (e.g. --epochs 6 --batch_size 1 --grad_accum 8 --freeze_layers 20)
#   ./run.sh deploy <speaker> <epoch> [--name 显示名 --gender m --restart <launchd label>]
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=${CONFIG:-voiceprep.json}
# interpreters; override with env vars if your venvs live elsewhere
PREP=${PREP:-.venv/bin/python}          # funasr + torch
TRAIN=${TRAIN:-.venv-train/bin/python}  # qwen-tts + torch
MLX=${MLX:-../.venv/bin/python}         # the console's venv (mlx-audio)

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
need() { [ -x "$1" ] || { echo "missing $1 — see README (Setup)"; exit 1; }; }
case "${1:-}" in prep|review|train|deploy) ;; *) usage ;; esac
[ -f "$CONFIG" ] || { echo "missing $CONFIG — cp voiceprep.example.json voiceprep.json and edit it"; exit 1; }
work_dir() { python3 -c "import json,os,sys; print(os.path.expanduser(json.load(open(sys.argv[1])).get('work_dir','~/Datasets/voiceprep')))" "$CONFIG"; }

case "${1:-}" in
  prep)
    need $PREP
    $PREP prep_data.py --config "$CONFIG"
    echo "review the clips, then: ./run.sh train <speaker>"
    exec $PREP review.py --config "$CONFIG" ;;
  review)
    need $PREP
    exec $PREP review.py --config "$CONFIG" ;;
  train)
    [ $# -ge 2 ] || usage
    spk=$2; shift 2
    need $PREP; need $TRAIN; need $MLX
    w="$(work_dir)/$spk"
    $TRAIN build_train.py "$spk" --config "$CONFIG"
    $TRAIN sft.py --train_jsonl "$w/train_with_codes.jsonl" --speaker_name "$spk" --output_dir "$w/ckpt" \
      --epochs 10 --save_epochs 2,4,6,8,10 "$@"
    $MLX gen_samples.py "$spk" --config "$CONFIG"
    $PREP score_samples.py "$spk" --config "$CONFIG"
    echo "listen and pick an epoch: ./run.sh review → http://127.0.0.1:18410/compare/$spk"
    echo "then: ./run.sh deploy $spk <epoch>" ;;
  deploy)
    shift
    need $MLX
    exec $MLX deploy.py --config "$CONFIG" "$@" ;;
  *) usage ;;
esac
