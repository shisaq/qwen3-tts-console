# Fine-tune a voice on your Mac

Turn a few videos or recordings of one person into a fine-tuned Qwen3-TTS voice, entirely on an Apple Silicon Mac,
then use it in the console like any other voice. Also runs on CUDA.

[中文说明](README.zh-CN.md)

## What it does

```
recordings ─ prep_data.py ─▶ 2–10 s clips + transcripts + "is this the right person?" check
           ─ review.py    ─▶ you listen, fix wrong words, drop bad clips   (web page)
           ─ build_train.py ─▶ speech-tokenizer codes
           ─ sft.py       ─▶ checkpoints (Qwen3-TTS-12Hz-1.7B-Base → custom_voice with one speaker)
           ─ gen_samples.py + score_samples.py ─▶ every epoch vs. zero-shot cloning, side by side
           ─ tuned_voices in the console config ─▶ done
```

Speech recognition is FunASR (Paraformer-zh with VAD, punctuation and CAM++ speaker embeddings), not Whisper:
it handles Chinese well, gives per-character timestamps, and the CAM++ check removes interviewers and other voices.
Mixed Chinese with English terms works; recordings that are mostly English are dropped.

## Measured on a Mac mini (M5 Pro, 64 GB)

| | |
|---|---|
| data prep, ~90 min of recordings | ~13 min |
| one speaker after review | 165 clips, 18 min |
| training, 10 epochs | 17 min, ~38 GB unified memory |
| speaker similarity on unseen sentences (CAM++ vs. the speaker's real clips) | zero-shot clone 0.72 → fine-tuned 0.84 |
| character error rate | < 1 % for every epoch |
| in the console | 4.9 s of audio in 1.7 s, model load 0.5 s, ~3 GB |

Similarity is measured against the training recordings, so part of the gain can be the recording conditions
(microphone, room) rather than the voice itself — listen before you pick. Training loss keeps falling to ~0.16
by epoch 10; the best-sounding checkpoints were epochs 6–10, chosen by ear.

The fine-tuned voice does **not** follow mood/style instructions: the Base model was never trained on them.

## Setup

```bash
cd finetune
uv venv --python 3.12 .venv       && VIRTUAL_ENV=.venv       uv pip install -r requirements-prep.txt
uv venv --python 3.12 .venv-train && VIRTUAL_ENV=.venv-train uv pip install -r requirements-train.txt
cp voiceprep.example.json voiceprep.json     # then edit it
```

`voiceprep.json` (git-ignored) lists each speaker: a short clean reference clip with its exact transcript, and the
files or folders to mine. A recording with two of your speakers (an interview) can be listed under both; each clip
goes to whoever it sounds like. Speaker keys must be ASCII — they become the speaker id inside the model.
`terms` fixes words the recogniser gets wrong (product names, English terms).

Tested with funasr 1.4.16, modelscope 1.40.2, qwen-tts 0.1.1, torch 2.14.1, mlx-audio as pinned in `../requirements.txt`.

## Run

```bash
.venv/bin/python prep_data.py                                   # all speakers
.venv/bin/python review.py                                      # http://127.0.0.1:18410  (no auth: keep it local)
.venv-train/bin/python build_train.py alice
.venv-train/bin/python sft.py --train_jsonl ~/Datasets/voiceprep/alice/train_with_codes.jsonl \
    --speaker_name alice --output_dir ~/Datasets/voiceprep/alice/ckpt --epochs 10 --save_epochs 2,4,6,8,10
../.venv/bin/python gen_samples.py alice                       # converts each epoch to MLX 8-bit
.venv/bin/python score_samples.py alice                         # then listen: http://127.0.0.1:18410/compare/alice
```

Review page keys: ↑↓ select · space play · K keep · D drop · Enter edit text. Edits are saved to `edits.json`
and survive re-running `prep_data.py`. Clips are flagged for English words, fillers, repeats and spelled-out numbers.

Then move the epoch you like into the console's models folder and add it to `~/.config/ttsd/config.json`:

```json
"tuned_voices": {
  "alice": {"name": "Alice", "model": "tuned/alice-e8-8bit", "speaker": "alice", "gender": "f", "lang": "zh"}
}
```

`speaker` must match `--speaker_name`. Restart ttsd; the voice shows up under "My voices".

## Fixes over the official `finetuning/sft_12hz.py`

Measured as the loss of the *untrained* Base model on real clips. A correct setup scores low, because the
pretrained model already predicts real speech well.

| | talker loss at init | sub-talker loss at init |
|---|---|---|
| official script | 11.9 | 9.1 |
| + labels shifted once ([#179](https://github.com/QwenLM/Qwen3-TTS/issues/179)) | 7.9 | – |
| + `text_projection` on text embeddings | **1.2** | – |
| + sub-talker gets the previous frame's hidden state, unshifted loss | – | **5.4** |

1. **Labels shifted twice** (#179): the talker's HF loss already shifts. The second shift teaches the model to skip
   frames, so speech gets faster with every epoch.
2. **Missing `text_projection`**: inference passes every text embedding through `talker.text_projection`, the
   official training loop doesn't. Shapes match (2048), so nothing errors; the model trains on inputs it never sees.
3. **Sub-talker double shift**: `forward_finetune` returns logits already aligned with `codec_ids[:, 1:]`, then its
   HF loss shifts them again. The hidden state paired with a frame must also be the one that predicted it (t−1).
4. **bf16 master weights**: at lr 2e-6 most AdamW updates are below bf16 resolution. We keep fp32 weights + bf16 autocast.

Also: the 311M-parameter text embedding is frozen by default (`--train_text_embedding` to train it). With it
trainable, a 64 GB Mac went into swap at the first checkpoint save and training slowed about 12×.

Data prep works around a FunASR 1.4.16 bug: `sentence_info` puts every punctuation mark one token late
("开发者大。家好很，高兴"), which cuts clips mid-word. `split_sentences` rebuilds sentences from the full text and
per-token timestamps.

Generate with language `auto`: training uses the no-language prefix, so a language tag switches to a prefix the
speaker never saw (the console does this for you).
