"""Single-speaker fine-tuning of Qwen3-TTS-12Hz Base → a custom_voice checkpoint.

Based on the official finetuning/sft_12hz.py, with these changes:
  * label alignment fixed (QwenLM/Qwen3-TTS#179): the talker's HF loss already shifts labels, the
    official script shifted them a second time, so the model learned to skip frames and spoke faster
    with every epoch. The sub-talker now pairs each frame's codes with the hidden state that predicts it,
    and its loss is computed unshifted (its built-in loss shifts already-aligned logits once more).
  * text embeddings go through talker.text_projection, as they do at inference. The official script
    feeds raw embeddings; at init that gives a talker loss of ~7.9 (near chance) instead of ~1.2.
  * fp32 master weights + bf16 autocast. The official script trains bf16 weights directly; at lr 2e-6
    most AdamW updates are smaller than bf16's resolution and get rounded away.
  * runs on CUDA, Apple MPS or CPU (sdpa attention, no flash-attn needed); plain loop, no accelerate.
  * speaker embedding = mean over many training clips instead of a single reference clip.
  * warmup + cosine LR, --max_steps for smoke tests, checkpoints saved in bf16 for selected epochs.
  * text embedding frozen by default (311M params, ~4 GB of grads + optimizer state; on a 64 GB Mac the
    full model plus a checkpoint save pushed the system into swap and training slowed ~12x).

Usage:
  .venv-train/bin/python sft.py --train_jsonl <work_dir>/<speaker>/train_with_codes.jsonl --speaker_name <speaker> \
      --output_dir <work_dir>/<speaker>/ckpt --epochs 10 --save_epochs 2,4,6,8,10
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig

from qwen_dataset import TTSDataset

SPK_SLOT = 3000   # codec_embedding row that holds the new speaker (same as the official script)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def speaker_embedding(model, dataset, n, device):
    """Mean speaker embedding over n random training clips (more stable than one reference)."""
    idx = random.sample(range(len(dataset.data_list)), min(n, len(dataset.data_list)))
    embs = []
    for i in idx:
        wav, sr = dataset._normalize_audio_inputs([dataset.data_list[i]["audio"]])[0]
        mel = dataset.extract_mels(audio=wav, sr=sr).to(device, next(model.speaker_encoder.parameters()).dtype)
        embs.append(model.speaker_encoder(mel).float())
    return torch.cat(embs).mean(0, keepdim=True)


def save_checkpoint(model, init_path, out_dir, speaker_name, spk_emb):
    os.makedirs(out_dir, exist_ok=True)
    # everything except the top-level weights (speech_tokenizer/ has its own model.safetensors)
    shutil.copytree(init_path, out_dir, dirs_exist_ok=True,
                    ignore=lambda d, names: ["model.safetensors"] if os.path.samefile(d, init_path) else [])
    cfg_path = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["tts_model_type"] = "custom_voice"
    cfg["talker_config"]["spk_id"] = {speaker_name: SPK_SLOT}
    cfg["talker_config"]["spk_is_dialect"] = {speaker_name: False}
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    with safe_open(os.path.join(init_path, "model.safetensors"), "pt") as f:
        keys = {k for k in f.keys() if not k.startswith("speaker_encoder")}
    # tensor by tensor to the CPU first: converting on the GPU would allocate a second 3.8 GB copy there
    state = {k: v.detach().to("cpu").to(torch.bfloat16).contiguous()
             for k, v in model.state_dict().items() if k in keys}
    state["talker.model.codec_embedding.weight"][SPK_SLOT] = spk_emb[0].to("cpu", torch.bfloat16)
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    del state
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="HF id or local folder")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--speaker_name", required=True)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_epochs", default="", help="comma list, e.g. 2,4,6; empty = every epoch")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many micro-steps (smoke test)")
    ap.add_argument("--spk_clips", type=int, default=64)
    ap.add_argument("--train_text_embedding", action="store_true",
                    help="also train the 311M-param text embedding (frozen by default: saves ~4 GB, few tokens are used)")
    ap.add_argument("--freeze_layers", type=int, default=0,
                    help="also freeze the first N of the talker's 28 layers (~50M params each) to save memory")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if not os.path.isdir(a.init_model_path):          # checkpoints copy files from a local folder
        from huggingface_hub import snapshot_download
        a.init_model_path = snapshot_download(a.init_model_path)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = pick_device()
    print(f"device {device}", flush=True)

    tts = Qwen3TTSModel.from_pretrained(a.init_model_path, torch_dtype=torch.float32, attn_implementation="sdpa")
    model = tts.model.to(device)
    config = AutoConfig.from_pretrained(a.init_model_path)
    data = [json.loads(l) for l in open(a.train_jsonl, encoding="utf-8")]
    dataset = TTSDataset(data, tts.processor, config)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model.speaker_encoder.eval()
    spk_emb = speaker_embedding(model, dataset, a.spk_clips, device)
    frozen = ("speaker_encoder",) + (() if a.train_text_embedding else ("talker.model.text_embedding",)) \
        + tuple(f"talker.model.layers.{i}." for i in range(a.freeze_layers))
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(frozen))
    params = [p for p in model.parameters() if p.requires_grad]
    shown = frozen[:2] + ((f"talker layers 0-{a.freeze_layers - 1}",) if a.freeze_layers else ())
    print(f"training {sum(p.numel() for p in params) / 1e6:.0f}M params (frozen: {', '.join(shown)})", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total = math.ceil(len(loader) / a.grad_accum) * a.epochs
    warm = max(1, int(total * a.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    save_epochs = {int(x) for x in a.save_epochs.split(",") if x} or set(range(1, a.epochs + 1))
    os.makedirs(a.output_dir, exist_ok=True)
    log = open(os.path.join(a.output_dir, "loss.csv"), "a")
    print(f"{len(dataset)} clips, {len(loader)} micro-steps/epoch, {total} optimizer steps", flush=True)

    model.train()
    model.speaker_encoder.eval()
    step = micro = 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                codec_mask = batch["codec_mask"]
                # inference always passes text through text_projection; the official script skips it
                text_emb = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0]))
                text_emb = text_emb * batch["text_embedding_mask"]
                codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"]
                codec_emb[:, 6, :] = spk_emb.to(codec_emb.dtype)
                emb = text_emb + codec_emb
                for i in range(1, 16):
                    e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    emb = emb + e * codec_mask.unsqueeze(-1)

                # labels unshifted: the talker's loss shifts them itself (#179)
                out = model.talker(inputs_embeds=emb, attention_mask=batch["attention_mask"],
                                   labels=batch["codec_0_labels"], output_hidden_states=True)
                hidden = out.hidden_states[0][-1]
                target = codec_mask[:, 1:]          # hidden state at t predicts the frame at t+1
                sub_ids = codec_ids[:, 1:][target]
                sub_logits, _ = model.talker.forward_sub_talker_finetune(sub_ids, hidden[:, :-1][target])
                # sub_logits[:, j] already predicts code j+1; its built-in loss would shift once more
                sub_loss = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]), sub_ids[:, 1:].reshape(-1))
                loss = out.loss + 0.3 * sub_loss

            (loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == 1:
                    el = time.time() - t0
                    mem = f" mem {torch.mps.driver_allocated_memory() / 2**30:.1f}GB" if device.type == "mps" else ""
                    print(f"epoch {epoch} step {step}/{total} loss {loss.item():.3f} (talker {out.loss.item():.3f} "
                          f"sub {sub_loss.item():.3f}) lr {sched.get_last_lr()[0]:.2e} {el / micro:.2f}s/micro "
                          f"eta {el / micro * (len(loader) * a.epochs - micro) / 60:.0f} min{mem}", flush=True)
                log.write(f"{epoch},{step},{loss.item():.4f},{out.loss.item():.4f},{sub_loss.item():.4f}\n"); log.flush()
            if a.max_steps and micro >= a.max_steps:
                print(f"smoke test done: {micro} micro-steps in {time.time() - t0:.0f}s", flush=True)
                if device.type == "mps":
                    print(f"mps memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GB")
                return
        if epoch in save_epochs:
            d = os.path.join(a.output_dir, f"epoch-{epoch}")
            save_checkpoint(model, a.init_model_path, d, a.speaker_name, spk_emb)
            print(f"saved {d}", flush=True)


if __name__ == "__main__":
    main()
