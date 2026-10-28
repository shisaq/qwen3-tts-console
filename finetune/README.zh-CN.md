# 在 Mac 上微调一个声音

把一个人的几段视频或录音，在 Apple Silicon Mac 上训练成一个微调好的 Qwen3-TTS 音色，然后像其他音色一样在控制台里使用。也能在 CUDA 上跑。

[English](README.md)

## 流程

```
录音 ─ prep_data.py   ─▶ 切成 2–10 秒片段 + 转写 + “是不是本人”检查
     ─ review.py      ─▶ 人工试听、改错字、丢掉不好的片段（网页）
     ─ build_train.py ─▶ 编码成语音 token
     ─ sft.py         ─▶ 检查点（Qwen3-TTS-12Hz-1.7B-Base → 只有一个说话人的 custom_voice）
     ─ gen_samples.py + score_samples.py ─▶ 每一轮和零样本复刻放在一起对比
     ─ 写进控制台配置 tuned_voices ─▶ 完成
```

语音识别用的是 FunASR（Paraformer-zh，加 VAD 断句、标点、CAM++ 声纹），不是 Whisper：中文准，带逐字时间戳，CAM++ 能把主持人和别人的声音筛掉。中文夹英文术语没问题；以英文为主的录音会被丢弃。

## 实测（Mac mini，M5 Pro，64 GB）

| | |
|---|---|
| 数据准备，约 90 分钟录音 | 约 13 分钟 |
| 一个人复核后 | 165 段，18 分钟 |
| 训练 10 轮 | 17 分钟，统一内存约 38 GB |
| 训练集外句子的声纹相似度（CAM++，对比本人真实录音） | 零样本复刻 0.72 → 微调后 0.84 |
| 错字率 | 每一轮都低于 1% |
| 在控制台里 | 4.9 秒音频 1.7 秒生成完，模型加载 0.5 秒，约占 3 GB |

相似度是对比训练录音算的，提升里可能有一部分是录音环境（麦克风、房间）被学进去了，挑版本前一定要亲耳听。训练损失到第 10 轮降到约 0.16，听感最好的是第 6–10 轮，最后靠耳朵选。

微调出来的音色**不支持**语气/风格指令：Base 模型从没学过听指令。

## 安装

```bash
cd finetune
uv venv --python 3.12 .venv       && VIRTUAL_ENV=.venv       uv pip install -r requirements-prep.txt
uv venv --python 3.12 .venv-train && VIRTUAL_ENV=.venv-train uv pip install -r requirements-train.txt
cp voiceprep.example.json voiceprep.json     # 然后编辑
```

`voiceprep.json`（已加入 .gitignore）里列出每个说话人：一段干净的参考音频和它的逐字文稿，以及要挖素材的文件或文件夹。两个人都出现的录音（比如访谈）可以同时列在两人名下，每个片段会分给听起来像的那个人。说话人的 key 必须是英文字母，它会成为模型里的 speaker id。`terms` 用来纠正识别经常写错的词（产品名、英文术语）。

测试版本：funasr 1.4.16、modelscope 1.40.2、qwen-tts 0.1.1、torch 2.14.1，mlx-audio 见 `../requirements.txt`。

## 运行

```bash
.venv/bin/python prep_data.py                                   # 所有说话人
.venv/bin/python review.py                                      # http://127.0.0.1:18410 （无登录，只在本机开）
.venv-train/bin/python build_train.py alice
.venv-train/bin/python sft.py --train_jsonl ~/Datasets/voiceprep/alice/train_with_codes.jsonl \
    --speaker_name alice --output_dir ~/Datasets/voiceprep/alice/ckpt --epochs 10 --save_epochs 2,4,6,8,10
../.venv/bin/python gen_samples.py alice                       # 每一轮转成 MLX 8bit
.venv/bin/python score_samples.py alice                         # 然后去听：http://127.0.0.1:18410/compare/alice
```

复核页快捷键：↑↓ 选择 · 空格 播放 · K 保留 · D 丢弃 · 回车 编辑文字。改动存在 `edits.json`，重跑 `prep_data.py` 不会丢。带英文、语气词、重复/口误、数字的片段会被标记出来，优先看。

选好一轮后，把它放进控制台的模型目录，在 `~/.config/ttsd/config.json` 里加上：

```json
"tuned_voices": {
  "alice": {"name": "Alice", "model": "tuned/alice-e8-8bit", "speaker": "alice", "gender": "f", "lang": "zh"}
}
```

`speaker` 要和 `--speaker_name` 一致。重启 ttsd，音色会出现在「我的音色」里。

## 相对官方 `finetuning/sft_12hz.py` 的修正

用*未训练*的 Base 模型在真实片段上的损失来衡量：设置正确时损失很低，因为预训练模型本来就能很好地预测真实语音。

| | talker 初始损失 | 子模型初始损失 |
|---|---|---|
| 官方脚本 | 11.9 | 9.1 |
| + 标签只错位一次（[#179](https://github.com/QwenLM/Qwen3-TTS/issues/179)） | 7.9 | – |
| + 文字向量经过 `text_projection` | **1.2** | – |
| + 子模型用上一帧的隐藏状态、损失不再错位 | – | **5.4** |

1. **标签错位两次**（#179）：talker 的 HF 损失本身会错位一次，官方脚本又手动错位一次，模型学会了跳帧，语速越训越快。
2. **漏了 `text_projection`**：推理时每个文字向量都要过 `talker.text_projection`，官方训练循环没有。维度一样（2048），不会报错，模型一直在用推理时根本见不到的输入训练。
3. **子模型也错位两次**：`forward_finetune` 返回的 logits 已经和 `codec_ids[:, 1:]` 对齐，内置的 HF 损失又错位一次。和一帧配对的隐藏状态也应该是预测出这一帧的那个（t−1）。
4. **bf16 存权重**：学习率 2e-6 时，大部分 AdamW 更新小于 bf16 能表示的精度，被舍入掉了。这里用 fp32 存权重、bf16 计算。

另外：3.11 亿参数的文字嵌入层默认冻结（`--train_text_embedding` 可以打开）。不冻结时，64 GB 的 Mac 在第一次存检查点时进入内存交换，训练慢了约 12 倍。

数据准备绕开了 FunASR 1.4.16 的一个 bug：`sentence_info` 把每个标点都放晚一个字（“开发者大。家好很，高兴”），片段会在词中间断开。`split_sentences` 用完整文本和逐字时间戳重新切句。

生成时语言用 `auto`：训练用的是不带语言的前缀，指定语言会切换到说话人从没见过的前缀（控制台会自动处理）。
