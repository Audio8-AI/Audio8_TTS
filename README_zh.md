<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8 TTS" width="760">

# Audio8 TTS Preview 0.1B

**足够小巧、也值得运行的零样本 TTS。**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.1b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge)](https://audio8-ai.github.io/Audio8_TTS/)
[![License](https://img.shields.io/badge/Model%20License-CC--BY--NC--4.0-blue?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)

**中英文优先 · 多语言实验支持 · 零样本音色克隆**

</div>

Audio8 TTS Preview 0.1B 是一个面向语音生成和零样本音色克隆的紧凑
audio-language 模型。完整 v4 mixed checkpoint、神经音频 codec、tokenizer、
processor 以及 Hugging Face remote code 都打包在同一个模型仓库中：

```text
Audio8/Audio8-TTS-Preview-0.1b
```

这个 GitHub 分支提供配套的 Falcon H1 训练配方和批量推理工具。原有 SGLang
和 ONNX runtime 已移除：它们面向另一代模型，当前没有适配该 checkpoint。

## 为什么是 0.1B？

这个版本最重要的特征是规模。主生成模型约 **170M 参数**， bundled codec
decoder 约 **120M 参数**；即使把 codec decoder 计入，完整音频生成栈仍明显
小于多数现代多语言 TTS 系统。

| 模型 | 主模型参考规模 |
|---|---:|
| **Audio8 TTS Preview 0.1B** | **~0.17B** |
| Audio8 TTS Preview 0.6B | ~0.6B |
| IndexTTS2.5 | ~0.8B |
| CosyVoice3 | ~1.5B |
| VoxCPM2 | ~2.3B |
| Fish S2 Pro | ~4.6B |
| Higgs Audio v2 | ~4.7B |
| MOSS-TTS | ~8.5B |

以上为各模型报告中的近似参考规模，不是严格对齐的参数审计。

## 支持语言

- **主要语言：** 中文、英文
- **实验语言：** 德语、西班牙语、法语、意大利语、日语、韩语

推荐以中英文作为生产目标。实验语言可用于评估，但质量和稳定性通常更弱。

## 架构

Audio8 使用 Falcon H1 风格的双自回归架构：

1. Slow AR 分支预测语义音频 token；
2. Fast AR 分支在 slow hidden state 条件下预测 codec codebooks；
3. bundled codec 将生成 token 解码为 44.1 kHz waveform。

| 组件 | 配置 |
|---|---|
| 主模型 | 约 170M 参数，不含 codec decoder |
| Slow AR | 24 层，宽度 512，8 attention heads，2 KV heads |
| Fast AR | 4 层，宽度 512，8 attention heads，2 KV heads |
| Acoustic tokens | 10 codebooks，每册 4,096 entries |
| Codec | 44.1 kHz，每 frame 2,048 samples（约 21.5 frames/s） |
| Codec decoder | 约 120M 参数，打包为 `codec.pth` |
| Context | 最多 2,048 packed text/audio positions |

## 安装

推荐 Python 3.11 和支持 CUDA 的 GPU。先安装与本机 CUDA 匹配的 PyTorch，
再安装推理依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
```

训练还需要外部 Fish Speech checkout，用于 tokenizer 与 conversation 工具。
请将其克隆或挂载到仓库外，并在 `.env` 中设置 `FISH_SPEECH_ROOT`。

## 推理

### 直接使用 Transformers

模型包含 custom Transformers code，请使用 `trust_remote_code=True` 加载：

```python
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

model_id = "Audio8/Audio8-TTS-Preview-0.1b"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, trust_remote_code=True, dtype=dtype
).eval().to(device)

inputs = processor(
    text=["这是一个语音合成测试。"],
    reference_audio=["reference.wav"],
    reference_text=["参考音频对应的完整文本。"],
    return_tensors="pt",
)
inputs = {name: value.to(device) for name, value in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        return_dict_in_generate=True,
    )
    waveforms, waveform_lengths = model.decode_audio(output.codes)

audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
sf.write("output.wav", audio, model.config.codec_sample_rate)
```

不做音色克隆时，可以省略 `reference_audio` 和 `reference_text`。

### JSONL 批量推理

每行一个 JSON 对象：

```json
{"id":"clone_zh","text":"这是一个零样本语音合成测试。","reference_audio":"/data/ref.wav","reference_text":"参考音频对应的完整文本。"}
{"id":"codes_en","text":"This is an Audio8 Falcon H1 inference test.","reference_codes":"/data/ref_codes.npy","reference_text":"Reference transcript."}
{"id":"no_reference","text":"This sample uses the model's non-cloning generation mode."}
```

运行：

```bash
MODEL=Audio8/Audio8-TTS-Preview-0.1b \
INPUT_JSONL=/data/prompts.jsonl \
OUTPUT_DIR=/data/generated \
bash scripts/infer/run_batch.sh
```

`MODEL` 可以是 Hugging Face model ID，也可以是本地导出模型目录。批量脚本会
输出 WAV 和 Audio8 codec-token NPY 文件。

## 训练

本仓库复现 v4 mixed 训练路线：

```text
AUDIO8_INIT_MODEL -> v1 SlowAR -> v2 FastAR -> v3 Joint -> v4 Mixed
```

```bash
cp .env.example .env
# 配置模型、数据、Fish Speech、存储和集群路径。
bash scripts/utils/preflight.sh

bash scripts/train/v1_slowar.sh
bash scripts/train/v2_fastar.sh
bash scripts/train/v3_joint.sh
bash scripts/train/v4_mixed.sh
```

所有启动脚本都支持通过环境变量覆盖 manifest、模型路径、拓扑、batch size、
学习率、端口、checkpoint 和恢复策略。默认配方面向 3 节点、每节点 8 GPU；
正式长跑前请先用小任务验证显存和吞吐。

可选 GRPO 训练入口是 `scripts/train/grpo.sh`。详细说明见：

- [环境与集群要求](docs/0.1B/ENVIRONMENT.md)
- [训练阶段与恢复语义](docs/0.1B/TRAINING.md)
- [数据集与 manifest 契约](docs/0.1B/DATA.md)

## Evaluation

在 CV3 上，0.1B checkpoint 达到 **3.619% 中文错误率** 和 **3.307% 英文
错误率**，同时规模约为 0.6B 版本的三分之一。在 Seed-TTS 上，它达到
**1.662% EN WER / 56.7 SIM** 与 **1.13% ZH CER / 68.2 SIM**。

完整多语言表格、评测协议和 baseline 引用见
[模型卡](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b#evaluation)。
不同 TTS 项目的 normalizer 和 evaluator 不完全一致，应作为参考对比，而不是
严格排名。

## 负责任使用

- 克隆音色前必须取得授权。
- 在适当场景披露音频由 AI 生成。
- 避免使用嘈杂、过长或转写错误的参考音频。
- 上线前评估准确率、speaker similarity、安全性和法律合规性。

## License

- **模型 checkpoint 与 Hugging Face remote code：** CC-BY-NC-4.0
- **本仓库中的训练和推理代码：** Apache-2.0
- **Fish Speech 及其他第三方资产：** 受各自许可证约束；见
  [`NOTICE`](NOTICE) 与
  [`third_party/FISH_SPEECH_LICENSE`](third_party/FISH_SPEECH_LICENSE)。
