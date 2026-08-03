<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8" width="100%">

# Audio8_TTS Preview

**支持多语言语音生成和零样本音色克隆的 0.6B 参数文本转语音模型。**

[![GitHub](https://img.shields.io/badge/GitHub-Audio8__TTS-black?style=for-the-badge&logo=github)](https://github.com/Audio8-AI/Audio8_TTS)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge&logo=githubpages)](https://audio8-ai.github.io/Audio8_TTS/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.6b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
[![ONNX INT4](https://img.shields.io/badge/ONNX-INT4-005CED?style=for-the-badge&logo=onnx)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

English: [README.md](README.md)

</div>

本仓库提供 audio8_tts Preview 版模型、Hugging Face remote code、推理工具和独立的 SFT
训练流程，支持多语言语音生成和零样本音色克隆。

> **Preview 说明：** 当前版本的语言覆盖仍然有限，建议优先在下列 11 种语言中使用。后续版本将
> 持续补齐多语言和中文方言能力。

## 支持语言

Preview 模型当前表现较好、推荐使用的语言如下：

| Language | 语言 |
|---|---|
| Cantonese | 粤语 |
| Chinese | 中文 |
| Dutch | 荷兰语 |
| English | 英语 |
| French | 法语 |
| German | 德语 |
| Italian | 意大利语 |
| Japanese | 日语 |
| Korean | 韩语 |
| Polish | 波兰语 |
| Spanish | 西班牙语 |

## 模型结构

audio8_tts 沿用了 [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech)
的 DualAR 架构思路。

| 组件 | 配置 |
|---|---|
| 主模型 | 601,159,424 参数，不包含 codec |
| Slow AR | 24 层、896 维、14 个 attention head、2 个 KV head |
| Fast AR | 4 层、896 维、14 个 attention head、2 个 KV head |
| 声学 token | 10 个 codebook，每个 codebook 包含 4,096 个条目 |
| Codec | 44.1 kHz，每个模型帧 2,048 个采样点，约 21.5 帧/秒 |
| 上下文 | 最多 2,048 个打包后的文本/音频位置 |

Slow AR 每个音频帧生成一个 semantic token；Fast AR 根据 slow hidden state 和当前帧已经生成的
codebook，继续生成该帧的十个 codec codebook。推理时 slow AR 和 fast AR 都使用静态 KV cache。
Checkpoint 内置神经音频 codec，不需要额外下载参考音频编码器或波形解码器。

## 安装

推荐使用 Python 3.10 或更高版本以及支持 CUDA 的 GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

请从 [Hugging Face](https://huggingface.co/AutoArk-AI/Audio8-TTS-Preview-0.6b)
下载模型，并将其放入仓库的 `model/` 文件夹中。默认本地模型路径为
`model/audio8_tts_0_6B_preview/`。所有命令也可以通过 `--model` 接收 Hugging Face 模型 ID。

## 推理

为获得更好的合成效果，建议单次输入不超过 150 个字。文本过长可能影响生成质量，请根据
需要拆分为较短的片段。

### 零样本音色克隆

参考文本应当与参考音频中实际说出的内容一致。

```bash
python audio8_tts_infer.py \
  --text "欢迎使用 audio8_tts。" \
  --reference-audio examples/reference.wav \
  --reference-text "参考录音对应的准确文本。" \
  --output outputs/clone.wav
```

### 无参考音频生成

```bash
python audio8_tts_infer.py \
  --text "这条语音不使用参考音色。" \
  --output outputs/no_reference.wav
```

### 批量推理

输入文件每行是一个独立 JSON 对象。相对音频路径相对于 JSONL 文件所在目录解析。

```json
{"id":"sample_001","text":"目标文本","reference_audio":"audio/ref.wav","reference_text":"参考文本"}
{"id":"sample_002","text":"不使用参考音色的文本"}
```

```bash
python audio8_tts_infer.py \
  --input-jsonl data/prompts.jsonl \
  --output-dir outputs/batch \
  --batch-size 2
```

批量命令会生成 `manifest.jsonl` 和 `failures.jsonl`。除非传入 `--overwrite`，否则已有 WAV 会
被跳过。采样和 codec codes 保存参数可通过 `python audio8_tts_infer.py --help` 查看。

## CPU ONNX Runtime

[`onnx_runtime/`](onnx_runtime/) 提供独立的 CPU 部署方案：Slow/Fast AR 使用
weight-only INT4，activation 和 KV cache 使用 FP16，codec 使用 FP16。它包含
命令行推理、本地网页与 HTTP 服务、流式 PCM 输出和参考音频音色注册，运行时不依赖
PyTorch 或 Transformers。

在 Apple M2 测试环境中，在线模型 session 约占 1 GiB 内存。注册音色前会先释放
在线 session，再单独加载 codec encoder，从而控制峰值内存。

从
[Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
下载模型，并参阅 [ONNX Runtime 中文指南](onnx_runtime/README_zh.md)。

## SGLang Omni 服务部署

[`sglang_omni/`](sglang_omni/) 中的适配器提供兼容 OpenAI API 的推理服务，包含 SGLang
paged attention、动态 batching、Fast AR 固定 KV cache、参考音频编码和 waveform 解码。
它以独立的 `audio8_tts` 模型插件安装，不会覆盖 SGLang Omni 核心文件。

### 兼容版本

适配器使用了 SGLang Omni 的内部接口，因此部署时应固定到以下已验证版本，不要直接使用最新
`main` 分支。

| 依赖 | 已验证版本 |
|---|---|
| SGLang Omni | `68a572348837f7b004857b4b07993c20ade4c017`（`0.1.0`） |
| SGLang | `0.5.8` |
| PyTorch | `2.9.1+cu128` |
| Transformers | `4.57.1` |
| 精度 | BF16 |

### 性能

单流 warm latency 在单张 NVIDIA H20 上测试，使用 BF16、CUDA Graph、greedy decoding，
并生成 128 帧。输出 WAV 时长为 5.85-5.94 秒，不包含冷启动和编译时间。RTF 越低越好。

| SGLang Omni 适配器 | Warm p50 latency | RTF |
|---|---:|---:|
| 当前实现 | **0.691 s** | **0.116** |

配置、实现细节和验证结果请参阅
[SGLang Omni 实现与评测报告](sglang_omni/OPTIMIZATION_REPORT.md)。

### 安装

在 Audio8 TTS 仓库根目录执行以下命令。示例使用 Python 3.12 和
[`uv`](https://docs.astral.sh/uv/)。

```bash
export SGLANG_OMNI_ROOT=/opt/sglang-omni
export MODEL=/models/Audio8-TTS-Preview-0.6b

git clone https://github.com/sgl-project/sglang-omni.git "${SGLANG_OMNI_ROOT}"
git -C "${SGLANG_OMNI_ROOT}" checkout 68a572348837f7b004857b4b07993c20ade4c017

uv venv .venv-sglang --python 3.12
source .venv-sglang/bin/activate
uv pip install -v -e "${SGLANG_OMNI_ROOT}"

hf download AutoArk-AI/Audio8-TTS-Preview-0.6b --local-dir "${MODEL}"
./sglang_omni/scripts/install_adapter.sh "${SGLANG_OMNI_ROOT}"
python3 ./sglang_omni/scripts/verify_install.py --model-path "${MODEL}"
```

如果 SGLang Omni 已通过 wheel 或 site-packages 安装，可以定位其包目录后安装适配器：

```bash
SGLANG_OMNI_PACKAGE="$(python3 -c 'import importlib.util, pathlib; s=importlib.util.find_spec("sglang_omni"); assert s and s.origin; print(pathlib.Path(s.origin).parent)')"
./sglang_omni/scripts/install_adapter.sh "${SGLANG_OMNI_PACKAGE}"
```

### 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
AUDIO8_TTS_ENABLE_TORCH_COMPILE=1 \
HOST=0.0.0.0 \
PORT=8010 \
./sglang_omni/scripts/run_server.sh
```

默认的 `fa3` attention backend 适用于 H20、H100 等 Hopper GPU。RTX 5090 等消费级
Blackwell GPU 的计算能力为 `(12, 0)`，没有对应的 FA3 kernel image，适配器会自动识别
并为 SGLang Slow AR 路径选择 FlashInfer；此时短序列固定 KV cache 的 Fast head 会使用
PyTorch SDPA。无需额外配置。

在任意 GPU 上显式设置该环境变量仍会覆盖自动检测结果：

```bash
AUDIO8_TTS_ATTENTION_BACKEND=flashinfer \
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
./sglang_omni/scripts/run_server.sh
```

默认配置使用模型名 `audio8/tts-0.6b`、BF16、单卡、`0.2` 静态显存比例和最多 32 个并发
请求。主要运行参数包括 `MODEL_NAME`、`AUDIO8_TTS_MEM_FRACTION_STATIC`、
`AUDIO8_TTS_MAX_RUNNING_REQUESTS`、`AUDIO8_TTS_CHUNKED_PREFILL_SIZE` 和
`AUDIO8_TTS_DISABLE_CUDA_GRAPH`。开启 Torch compile 后，适配器沿用 SGLang 原生的 batch
策略；只有需要明确限制 compile 上限时才设置 `AUDIO8_TTS_TORCH_COMPILE_MAX_BS`。
`AUDIO8_TTS_ATTENTION_BACKEND` 默认为 `fa3`；在没有 FA3 kernel image 的 GPU（如消费级
Blackwell）上默认为 `flashinfer`，显式设置可覆盖。如果运行依赖安装在单独的
site-packages 目录中，请设置
`SGLANG_OMNI_SITE_PACKAGES`。

### 常见问题

- 如果 `sgl_kernel` 导入失败，请安装系统中提供 `libnuma` 的软件包，例如 `numactl` 或
  `libnuma1`。
- CUDA toolkit 的 `bin` 目录需要加入 `PATH`。如果 `deep_gemm` JIT 编译时找不到
  `nvcc`，还需要将 `CUDA_PATH` 指向 CUDA toolkit 根目录。
- Transformers 应保持在支持的 4.x 范围（`>=4.57.0,<5`）。Transformers 5.x 可能使该
  custom-code 模型生成无效的全零 codes。

### 调用 API

无参考音频生成：

```bash
curl -sS --fail-with-body \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "你好，这是 Audio8 TTS 服务测试。",
    "response_format": "wav",
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
  }' \
  http://127.0.0.1:8010/v1/audio/speech \
  -o audio8.wav
```

使用一个参考音色生成：

```bash
curl -sS --fail-with-body \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "这句话使用参考音色生成。",
    "response_format": "wav",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "references": [{
      "audio_path": "/data/reference.wav",
      "text": "参考录音对应的准确文本。"
    }]
  }' \
  http://127.0.0.1:8010/v1/audio/speech \
  -o audio8_clone.wav
```

参考音频路径必须在服务运行环境中可见。当前适配器支持 TP=1 和每个请求一个 reference。
设置 `"stream": true` 可在生成过程中接收 SSE 音频分块。使用
`"response_format": "pcm"` 开销最低，每个事件的 `audio.data` 包含 Base64 编码音频。
流式输出默认每 12 个 codec frames 生成一块，使用 128 帧 decoder context 和 1 帧边界
guard。可通过 `AUDIO8_TTS_STREAM_CHUNK_FRAMES`、
`AUDIO8_TTS_STREAM_CONTEXT_FRAMES` 和 `AUDIO8_TTS_STREAM_GUARD_FRAMES` 调整。

运行以下 smoke test 可以验证部署：

```bash
BASE_URL=http://127.0.0.1:8010 ./sglang_omni/scripts/smoke_test.sh
python3 ./sglang_omni/scripts/stream_smoke_test.py \
  --base-url http://127.0.0.1:8010 \
  --output /tmp/audio8_stream.wav
```

构建镜像时，请在安装 SGLang Omni 包及其 Python 依赖后追加
[`sglang_omni/Dockerfile.snippet`](sglang_omni/Dockerfile.snippet)。

## SFT 训练

先安装训练依赖：

```bash
pip install -r requirements-train.txt
```

### 1. 创建原始数据 manifest

目标 `audio` 必填；`reference_audio` 和 `reference_text` 可选，但必须同时出现。

```json
{"id":"utt_001","text":"目标音频文本","audio":"audio/target.wav","reference_audio":"audio/reference.wav","reference_text":"参考音频文本"}
{"id":"utt_002","text":"另一条文本","audio":"audio/another.wav"}
```

### 2. 预计算 codec indices

```bash
python audio8_tts_prepare.py \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared_data/train.jsonl \
  --batch-size 4
```

生成的 manifest 使用相对路径指向经过校验的 `[10, T]` NumPy 数组。除非传入
`--overwrite`，否则已有的有效数组会被复用。

### 3. 开始训练

单卡：

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=1 \
bash audio8_tts_sft.sh
```

单机八卡：

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=8 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash audio8_tts_sft.sh
```

多机训练时，在每个节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。常用超参
和输出路径可通过 `audio8_tts_sft.sh` 中列出的环境变量覆盖，也可以在命令末尾附加 Transformers
参数。

SFT 同时优化 slow semantic/EOS loss 和 fast codebook teacher-forcing loss。只训练其中一个分支时，
可设置 `FREEZE_SLOW_AR=true` 或 `FREEZE_FAST_AR=true`。导出目录仍可通过标准 `AutoModel` 和
`AutoProcessor` 接口配合 `trust_remote_code=True` 加载。

## 评测结果

Audio8 TTS Preview 仅有 **0.6B 参数**，是本次对比中规模最小的模型。尽管参数量仅为其他模型的
一小部分，它在以下基准测试中依然达到业界领先 TTS 模型的第一梯队水平。其中，Audio8 TTS
Preview 在 Seed-TTS 上取得了最低的英文 WER 和有竞争力的中文 CER，并在 CV3 多语言评测中
保持了具有竞争力的整体表现。

WER/CER 越低越好，SIM 越高越好。Seed-TTS 的相似度统一显示为百分数。

### Seed-TTS

| 模型 | 参数量 | EN WER / SIM | ZH CER / SIM | Hard ZH CER / SIM |
|---|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **1.506** / 63.2 | 0.950 / 73.1 | 11.510 / 68.7 |
| Fish S2 Pro | 4.6B | 1.607 / 64.6 | 1.038 / 73.8 | 10.149 / 70.1 |
| Higgs Audio v2 | 4.7B | 1.524 / 66.4 | **0.806** / 72.1 | 10.622 / 69.3 |
| CosyVoice3-1.5B | 1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / **75.8** |
| MOSS-TTS | 8.5B | 1.85 / 73.4 | 1.20 / 78.8 | - |
| VoxCPM2 | 2.3B | 1.84 / **75.3** | 0.97 / **79.5** | 8.13 / 75.3 |

![Seed-TTS WER 和 CER 对比](assets/evaluation/seed_tts_error_rates.png)

### CV3 多语言错误率

| 模型 | 参数量 | zh | en | hard-zh | hard-en | ja | ko | de | es | fr | it | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **3.205** | **3.128** | 10.535 | 5.997 | 7.205 | 4.223 | 3.447 | 3.641 | 8.790 | 4.790 | - |
| Fish S2 Pro | 4.6B | 3.600 | 3.493 | 10.588 | 7.349 | 5.139 | **4.111** | 3.605 | 2.972 | **8.600** | 4.229 | **4.702** |
| Higgs Audio v2 | 4.7B | 3.378 | 3.404 | 10.424 | **5.754** | **4.742** | 4.260 | **3.300** | **2.929** | 9.425 | **3.555** | 5.423 |
| CosyVoice3-1.5B | 1.5B | 3.91 | 4.99 | 9.77 | 10.55 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| VoxCPM2 | 2.3B | 3.65 | 5.00 | **8.55** | 8.48 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |

![CV3 多语言 WER 和 CER 对比](assets/evaluation/cv3_error_rates.png)

参数量直接根据已发布的权重张量统计。MOSS-TTS 包含 8,489,841,664 个参数；VoxCPM2 主模型
包含 2,290,004,544 个参数，单独的 AudioVAE 不计入本次参数量对比。

Fish S2 Pro 因官方评测使用自身 normalizer 而重新评测。Higgs Audio v2 因未公布具体数值而自行
评测。其他基线均采用 [VoxCPM 官方仓库](https://github.com/OpenBMB/VoxCPM)汇总的各项目官方值。

不同 normalizer 和评测器下的跨项目数值只能作为参考，不能视为严格同口径排名；评测覆盖也不
会将 Preview 的正式支持范围扩展到前述 11 种语言之外。

## 限制与负责任使用

- 当前是 Preview 模型，多语言和中文方言覆盖仍然有限。
- 过长、噪声较大或转写不准确的参考音频可能降低稳定性和音色相似度。
- 合成语音可能被用于冒充或传播虚假信息。克隆他人声音前应获得许可，并在适当场景明确标注合成内容。
- 部署前应针对具体业务完成准确性、安全性和合规测试。

## 许可证与致谢

本仓库代码和模型权重采用 [Apache License 2.0](LICENSE)，归属说明见 [NOTICE](NOTICE)。

感谢 Fish Audio 团队公开 Fish S2 Pro 使用的 DualAR 架构。

## Star 历史

<a href="https://www.star-history.com/?type=date&repos=Audio8-AI%2FAudio8_TTS">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&theme=dark&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
 </picture>
</a>
