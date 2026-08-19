# Audio8 TTS 0.1B

Audio8-TTS-0.1B 是紧凑的零样本音色克隆 TTS 模型。本分支包含 Falcon H1 多阶段
SFT、可选 GRPO，以及 Hugging Face 批量推理代码。SGLang 和 ONNX 运行时被明确移除：
它们面向另一代 checkpoint，当前尚未适配 0.1B Falcon H1 架构。

## 安装

训练配方在 Python 3.11、PyTorch 2.8.0（CUDA 12.8）、Transformers 4.57.6、
DeepSpeed 0.18.4 上验证。请先安装与本机匹配的 CUDA 版 PyTorch，再安装锁定依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
```

SFT 需要外部 Fish Speech checkout 提供 tokenizer/conversation 工具。请将其克隆或
挂载到仓库外，并在 `.env` 设置 `FISH_SPEECH_ROOT`。集群与数据契约见
[`docs/0.1B/ENVIRONMENT.md`](docs/0.1B/ENVIRONMENT.md) 和
[`docs/0.1B/DATA.md`](docs/0.1B/DATA.md)。

```bash
cp .env.example .env
# 修改路径与分布式拓扑。
bash scripts/utils/preflight.sh
```

## 训练

可复现阶段链为：

```text
AUDIO8_INIT_MODEL -> v1_slowar -> v2_fastar -> v3_joint -> v4_mixed
```

按顺序执行：

```bash
bash scripts/train/v1_slowar.sh
bash scripts/train/v2_fastar.sh
bash scripts/train/v3_joint.sh
bash scripts/train/v4_mixed.sh
```

所有启动脚本都支持通过环境变量覆盖模型路径、manifest、batch size、学习率、拓扑、
端口和恢复策略。生产默认值面向 3 节点、每节点 8 GPU；正式长跑前请先用小任务验证
显存与吞吐。详见 [`docs/0.1B/TRAINING.md`](docs/0.1B/TRAINING.md)。

可选 GRPO 通过 `scripts/train/grpo.sh` 启动，需要额外 ASR 与 speaker similarity
资产，详见环境说明。

## 推理

批量推理使用导出的 Hugging Face 模型包及其 remote code，不依赖 SGLang 或 ONNX。
manifest 每行需要 `id` 与 `text`；做音色克隆时，还可提供
`reference_audio`/`reference_text` 或 `reference_codes`/`reference_text`：

```json
{"id":"clone","text":"欢迎使用 Audio8。","reference_audio":"/data/ref.wav","reference_text":"参考音频转写。"}
{"id":"no_ref","text":"这条音频不使用参考音色。"}
```

```bash
MODEL=/path/to/audio8-0.1B-export \
INPUT_JSONL=/data/prompts.jsonl \
OUTPUT_DIR=/data/generated \
bash scripts/infer/run_batch.sh
```

`MODEL` 也可以是 Hugging Face model ID。脚本会在 `OUTPUT_DIR` 输出 WAV 和 Audio8
codec token NPY 文件。

## 目录结构

```text
configs/deepspeed/       ZeRO-2 训练配置
docs/0.1B/               环境、数据与训练契约
examples/                hostfile 与推理 manifest 示例
scripts/train/           v1-v4 SFT 与 GRPO 启动脚本
scripts/infer/           Hugging Face 批量推理
scripts/utils/           预检、分片与行数统计工具
src/                     数据集、trainer 与 reward worker
```

## 许可证

仓库代码使用 Apache-2.0。Fish Speech、数据集、checkpoint 与奖励模型仍受各自许可
约束；见 `NOTICE` 与 `third_party/FISH_SPEECH_LICENSE`。
