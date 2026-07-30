<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8" width="100%">

# audio8_tts Preview

**A 0.6B-parameter multilingual text-to-speech model with zero-shot voice cloning.**

[![GitHub](https://img.shields.io/badge/GitHub-Audio8__TTS-black?style=for-the-badge&logo=github)](https://github.com/Audio8-AI/Audio8_TTS)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge&logo=githubpages)](https://audio8-ai.github.io/Audio8_TTS/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.6b-yellow?style=for-the-badge)](https://huggingface.co/AutoArk-AI/Audio8-TTS-Preview-0.6b)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

中文文档: [README_zh.md](README_zh.md)

</div>

This repository provides the audio8_tts Preview checkpoint, Hugging Face remote
code, inference tools, and an independent SFT pipeline for multilingual speech
generation and zero-shot voice cloning.

> **Preview status:** language coverage is intentionally limited in this
> release. Use the model primarily with the 11 recommended languages below.
> Multilingual coverage and Chinese dialect support will be expanded in later
> releases.

## Supported Languages

The Preview checkpoint performs best in the following languages:

| Language | Name |
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

## Architecture

audio8_tts uses a DualAR architecture inspired by
[Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech).

| Component | Configuration |
|---|---|
| Main model | 601,159,424 parameters, excluding the codec |
| Slow AR | 24 layers, width 896, 14 attention heads, 2 KV heads |
| Fast AR | 4 layers, width 896, 14 attention heads, 2 KV heads |
| Acoustic tokens | 10 codebooks, 4,096 entries per codebook |
| Codec | 44.1 kHz, 2,048 samples per model frame (~21.5 frames/s) |
| Context | Up to 2,048 packed text/audio positions |

The slow AR transformer predicts one semantic token for each audio frame. The
fast AR transformer then predicts the frame's codec codebooks, conditioned on
the slow hidden state and preceding codebooks. Static KV caches are used by
both branches during generation. The checkpoint also bundles its neural codec,
so reference encoding and waveform decoding require no separate model.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the checkpoint from
[Hugging Face](https://huggingface.co/AutoArk-AI/Audio8-TTS-Preview-0.6b) and
place it in the repository's `model/` directory. The expected local checkpoint
path is `model/audio8_tts_0_6B_preview/`. All commands also accept a Hugging
Face model ID through `--model`.

## Inference

### Zero-shot voice cloning

The reference transcript should match the spoken content in the reference
audio.

```bash
python audio8_tts_infer.py \
  --text "Welcome to audio8_tts." \
  --reference-audio examples/reference.wav \
  --reference-text "Transcript of the reference recording." \
  --output outputs/clone.wav
```

### Generation without a reference

```bash
python audio8_tts_infer.py \
  --text "This utterance does not use a reference voice." \
  --output outputs/no_reference.wav
```

### Batch inference

Each line in the input manifest is an independent JSON object. Relative audio
paths are resolved from the manifest directory.

```json
{"id":"sample_001","text":"Target text","reference_audio":"audio/ref.wav","reference_text":"Reference transcript"}
{"id":"sample_002","text":"Text without a reference voice"}
```

```bash
python audio8_tts_infer.py \
  --input-jsonl data/prompts.jsonl \
  --output-dir outputs/batch \
  --batch-size 2
```

The batch command writes `manifest.jsonl` and `failures.jsonl`. Existing WAV
files are skipped unless `--overwrite` is passed. See
`python audio8_tts_infer.py --help` for sampling and code-saving options.

## Supervised Fine-tuning

Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

### 1. Create a raw manifest

The target `audio` field is required. `reference_audio` and `reference_text`
are optional, but must be provided together.

```json
{"id":"utt_001","text":"Target transcript","audio":"audio/target.wav","reference_audio":"audio/reference.wav","reference_text":"Reference transcript"}
{"id":"utt_002","text":"Another transcript","audio":"audio/another.wav"}
```

### 2. Precompute codec indices

```bash
python audio8_tts_prepare.py \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared_data/train.jsonl \
  --batch-size 4
```

The prepared manifest points to validated `[10, T]` NumPy arrays using paths
relative to the prepared manifest. Existing valid arrays are reused unless
`--overwrite` is passed.

### 3. Train

Single GPU:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=1 \
bash audio8_tts_sft.sh
```

Eight GPUs on one node:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=8 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash audio8_tts_sft.sh
```

For multi-node training, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` on each node. Common hyperparameters and output paths can be
overridden through the environment variables in `audio8_tts_sft.sh`; additional
Transformers arguments may be appended to the command.

SFT optimizes both the slow semantic/EOS objective and the fast codebook
teacher-forcing objective. Set `FREEZE_SLOW_AR=true` or `FREEZE_FAST_AR=true`
when adapting only one branch. The exported directory remains loadable with
standard `AutoModel` and `AutoProcessor` APIs using `trust_remote_code=True`.

## Evaluation

Audio8 TTS Preview is the smallest model in this comparison at just **0.6B
parameters**. Despite using only a fraction of the parameters of the other
systems, it delivers results in the first tier of industry-leading SOTA TTS
models on the benchmarks below. In particular, it achieves the best English
WER and competitive Chinese CER on Seed-TTS, while remaining competitive
across the CV3 multilingual evaluation.

Lower WER/CER is better; higher SIM is better. Seed-TTS similarity values are
shown as percentages.

### Seed-TTS

| Model | Parameters | EN WER / SIM | ZH CER / SIM | Hard ZH CER / SIM |
|---|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **1.506** / 63.2 | 0.950 / 73.1 | 11.510 / 68.7 |
| Fish S2 Pro | 4.6B | 1.607 / 64.6 | 1.038 / 73.8 | 10.149 / 70.1 |
| Higgs Audio v2 | 4.7B | 1.524 / 66.4 | **0.806** / 72.1 | 10.622 / 69.3 |
| CosyVoice3-1.5B | 1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / **75.8** |
| MOSS-TTS | 8.5B | 1.85 / 73.4 | 1.20 / 78.8 | - |
| VoxCPM2 | 2.3B | 1.84 / **75.3** | 0.97 / **79.5** | 8.13 / 75.3 |

![Seed-TTS WER and CER comparison](figures/evaluation/fig1_seed_tts_error_rates.png)

### CV3 multilingual error rate

| Model | Parameters | zh | en | hard-zh | hard-en | ja | ko | de | es | fr | it | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **3.205** | **3.128** | 10.535 | 5.997 | 7.205 | 4.223 | 3.447 | 3.641 | 8.790 | 4.790 | - |
| Fish S2 Pro | 4.6B | 3.600 | 3.493 | 10.588 | 7.349 | 5.139 | **4.111** | 3.605 | 2.972 | **8.600** | 4.229 | **4.702** |
| Higgs Audio v2 | 4.7B | 3.378 | 3.404 | 10.424 | **5.754** | **4.742** | 4.260 | **3.300** | **2.929** | 9.425 | **3.555** | 5.423 |
| CosyVoice3-1.5B | 1.5B | 3.91 | 4.99 | 9.77 | 10.55 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| VoxCPM2 | 2.3B | 3.65 | 5.00 | **8.55** | 8.48 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |

![CV3 multilingual WER and CER comparison](figures/evaluation/fig2_cv3_error_rates.png)

Parameter counts are calculated directly from the released weight tensors.
MOSS-TTS contains 8,489,841,664 parameters. VoxCPM2's main model contains
2,290,004,544 parameters; the separate AudioVAE is not included in the
parameter comparison.

Fish S2 Pro was reevaluated because its official evaluation uses its own
normalizer. Higgs Audio v2 was evaluated locally because concrete values were
unavailable. All other baseline values were collected from their official
reports through the [VoxCPM repository](https://github.com/OpenBMB/VoxCPM).

Different normalizers and evaluators make cross-project values reference
comparisons rather than a strictly matched ranking. Evaluation coverage does
not expand the Preview's supported-language claim beyond the 11 languages
listed above.

## Limitations and Responsible Use

- This is a Preview checkpoint with limited multilingual and dialect coverage.
- Very long, noisy, or inaccurate reference clips can reduce stability and
  speaker similarity.
- Generated speech can be misused for impersonation or misinformation. Obtain
  consent before cloning a voice and clearly disclose synthetic audio where
  appropriate.
- Test the model for accuracy, safety, and legal compliance before deployment.

## License and Acknowledgements

Code and model weights in this repository are released under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution details.

We thank the Fish Audio team for publishing the DualAR architecture used in
Fish S2 Pro.
