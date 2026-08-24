<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8" width="100%">

# Audio8_TTS Preview

**A 0.6B-parameter multilingual text-to-speech model with zero-shot voice cloning.**

[![GitHub](https://img.shields.io/badge/GitHub-Audio8__TTS-black?style=for-the-badge&logo=github)](https://github.com/Audio8-AI/Audio8_TTS)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge&logo=githubpages)](https://audio8-ai.github.io/Audio8_TTS/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.6b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
[![ONNX INT4](https://img.shields.io/badge/ONNX-INT4-005CED?style=for-the-badge&logo=onnx)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

中文文档: [README_zh.md](README_zh.md)

</div>

> 🚀 **How small a zero-shot cloning TTS can be?** Meet our latest release,
> **Audio8-TTS-0.1B**: a powerful, compact, and even portable text-to-speech
> model!
>
> [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.1b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)
> [![Demo-0.1B Audio Samples](https://img.shields.io/badge/Demo--0.1B-Audio%20Samples-brightgreen?style=for-the-badge&logo=githubpages)](https://audio8-ai.github.io/Audio8_TTS/0.1B/)

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
[Hugging Face](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) and
place it in the repository's `model/` directory. The expected local checkpoint
path is `model/audio8_tts_0_6B_preview/`. All commands also accept a Hugging
Face model ID through `--model`.

## Inference

For best synthesis quality, keep each input within 150 characters. Longer text
may reduce generation quality; split it into shorter segments when needed.

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

## CPU ONNX Runtime

[`onnx_runtime/`](onnx_runtime/) provides a standalone CPU deployment using
weight-only INT4 Slow/Fast AR models, FP16 activations and KV caches, and an
FP16 codec. It includes CLI inference, a local web and HTTP service, streaming
PCM output, and reference-voice registration without PyTorch or Transformers.

The online sessions use about 1 GiB of memory in the tested Apple M2 setup.
During voice registration, the online sessions are released before the codec
encoder is loaded to keep peak memory low.

Download the ONNX model from
[Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
and follow the [ONNX Runtime guide](onnx_runtime/README.md).

## Rust Runtime

[`rust_runtime/`](rust_runtime/) is a native Rust port of the ONNX Runtime
inference path, built against the [`ort`](https://ort.pyke.io/) crate. It runs
the same INT4 Slow/Fast AR graphs and FP16 codec as `onnx_runtime/`, with a
from-scratch Rust port of the tokenizer/prompt builder, autoregressive
sampling loop, and KV-cache management.

CPU and CUDA execution providers are both supported (`--cuda` flag). The CUDA
path uses `ort`'s IoBinding API to keep the KV-cache and activation tensors
address-stable across steps, avoiding a host-side reallocation on every
autoregressive step.

```bash
cd rust_runtime
cargo build --release --bin synth
./target/release/synth.exe --cuda "Your text here."
```

Requires a registered voice at `../onnx_runtime/voices/` (see the [ONNX
Runtime guide](onnx_runtime/README.md) for registration) and the downloaded
ONNX model at `../onnx_runtime/model/`. The CUDA path additionally requires:

- **cuDNN 9.x** for CUDA 12 (this repo was developed against 9.25.0.15).
- **CUDA 13 cuBLAS runtime libraries**, even on a machine with only CUDA
  Toolkit 12.x installed: `ort` 2.0.0-rc.13's `download-binaries` feature
  fetches a CUDA-13-built ONNX Runtime, which needs `cublasLt64_13.dll` at
  runtime regardless of what CUDA Toolkit version is on the system PATH. The
  fastest way to get it without a full CUDA 13 Toolkit install:
  `pip install nvidia-cublas --target rust_runtime/cuda13_libs`, then add
  `rust_runtime/cuda13_libs/nvidia/cu13/bin/x86_64` to `PATH` alongside the
  cuDNN directory before running.

Both requirements' DLL directories need to be on `PATH` at build and run
time; see `rust_runtime/cudnn/` and `rust_runtime/cuda13_libs/` (both
gitignored - project-local, not committed).

### Performance

Measured on an RTX 3060 Laptop GPU (6 GB VRAM, consumer Ampere) and a 16-core
Windows machine, same registered reference voice, comparable text lengths.
Lower RTF is better.

| Path | RTF |
|---|---:|
| **Rust, ONNX Runtime, CUDA (`rust_runtime/`)** | **2.01-3.36** (best warm run: 2.01) |
| Python, ONNX Runtime, CPU (`onnx_runtime/`) | 2.72 |
| Rust, candle, CUDA (`candle_runtime/`) | 2.93-3.91 (best warm run: 2.93) |
| Rust, ONNX Runtime, CPU (`rust_runtime/`) | 3.21-4.63 |
| Python, PyTorch, CUDA, eager mode (`audio8_tts_infer.py`) | 6.44 |
| Rust, candle, CPU (`candle_runtime/`) | 14.78 |

This machine is shared with other processes at measurement time (this
development session's own GPU/CPU load among them) - `nvidia-smi`/CPU-counter
checks during otherwise-idle moments still showed 40%+ GPU and ~79% CPU
utilization from other work, which is enough to move these numbers by a
factor of ~1.5x run to run. The ranges above reflect that; the best observed
warm CUDA run (2.01) is the number to trust as the ceiling on this hardware
when it isn't contended.

Rust CUDA is tuned further on top of the base fix (TF32, explicit attention
backend selection, exhaustive cuDNN conv algorithm search, and 2 pinned
intra-op threads - CUDA-path only, see below) for the current best numbers.
One real regression was caught and fixed along the way: pinning intra-op
threads globally first *broke* the CPU path (3.48-4.42 -> 4.63, measured, not
assumed) because it genuinely relies on multi-threaded matmul, unlike the
CUDA path's few remaining CPU-fallback ops (see below), which are tiny
single-sequence shape ops that gain nothing from extra threads. Scoping the
thread pin to the CUDA execution provider only fixed it.

`ort`'s `with_disable_cpu_fallback` (forcing every op onto CUDA) was also
tried, on the theory that ORT's own CPU-fallback choice for cheap shape ops
is tuned for larger/batched workloads and might lose to this model's
hundreds of tiny sequential decode-loop calls. It hard-fails instead -
`disable_cpu_fallback` blocks ORT's own perf-motivated placements exactly
like a genuinely-unsupported op, with no way to distinguish the two through
`ort`'s public API. Not pursued further.

Rust CUDA is the fastest local path measured. Getting here took real,
witnessed debugging, not just parameter tuning - worth recording because the
mistake is easy to repeat:

- The first pass at this (see git history) chased a "GPU slower than CPU"
  result through a real bug (the KV cache was being fully re-copied
  host-to-device every decode step instead of just the changed position -
  fixed, and worth keeping regardless) and a "CUDA graph capture isn't
  activating" finding that looked well-evidenced (`nvidia-smi` utilization,
  per-call timing) but was built on a false premise.
- The false premise: **the CUDA execution provider was silently failing to
  register on every single run**, the entire time. `ep=Cuda` was logged,
  `--cuda` was passed, cuDNN loaded without error - but every actual
  compute node was running on `CPUExecutionProvider`. This was invisible
  because `ort`'s own EP-registration errors are emitted through the Rust
  `tracing` crate, not the ONNX Runtime C-API logger - and no `tracing`
  subscriber was installed, so the error was silently discarded on every
  run. Installing one (`tracing_subscriber::fmt()...init()`, now done
  unconditionally at `WARN` level in `synth.rs`) surfaced the real error
  immediately: `Error loading onnxruntime_providers_cuda.dll which depends
  on cublasLt64_13.dll which is missing`.
- The real root cause: `ort` 2.0.0-rc.13's `download-binaries` feature only
  ships a CUDA-13-built ONNX Runtime (confirmed in `ort-sys`'s own build
  script - "we only ship 13 for now"), but this machine only has CUDA
  Toolkit 12.6 installed. A version mismatch, not a hardware or model
  limitation. Fixed by installing just the missing CUDA 13 cuBLAS runtime
  DLLs (`pip install nvidia-cublas`, far faster than a full Toolkit
  reinstall) - see the setup instructions above.
- With the CUDA EP genuinely active, ORT's own node-placement logs confirm
  real GPU execution (2089/308/1159 nodes on `CUDAExecutionProvider` across
  the three sessions) and `nvidia-smi` shows real bursty 0-53% utilization,
  not the earlier flat ~15%.
- **CUDA graph capture is still off** (`with_cuda_graph` defaults to
  `false`; `ARKTTS_CUDA_GRAPH=1` re-enables it) - not because it isn't
  activating, but because ONNX Runtime explicitly refuses it for this
  model: some ops (816/124/367 nodes across the three sessions) are
  CPU-assigned even with the CUDA EP active, and ORT hard-errors rather
  than degrading gracefully when graph capture is requested against a
  mixed-provider graph. This is a genuine, documented ONNX Runtime
  constraint for this model as exported, not a bug in this code or in
  `ort`.
- Those CPU-fallback nodes were inspected directly (full node-by-node
  placement logs, `ARKTTS_VERBOSE_ORT=1`): every one is `Gather`, `Concat`,
  `Unsqueeze`, `Slice`, or `Cast` - cheap shape/indexing bookkeeping for the
  KV cache, never `MatMul`, `Softmax`, `Attention`, or the INT4 quantization
  ops. All real compute already runs on CUDA. ORT's own log explains this is
  a deliberate choice ("shape related ops assigned to CPU to improve
  perf"), not a coverage gap - confirmed by `with_disable_cpu_fallback`
  hard-failing rather than running everything on GPU (see above). Chasing
  full CUDA placement was correctly not worth pursuing further.

The [SGLang Omni](#sglang-omni-serving) path is still faster (RTF 0.116) on
the datacenter-class hardware it was validated against - full CUDA graph
capture there comes from a model/runtime combination with complete CUDA op
coverage, which this ONNX export does not have. Closing that remaining gap
would mean re-exporting the model with the quantization ops replaced by
CUDA-covered equivalents, which is separate, larger work.

## Rust, candle Runtime (`candle_runtime/`)

A second, independent Rust implementation of the DualAR model, built directly
on [candle](https://github.com/huggingface/candle) (HuggingFace's Rust ML
framework) instead of the ONNX Runtime. The motivation: ONNX Runtime refuses
CUDA graph capture outright for this model because a handful of cheap
shape-bookkeeping ops (`Gather`/`Concat`/`Unsqueeze`/`Slice`/`Cast`, never
real compute) stay CPU-assigned, and ORT hard-errors on graph capture against
any mixed-provider graph rather than degrading gracefully (see above). A
model built natively in candle has no such mixed-provider constraint - every
op, including the small shape ones, runs through the same CUDA backend, so
graph capture was worth investigating as a path toward SGLang-class
performance without needing SGLang itself (Linux-only compiled CUDA kernels,
no Windows wheels - see below).

**What's here:** a full from-scratch port of the DualAR architecture (24-layer
slow AR + 4-layer fast AR, GQA, RoPE, RMSNorm) and the codec decoder (RVQ
dequant, causal-masked transformer, ConvNeXt upsampling, DAC-style Snake
vocoder) to candle, plus a from-scratch ONNX-to-candle weight conversion
(`scripts/extract_onnx_weights.py` dequantizes the ONNX
`GatherBlockQuantized`/`MatMulNBits` INT4 weights, `repack_quantized_weights`
requantizes them into candle's Q4_0 GGUF format - the two formats are
incompatible on block size, symmetry, and zero-point handling, so this is a
real dequantize-then-requantize conversion, not a bit repack). Correctness
was verified independently at each stage: model logits correlate 0.9956
against the ONNX reference, the codec decoder correlates 1.0000 (mae
0.000025), and end-to-end synthesis was verified via direct waveform
inspection (RMS, duration, non-silence, and spectral speech-band
concentration) exactly as done for every other engine in this project.

**CUDA graph capture was not made reliable.** This is the honest negative
result the investigation converged on, not a partial success. candle-core
0.11.0 (and a git-main pin, hoping upstream's post-0.11.0 "Improve CUDA
stream consistency" and "Add htod cache for cuda graphs" fixes would help)
both show the same intermittent `CUDA_ERROR_ILLEGAL_ADDRESS` on the first
launch after a successful capture+instantiate, once the captured region
exceeds roughly one transformer layer's worth of ops: a single simple op
captures reliably (8/8, 22/22 repeated runs), a single full layer succeeds
only ~40% of the time (4/10 measured), and the full 4-layer forward pass
captured as one graph failed 0/10+ times. Retry-until-success (up to 20
attempts, since capture only happens once at model load) plus a per-layer
chained-capture redesign (5 small graphs instead of 1 large one) were both
tried as workarounds - the chained approach failed 0/8 on the first attempt
per fresh process and 0/5 with 20 retries each (100 total attempts, zero
successes), decisively worse than the single-op case, not solved by
retrying. This is a real, reproducible reliability limitation in
candle-core's current CUDA graph implementation on this GPU/driver
combination, not a bug in this project's code - see
`candle_runtime/src/graph_decode.rs`'s module doc for the full trail.

**Result: correct, but not faster.** Without graph capture, the candle
engine's plain (uncaptured) forward-pass execution measured RTF 2.93-3.91
(best warm run 2.93) on CUDA - slightly worse than the already-tuned ONNX
Runtime CUDA path (2.01-3.36, best 2.01) on the same hardware, same
methodology (same text, same voice, same seed, same sampling parameters).
The candle port's real value going forward is a correct, portable,
from-scratch model implementation with no ONNX Runtime dependency, not a
speed win - closing the gap to SGLang's 0.116 RTF still requires either an
upstream candle-core CUDA graph reliability fix or a different graph-capture
strategy not yet tried.

### Building and running

```
cd candle_runtime
# CPU:
cargo build --release --bin synth
ARKTTS_MODEL_DIR=../onnx_runtime/model ARKTTS_VOICES_DIR=../onnx_runtime/voices \
  ./target/release/synth.exe "Your text here."

# CUDA (needs cuDNN + the CUDA 13 cuBLAS runtime DLLs on PATH, same as
# rust_runtime - see Setup above; candle-core is pinned to git main, which
# additionally requires cl.exe on PATH to build its CUDA kernels from source):
cargo build --release --bin synth --features cuda
ARKTTS_MODEL_DIR=../onnx_runtime/model ARKTTS_VOICES_DIR=../onnx_runtime/voices \
  ./target/release/synth.exe --cuda --repeat 3 "Your text here."
```

Weights are extracted from the ONNX model into `candle_runtime/weights/`
(gitignored, regenerable) via `scripts/extract_onnx_weights.py` followed by
the `repack_quantized_weights` binary; `synth` loads them from there directly.

## SGLang Omni Serving

The adapter in [`sglang_omni/`](sglang_omni/) provides an OpenAI-compatible
service with SGLang paged attention, dynamic batching, a fixed KV cache for the
fast codebook decoder, reference-audio encoding, and waveform decoding. It is
installed as an independent `audio8_tts` model plugin and does not overwrite
SGLang Omni core files.

### Compatibility

The adapter uses internal SGLang Omni interfaces, so deploy it with the tested
revision instead of the latest `main` branch.

| Dependency | Tested version |
|---|---|
| SGLang Omni | `68a572348837f7b004857b4b07993c20ade4c017` (`0.1.0`) |
| SGLang | `0.5.8` |
| PyTorch | `2.9.1+cu128` |
| Transformers | `4.57.1` |
| Precision | BF16 |

### Performance

Warm single-stream latency was measured on one NVIDIA H20 with BF16 weights,
CUDA Graph, greedy decoding, and 128 generated frames. The output WAV was
5.85-5.94 seconds long; cold start and compilation time were excluded. Lower
RTF is better.

| SGLang Omni adapter | Warm p50 latency | RTF |
|---|---:|---:|
| Current implementation | **0.691 s** | **0.116** |

See the [SGLang Omni implementation and evaluation report](sglang_omni/OPTIMIZATION_REPORT.md)
for the configuration, implementation details, and validation results.

### Install

Run these commands from the Audio8 TTS repository root. The example uses
Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

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

For an existing wheel or site-packages installation, resolve the package
directory and install the adapter there:

```bash
SGLANG_OMNI_PACKAGE="$(python3 -c 'import importlib.util, pathlib; s=importlib.util.find_spec("sglang_omni"); assert s and s.origin; print(pathlib.Path(s.origin).parent)')"
./sglang_omni/scripts/install_adapter.sh "${SGLANG_OMNI_PACKAGE}"
```

### Start the service

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
AUDIO8_TTS_ENABLE_TORCH_COMPILE=1 \
HOST=0.0.0.0 \
PORT=8010 \
./sglang_omni/scripts/run_server.sh
```

The default `fa3` attention backend is intended for Hopper GPUs such as H20
and H100. Consumer Blackwell GPUs such as RTX 5090 report compute capability
`(12, 0)` and have no FA3 kernel image, so the adapter detects them and selects
FlashInfer for the SGLang slow-AR path automatically; the short fixed-cache
fast head then uses PyTorch SDPA. No configuration is required.

Setting the variable explicitly still overrides the detection on any GPU:

```bash
AUDIO8_TTS_ATTENTION_BACKEND=flashinfer \
CUDA_VISIBLE_DEVICES=0 \
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT}" \
MODEL="${MODEL}" \
./sglang_omni/scripts/run_server.sh
```

The defaults use model name `audio8/tts-0.6b`, BF16, one GPU, a `0.2` static
memory fraction, and up to 32 running requests. The main runtime controls are
`MODEL_NAME`, `AUDIO8_TTS_MEM_FRACTION_STATIC`,
`AUDIO8_TTS_MAX_RUNNING_REQUESTS`, `AUDIO8_TTS_CHUNKED_PREFILL_SIZE`, and
`AUDIO8_TTS_DISABLE_CUDA_GRAPH`. When Torch compilation is enabled, the adapter
uses SGLang's native batch-size policy. Set `AUDIO8_TTS_TORCH_COMPILE_MAX_BS`
only when an explicit compile limit is needed. `AUDIO8_TTS_ATTENTION_BACKEND`
defaults to `fa3`, except on GPUs with no FA3 kernel image such as consumer
Blackwell, where it defaults to `flashinfer`; set it explicitly to override. Set
`SGLANG_OMNI_SITE_PACKAGES` when the runtime dependencies are installed in a
separate site-packages directory.

### Troubleshooting

- Install the distribution package that provides `libnuma` if `sgl_kernel`
  fails to import (for example, `numactl` or `libnuma1`).
- Put the CUDA toolkit `bin` directory on `PATH`. If `deep_gemm` cannot find
  `nvcc` during JIT compilation, also set `CUDA_PATH` to the toolkit root.
- Keep Transformers on the supported 4.x range (`>=4.57.0,<5`). Transformers
  5.x can produce invalid all-zero codes for this custom-code model.

### Call the API

Generate speech without a reference:

```bash
curl -sS --fail-with-body \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "Hello from Audio8 TTS.",
    "response_format": "wav",
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
  }' \
  http://127.0.0.1:8010/v1/audio/speech \
  -o audio8.wav
```

Generate speech with one reference voice:

```bash
curl -sS --fail-with-body \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "This sentence uses the reference voice.",
    "response_format": "wav",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "references": [{
      "audio_path": "/data/reference.wav",
      "text": "The exact transcript of the reference recording."
    }]
  }' \
  http://127.0.0.1:8010/v1/audio/speech \
  -o audio8_clone.wav
```

The reference path must be visible inside the service environment. The current
adapter supports TP=1 and one reference per request. Set `"stream": true` to
receive SSE audio chunks as they are generated. For the lowest overhead, use
`"response_format": "pcm"`; each event contains Base64-encoded audio in
`audio.data`. Streaming defaults to 12 codec frames per chunk, with 128 frames
of decoder context and a one-frame boundary guard. Override these values with
`AUDIO8_TTS_STREAM_CHUNK_FRAMES`, `AUDIO8_TTS_STREAM_CONTEXT_FRAMES`, and
`AUDIO8_TTS_STREAM_GUARD_FRAMES`.

Streaming is a server-side opt-in. Start the service with
`AUDIO8_TTS_STREAM_ENABLED=1` to enable SSE streaming; the default is off, and
requests are answered with the complete audio after generation finishes
(a request-level `"stream": true` is then ignored). Non-streaming remains the
default for maximum throughput and stable memory usage. `response_format` of
`codes`/`codec`/`npy` always returns codec codes without streaming.

Run the smoke test to verify a deployment:

```bash
BASE_URL=http://127.0.0.1:8010 ./sglang_omni/scripts/smoke_test.sh
python3 ./sglang_omni/scripts/stream_smoke_test.py \
  --base-url http://127.0.0.1:8010 \
  --output /tmp/audio8_stream.wav
```

To build the adapter into an existing image, append
[`sglang_omni/Dockerfile.snippet`](sglang_omni/Dockerfile.snippet) after the
SGLang Omni package and its Python dependencies are installed.

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

![Seed-TTS WER and CER comparison](assets/evaluation/seed_tts_error_rates.png)

### CV3 multilingual error rate

| Model | Parameters | zh | en | hard-zh | hard-en | ja | ko | de | es | fr | it | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **3.205** | **3.128** | 10.535 | 5.997 | 7.205 | 4.223 | 3.447 | 3.641 | 8.790 | 4.790 | - |
| Fish S2 Pro | 4.6B | 3.600 | 3.493 | 10.588 | 7.349 | 5.139 | **4.111** | 3.605 | 2.972 | **8.600** | 4.229 | **4.702** |
| Higgs Audio v2 | 4.7B | 3.378 | 3.404 | 10.424 | **5.754** | **4.742** | 4.260 | **3.300** | **2.929** | 9.425 | **3.555** | 5.423 |
| CosyVoice3-1.5B | 1.5B | 3.91 | 4.99 | 9.77 | 10.55 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| VoxCPM2 | 2.3B | 3.65 | 5.00 | **8.55** | 8.48 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |

![CV3 multilingual WER and CER comparison](assets/evaluation/cv3_error_rates.png)

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

## Star History

<a href="https://www.star-history.com/?type=date&repos=Audio8-AI%2FAudio8_TTS">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&theme=dark&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Audio8-AI/Audio8_TTS&type=date&legend=top-left&sealed_token=ShFu9kcwBvymYQ4SjQ_NhkplHrefNRbYVYCiBIvIxnaBLKbEQ1cjHQBs2kZm7K5LNMWpU13JxWgA6zpHvmwh49FokyJ26axmq-0gG8b68Q8IJCyUDZW1jQ" />
 </picture>
</a>
