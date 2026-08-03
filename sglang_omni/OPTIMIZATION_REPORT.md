# SGLang Omni Inference Optimization Report

## Overview

This update optimizes the Audio8 TTS SGLang Omni serving path for low-latency
single-stream inference while retaining the existing dynamic batching path. The
implementation was validated on an NVIDIA H20 with BF16 weights and CUDA Graph.
No CUDA, PyTorch, or model checkpoint changes are required.

## Code Changes

### Reduced semantic projection

Decode-time semantic logits are now computed only for the 4,096 valid semantic
tokens plus EOS. The previous implementation projected to all 155,776 tokenizer
entries and masked invalid entries afterward.

The reduced logits use a local candidate index during sampling and map the
selected result back to the original token ID. The full vocabulary projection
remains available before the Audio8 decode buffers are initialized.

### Fused fast decoder operations

- Fast decoder RMSNorm now uses the fused SGLang RMSNorm kernel. The wrapper
  flattens `[batch, 1, hidden]` to the two-dimensional input expected by the
  kernel and restores the original shape afterward.
- The fast decoder gate and up projections are merged into one linear layer
  after checkpoint loading. Existing `w1` and `w3` weights are concatenated
  once during service initialization.
- Checkpoint loading remains compatible with the original parameter names
  because fusion occurs only after all weights have been loaded.

### Optional Torch compilation

Two environment variables expose SGLang's existing `torch.compile` support:

```text
AUDIO8_TTS_ENABLE_TORCH_COMPILE=0|1
AUDIO8_TTS_TORCH_COMPILE_MAX_BS=<optional positive integer>
```

Compilation is disabled by default. When enabled, the adapter preserves
SGLang's native compile batch-size policy instead of imposing an Audio8-specific
limit. `AUDIO8_TTS_TORCH_COMPILE_MAX_BS` is an optional override for deployments
that need to bound compilation time or memory use. Batches outside the compiled
set continue through SGLang's existing dynamic batching path. The first startup
performs compilation and kernel autotuning; later startups reuse
`TORCHINDUCTOR_CACHE_DIR`.

### Optional static greedy path

```text
AUDIO8_TTS_GREEDY_FASTPATH=0|1
```

When enabled, sampling is replaced by a direct argmax and all top-k, top-p, and
random sampling work is removed from the captured graph. This is a server-wide
mode and must only be enabled when every request is intended to use greedy
decoding. It overrides per-request sampling settings.

The CV3 quality evaluations in this report used
`AUDIO8_TTS_GREEDY_FASTPATH=0`.

### Portable Blackwell attention path

`AUDIO8_TTS_ATTENTION_BACKEND` controls both the SGLang slow-AR backend and the
fast-head cache implementation:

```text
AUDIO8_TTS_ATTENTION_BACKEND=fa3         # default for Hopper/H20
AUDIO8_TTS_ATTENTION_BACKEND=flashinfer  # consumer Blackwell
```

The default path retains `flash_attn_with_kvcache`. When a non-`fa3` backend is
selected, the fast head writes K/V at the same fixed-cache positions in place
and uses SDPA with an explicit valid-prefix mask and GQA. Shapes remain static
for CUDA Graph capture. This avoids the Hopper-only FA3 custom op on `sm_120`
without changing the Hopper path.

When the variable is unset, the backend is resolved by
`models/audio8_tts/attention_backend.py`. Devices whose compute capability has
no FA3 kernel image — currently `(12, 0)`, consumer Blackwell — default to
`flashinfer`; every other device keeps `fa3`. The probe is cached and runs once
per process, and an explicit `AUDIO8_TTS_ATTENTION_BACKEND` always takes
precedence, so Hopper deployments are unaffected.

For consumer Blackwell deployment, `sgl_kernel` also requires the system
`libnuma` library. The CUDA toolkit `bin` directory must be on `PATH`, and
`CUDA_PATH` may be required by `deep_gemm` JIT. Transformers must remain in the
supported `>=4.57.0,<5` range.

## Recommended Service Configuration

```bash
PATH=/path/to/ninja/bin:/usr/local/cuda/bin:${PATH} \
TORCHINDUCTOR_CACHE_DIR=/data/yumu/audio8_sglang_perf/torchinductor \
CUDA_VISIBLE_DEVICES=0 \
AUDIO8_TTS_MEM_FRACTION_STATIC=0.10 \
AUDIO8_TTS_MAX_RUNNING_REQUESTS=4 \
AUDIO8_TTS_ATTENTION_BACKEND=fa3 \
AUDIO8_TTS_ENABLE_TORCH_COMPILE=1 \
AUDIO8_TTS_GREEDY_FASTPATH=0 \
sglang_omni/scripts/run_server.sh
```

`ninja` is needed only for first-time TorchInductor compilation. It does not
change the CUDA runtime.

## Performance

Environment:

```text
GPU: NVIDIA H20
PyTorch: 2.9.1+cu128
SGLang: 0.5.8
SGLang Omni: 0.1.0
Transformers: 4.57.1
Precision: BF16
CUDA Graph batch sizes: 1, 2, 4
```

The benchmark used one fixed Chinese request, 128 generated frames, greedy
decoding, and a 5.85-5.94 second WAV. Cold-start and compilation time were not
included.

| Path | Warm p50 latency | RTF | Relative speed |
|---|---:|---:|---:|
| Original PR service | 1.326 s | 0.227 | 1.00x |
| Optimized service | 0.691 s | 0.116 | 1.92x |

The server was tested with CUDA Graph capture for batch sizes 1, 2, and 4.
Ordinary TTS, reference-voice TTS, and four-request dynamic batching completed
without errors.


## Validation Summary

- CUDA Graph capture succeeded for batch sizes 1, 2, and 4.
- Ordinary and reference-voice HTTP requests returned valid 44.1 kHz WAV files.
- Main CV3 run: 1,124/1,124 files generated, zero failures.
- Temperature 0.7 hard-set run: 124/124 files generated, zero failures.
- WER/CER and speaker-similarity scoring completed for every requested split.
- The portable SDPA cache path was checked on H20 against the full causal
  forward (`cosine=0.99999177` in BF16). CUDA Graph capture succeeded for batch
  sizes 1, 2, and 4, and an HTTP request returned a valid mono 44.1 kHz WAV.
- Consumer Blackwell `sm_120` execution and CUDA Graph capture were reported by
  the issue author; no `sm_120` GPU was available in the local validation
  environment.
