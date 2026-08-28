# Audio8 0.1B INT8 ONNX Runtime

This directory is a self-contained runtime for the
`Audio8/audio8-TTS-0.1B-ONNX-INT8` export. It is intentionally separate from
the existing `onnx_runtime/` implementation, which targets the incompatible
0.6B INT4 graph.

## Why this runtime is separate

The 0.1B checkpoint uses a Falcon-H1 hybrid slow graph. The graph accepts one
`[1, 11, 1]` token column and carries four recurrent tensors between calls:

```text
cache_keys    [24, 1, 2, 2048, 64]
cache_values  [24, 1, 2, 2048, 64]
conv_states   [24, 1, 896, 4]
ssm_states    [24, 1, 24, 32, 64]
```

The implementation below performs prompt prefill one position at a time,
updates the returned state deltas, maps the compact 4097-way semantic logits,
then runs the four-layer Fast AR codebook graph and the FP16 codec decoder.

## Install

Python 3.10 or newer is supported, including the Python 3.10 environment used
by the Jetson issue report. From this directory:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
python scripts/register_default_voice.py
```

The model directory must contain:

```text
model/
|- slow_ar_int8.onnx(.data)
|- fast_ar_int8.onnx(.data)
|- codec_decoder_fp16.onnx(.data)
|- runtime_manifest.json
|- reference_codes.npy
|- tokenizer/tokenizer.json
`- registration/
   |- codec_encoder_fp16.onnx(.data)        (optional)
   `- registration_manifest.json             (optional)
```

`register_default_voice.py` only uses `reference_codes.npy` and does not load
the optional encoder. To register a new voice from audio, use the service API
after the optional `registration/` files have been downloaded.

## CLI

```bash
bash run_infer.sh \
  --text "这是一个中文测试" \
  --voice default \
  --max-new-tokens 128 \
  --output outputs/test.wav
```

The command also writes `outputs/test.npy` with generated codes shaped
`[10, frames]`.

Set `ARKTTS_MODEL_DIR` or `ARKTTS_VOICES_DIR` to use another location. The
runtime always selects `CPUExecutionProvider`; GPU execution is outside the
scope of this export.

## Local HTTP service

The service includes the same local browser UI as the 0.6B ONNX runtime. Open
`http://127.0.0.1:8024` after startup to enter text, select a voice, play or
download WAV output, register a reference voice, inspect memory, and reload
the runtime. The developer API remains available at `/docs`.

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
curl http://127.0.0.1:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，这是一个测试。","voice_name":"default"}' \
  -o outputs/api.wav
```

The service exposes `/api/tts`, `/api/tts/stream`, `/api/tts/cancel`,
`/api/voices`, `/api/voices/register`, `/api/registration/status`, and the
OpenAI-compatible `/v1/audio/speech`. Stop it with `bash stop_server.sh`.

## Verification

```bash
"$PWD/.venv/bin/python" -m pytest -q tests
```

`test_contract.py` checks the exact input/output names and shapes when
`ARKTTS_MODEL_DIR` points at a downloaded model. Without model files it skips
the ONNX-dependent check; the pure prompt and state-shape tests still run.

The 0.1B model and this runtime are intended for local, authorized voice
generation. Obtain consent before cloning a voice and disclose synthetic audio
where appropriate.
