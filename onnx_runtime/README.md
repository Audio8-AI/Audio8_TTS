# Audio8 TTS ONNX Runtime

CPU-only ONNX Runtime inference for
[Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4),
including CLI inference, a web and HTTP service, streaming PCM output, and
reference-voice registration. The runtime does not require PyTorch,
Transformers, or Hugging Face Hub after the model files have been downloaded.

Chinese documentation: [README_zh.md](README_zh.md)

## Precision and resource use

| Component | Precision |
|---|---|
| Slow/Fast AR weights | Weight-only INT4 |
| Activations, hidden states, and KV cache | FP16 |
| Codec encoder and decoder | FP16 |
| Waveform output | FP32 |

Normal synthesis loads only the Slow AR, Fast AR, and codec decoder sessions.
On a 16 GB Apple M2 MacBook Air with five ONNX Runtime threads, the service
used about 1004 MiB after loading and approximately 1.1-1.2 GiB at synthesis
peak. Voice registration unloads those sessions before loading the encoder;
the measured registration peak was approximately 1.55 GiB. Measurements vary
by operating system and ONNX Runtime allocator behavior.

The online model files occupy about 572 MiB. The optional voice-registration
encoder brings the complete download to about 968 MiB.

## Install

Python 3.11 or newer is required. The current release is tested on macOS arm64
with `CPUExecutionProvider`.

```bash
git clone https://github.com/Audio8-AI/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime

python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 --local-dir model
bash setup.sh
```

The downloaded directory must have this layout:

```text
model/
|- slow_ar_int4.onnx(.data)
|- fast_ar_int4.onnx(.data)
|- codec_decoder_fp16.onnx(.data)
|- runtime_manifest.json
|- tokenizer/tokenizer.json
`- registration/
   |- codec_encoder_fp16.onnx(.data)
   `- registration_manifest.json
```

Set `ARKTTS_MODEL_DIR` if the model is stored elsewhere. Voice profiles are
stored in `voices/` by default; override this with `ARKTTS_VOICES_DIR`.

## Register a voice

A voice profile contains codec codes extracted from a reference recording and
its exact transcript. The recording must be 0.5-30 seconds, no larger than
50 MiB, and readable by libsndfile. The service converts it to mono and 44.1
kHz automatically.

Start the local service and open <http://127.0.0.1:8024>. If there are no voice
profiles, the page opens the registration view first.

```bash
bash start_server.sh
```

Registration is also available over HTTP:

```bash
curl http://127.0.0.1:8024/api/voices/register \
  -F 'audio=@/absolute/path/reference.wav' \
  -F 'text=The exact transcript of the reference recording.' \
  -F 'name=speaker_a' \
  -F 'overwrite=false'
```

The encoder is loaded only during registration. The online inference runtime
is released first and restored after registration completes.

## Inference

Command line:

```bash
bash run_infer.sh \
  --text "Welcome to Audio8 TTS ONNX Runtime." \
  --voice speaker_a \
  --max-new-tokens 256 \
  --output outputs/example.wav
```

The command also writes `outputs/example.npy` containing generated codec codes
with shape `[10, T]`.

HTTP WAV response:

```bash
curl http://127.0.0.1:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Welcome to Audio8 TTS.","voice_name":"speaker_a","max_new_tokens":256}' \
  -o outputs/api.wav
```

OpenAI-compatible WAV response:

```bash
curl http://127.0.0.1:8024/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"arktts","input":"Welcome to Audio8 TTS.","voice":"speaker_a","response_format":"wav"}' \
  -o outputs/openai.wav
```

`POST /api/tts/stream` returns newline-delimited JSON. Audio chunks contain
base64-encoded signed 16-bit little-endian mono PCM at 44.1 kHz. Cancel the
active stream with `POST /api/tts/cancel`.

The service serializes synthesis and registration requests to bound memory
use. It is intended for local use and low-concurrency CPU deployments.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ARKTTS_MODEL_DIR` | `onnx_runtime/model` | Downloaded Hugging Face model |
| `ARKTTS_VOICES_DIR` | `onnx_runtime/voices` | Registered voice profiles |
| `ARKTTS_REGISTRATION_DIR` | `$ARKTTS_MODEL_DIR/registration` | Codec encoder directory |
| `ARKTTS_THREADS` | `5` | ONNX Runtime CPU threads |
| `HOST` | `127.0.0.1` | Service listen address |
| `PORT` | `8024` | Service listen port |

Stop a managed background service with `bash stop_server.sh`. Logs are written
to `service.log`.

## Responsible use

The reference transcript must match the spoken content. Noisy, long, or
incorrectly transcribed references can reduce stability and speaker
similarity. Obtain consent before cloning a voice and disclose synthetic audio
where appropriate. Evaluate accuracy, safety, and legal compliance before
deployment.

The code and model weights are released under the Apache License 2.0. See the
repository [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
