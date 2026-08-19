<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8 TTS" width="760">

# Audio8 TTS Preview 0.1B

**The smallest zero-shot TTS worth running.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.1b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge)](https://audio8-ai.github.io/Audio8_TTS/)
[![License](https://img.shields.io/badge/Model%20License-CC--BY--NC--4.0-blue?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)

**Chinese / English first · experimental multilingual support · zero-shot voice cloning**

</div>

Audio8 TTS Preview 0.1B is a compact audio-language model for speech generation
and zero-shot voice cloning. It packs the complete v4 mixed checkpoint, neural
audio codec, tokenizer, processor, and Hugging Face remote code into a stack
small enough to run on modest GPU hardware.

The checkpoint lives on Hugging Face:

```text
Audio8/Audio8-TTS-Preview-0.1b
```

This GitHub branch provides the matching Falcon H1 training recipes and batch
inference tools. The former SGLang and ONNX runtimes have been removed because
they target a different model family and are not adapted to this checkpoint.

## Why 0.1B?

The defining characteristic of this release is size. The main generative model
has approximately **170M parameters**. Its bundled neural codec decoder adds
approximately **120M parameters**, so the complete audio generation stack is
still substantially smaller than most modern multilingual TTS systems.

| Model | Reported main-model scale |
|---|---:|
| **Audio8 TTS Preview 0.1B** | **~0.17B** |
| Audio8 TTS Preview 0.6B | ~0.6B |
| IndexTTS2.5 | ~0.8B |
| CosyVoice3 | ~1.5B |
| VoxCPM2 | ~2.3B |
| Fish S2 Pro | ~4.6B |
| Higgs Audio v2 | ~4.7B |
| MOSS-TTS | ~8.5B |

These are approximate reference scales from the respective model reports, not
a strictly matched parameter-count audit.

## Supported languages

- **Primary:** Chinese and English
- **Experimental:** German, Spanish, French, Italian, Japanese, and Korean

Chinese and English are the recommended production targets. Other languages are
usable for evaluation, but generally have weaker and more variable quality.

## Architecture

Audio8 uses a Falcon H1 style dual-autoregressive architecture:

1. The slow AR branch predicts semantic audio tokens.
2. The fast AR branch predicts codec codebooks conditioned on the slow hidden
   state.
3. The bundled codec decodes generated tokens into 44.1 kHz waveforms.

| Component | Configuration |
|---|---|
| Main model | ~170M parameters, excluding codec decoder |
| Slow AR | 24 layers, width 512, 8 attention heads, 2 KV heads |
| Fast AR | 4 layers, width 512, 8 attention heads, 2 KV heads |
| Acoustic tokens | 10 codebooks, 4,096 entries per codebook |
| Codec | 44.1 kHz, 2,048 samples per frame (~21.5 frames/s) |
| Codec decoder | ~120M parameters, bundled as `codec.pth` |
| Context | Up to 2,048 packed text/audio positions |

## Installation

Python 3.11 or newer and a CUDA-capable GPU are recommended. Install PyTorch
with the CUDA build matching your machine, then install inference dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
```

Training additionally requires a Fish Speech checkout for tokenizer and
conversation utilities. Clone or mount it outside this repository and set
`FISH_SPEECH_ROOT` in `.env`.

## Inference

### Direct Transformers usage

The model includes custom Transformers code. Load it with
`trust_remote_code=True`:

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

For synthesis without voice cloning, omit `reference_audio` and
`reference_text`.

### Batch inference from JSONL

Create one JSON object per line:

```json
{"id":"clone_zh","text":"这是一个零样本语音合成测试。","reference_audio":"/data/ref.wav","reference_text":"参考音频对应的完整文本。"}
{"id":"codes_en","text":"This is an Audio8 Falcon H1 inference test.","reference_codes":"/data/ref_codes.npy","reference_text":"Reference transcript."}
{"id":"no_reference","text":"This sample uses the model's non-cloning generation mode."}
```

Then run:

```bash
MODEL=Audio8/Audio8-TTS-Preview-0.1b \
INPUT_JSONL=/data/prompts.jsonl \
OUTPUT_DIR=/data/generated \
bash scripts/infer/run_batch.sh
```

`MODEL` can be a Hugging Face model ID or a local exported model directory.
The batch runner writes WAV files and Audio8 codec-token NPY files.

## Training

This repository reproduces the v4 mixed training route:

```text
AUDIO8_INIT_MODEL → v1 SlowAR → v2 FastAR → v3 Joint → v4 Mixed
```

```bash
cp .env.example .env
# Configure model, data, Fish Speech, storage, and cluster paths.
bash scripts/utils/preflight.sh

bash scripts/train/v1_slowar.sh
bash scripts/train/v2_fastar.sh
bash scripts/train/v3_joint.sh
bash scripts/train/v4_mixed.sh
```

All launchers expose environment-variable controls for manifests, model paths,
topology, batch size, learning rate, ports, checkpointing, and resume behavior.
The supplied defaults reproduce a three-node, eight-GPU-per-node recipe; first
validate memory and throughput with a small run.

Optional GRPO training is available through `scripts/train/grpo.sh`. See:

- [Environment and cluster requirements](docs/0.1B/ENVIRONMENT.md)
- [Training stages and resume semantics](docs/0.1B/TRAINING.md)
- [Dataset and manifest contracts](docs/0.1B/DATA.md)

## Evaluation

On CV3, the 0.1B checkpoint reaches **3.619% Chinese error rate** and
**3.307% English error rate**, while remaining around one third the size of the
0.6B release. On Seed-TTS, it reaches **1.662% EN WER / 56.7 SIM** and
**1.13% ZH CER / 68.2 SIM**.

The complete multilingual tables, protocol notes, and baseline citations are
maintained in the
[model card](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b#evaluation).
Cross-project TTS evaluations use different normalizers and evaluators, so
treat them as reference comparisons rather than a strict ranking.

## Responsible use

- Obtain consent before cloning a voice.
- Disclose synthetic audio where appropriate.
- Avoid noisy, very long, or incorrectly transcribed reference clips.
- Evaluate accuracy, speaker similarity, safety, and legal compliance before
  deploying the model in a production setting.

## License

- **Model checkpoint and Hugging Face remote code:** CC-BY-NC-4.0
- **Training and inference code in this repository:** Apache-2.0
- **Fish Speech and other third-party assets:** governed by their own licenses;
  see [`NOTICE`](NOTICE) and
  [`third_party/FISH_SPEECH_LICENSE`](third_party/FISH_SPEECH_LICENSE).
