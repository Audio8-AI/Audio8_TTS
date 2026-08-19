# Audio8 TTS 0.1B

Audio8-TTS-0.1B is a compact zero-shot voice-cloning TTS model. This release
branch contains the Falcon H1 multi-stage SFT, optional GRPO, and Hugging Face
batch inference code. SGLang and ONNX runtimes are intentionally absent: they
were built for a different checkpoint family and are not adapted to the 0.1B
Falcon H1 architecture.

## Installation

The training recipes were validated on Python 3.11, PyTorch 2.8.0 (CUDA 12.8),
Transformers 4.57.6, and DeepSpeed 0.18.4. Install PyTorch with the CUDA build
matching your machine, then install the locked Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
```

SFT requires a Fish Speech checkout for its tokenizer/conversation utilities.
Clone or mount it outside this repository and set `FISH_SPEECH_ROOT` in `.env`.
See [`docs/0.1B/ENVIRONMENT.md`](docs/0.1B/ENVIRONMENT.md) and
[`docs/0.1B/DATA.md`](docs/0.1B/DATA.md) for cluster and data contracts.

```bash
cp .env.example .env
# Edit paths and distributed topology.
bash scripts/utils/preflight.sh
```

## Training

The reproducible stage chain is:

```text
AUDIO8_INIT_MODEL -> v1_slowar -> v2_fastar -> v3_joint -> v4_mixed
```

Run the stages in order:

```bash
bash scripts/train/v1_slowar.sh
bash scripts/train/v2_fastar.sh
bash scripts/train/v3_joint.sh
bash scripts/train/v4_mixed.sh
```

Every launcher accepts environment-variable overrides for model paths, manifests,
batch size, learning rate, topology, ports, and resume behavior. Production
defaults were tuned for a three-node, eight-GPU-per-node cluster; validate memory
and throughput with a small run first. Details are in
[`docs/0.1B/TRAINING.md`](docs/0.1B/TRAINING.md).

Optional GRPO training is available through `scripts/train/grpo.sh`. It requires
the additional ASR and speaker-similarity assets listed in the environment guide.

## Inference

Batch inference uses the exported Hugging Face package with its bundled remote
code; it does not require SGLang or ONNX. Each input manifest line needs `id`
and `text`, plus either `reference_audio`/`reference_text` or
`reference_codes`/`reference_text` for voice cloning:

```json
{"id":"clone","text":"Welcome to Audio8.","reference_audio":"/data/ref.wav","reference_text":"Reference transcript."}
{"id":"no_ref","text":"This sample does not use a reference voice."}
```

```bash
MODEL=/path/to/audio8-0.1B-export \
INPUT_JSONL=/data/prompts.jsonl \
OUTPUT_DIR=/data/generated \
bash scripts/infer/run_batch.sh
```

`MODEL` may also be a Hugging Face model ID. The script writes WAV files and
Audio8 codec-token NPY files to `OUTPUT_DIR`.

## Repository layout

```text
configs/deepspeed/       ZeRO-2 training configurations
docs/0.1B/               Environment, data, and training contracts
examples/                Hostfile and inference manifest examples
scripts/train/           v1-v4 SFT and GRPO launchers
scripts/infer/           Hugging Face batch inference
scripts/utils/           Preflight, sharding, and row-count tools
src/                     Datasets, trainers, and reward worker
```

## License

Apache-2.0 for repository code. Fish Speech, datasets, checkpoints, and reward
models remain governed by their separate licenses; see `NOTICE` and
`third_party/FISH_SPEECH_LICENSE`.
