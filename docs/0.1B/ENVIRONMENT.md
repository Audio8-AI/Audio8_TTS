# Environment

## Validated stack

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchaudio | 2.8.0 |
| CUDA build | 12.8 |
| Transformers | 4.57.6 |
| DeepSpeed | 0.18.4 |
| NumPy | 2.2.6 |
| safetensors | 0.7.0 |

PyTorch wheels must match the cluster driver and CUDA runtime. The plain requirement file records
the verified Python version but cannot choose the correct CUDA index for every machine. For a new
cluster, install PyTorch from the official CUDA-specific index first and then install the remaining
requirements.

## External repositories and assets

`FISH_SPEECH_ROOT` must point to a checkout containing `fish_speech/`. Pin its commit in every run
record. `AUDIO8_INIT_MODEL` is the loadable Hugging Face directory used by v1. Later stages consume
the previous stage's export.

GRPO additionally requires:

| Variable | Contents |
| --- | --- |
| `SEEDTTS_ROOT` | Seed-TTS evaluation checkout/assets and `wavlm_large_finetune.pth` |
| `OMNIVOICE_REPO` | OmniVoice checkout and downloaded speaker-sim models |
| `WHISPER_PATH` | Local Whisper-large-v3 Hugging Face model |
| `ARK_ASR_PATH` | Local ARK-ASR model used for Chinese |
| `REWARD_EXTRA_PYTHONPATH` | Optional reward-only site-packages path |

These assets are not redistributable merely because this repository is Apache-2.0 licensed.

## Cluster requirements

- Identical project, dependency and model paths on every node.
- Shared access to manifests, code shards and output directories.
- Passwordless SSH from the launcher to every hostfile entry.
- Consistent NVIDIA driver, CUDA, NCCL and network interface names.
- Sufficient `/dev/shm` or an explicit `RUNTIME_ROOT` on local fast storage.
- Open `MASTER_PORT` between workers.

DeepSpeed passes selected variables through a generated environment file. It contains credentials
when object storage is enabled and is created with mode `0600`; never commit or archive it.
