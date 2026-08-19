# Data contracts

## v1 and v2 JSONL

Each non-empty line is a JSON object:

```json
{
  "text": "target transcript",
  "fish_audio_ids_path": "/data/codes/target.npy",
  "pair_text": "reference transcript",
  "pair_fish_audio_ids_path": "/data/codes/reference.npy"
}
```

Codec arrays are integer NumPy files with shape `[10, T]`, `T > 0`, and values in `[0, 4095]`.
Row zero is the semantic codebook; rows 1-9 feed FastAR. v1/v2 build one rank shard per distributed
rank at `<TRAIN_JSONL>.shards<WORLD_SIZE>/`.

## v3 and v4 mixed shards

These stages consume balanced `rank_*/worker_*/chunk_*.npz` shards through
`audio8_code_shard_dataset.py`. `counts.txt` must be readable, and `DATALOADER_NUM_WORKERS` must
match the worker layout used when the shards were produced. The manifest contains chunk pointers
and associated text/reference metadata. v4 mixed is expected to combine full-regeneration and
text-tail data before launch.

## GRPO JSONL

GRPO requires `text`, `pair_text`, and `pair_fish_audio_ids_path`; a target code path is optional.
Language metadata must agree with the reward router: Chinese uses ARK-ASR and supported non-Chinese
languages use Whisper. Validate every language code before a long run.

## Inference JSONL

Required fields are `id` and `text`. A row may include:

- `reference_audio` plus `reference_text`; or
- `reference_codes` plus `reference_text`; or
- no reference fields for non-cloning synthesis.

Relative paths are resolved by the inference script process, so absolute paths are recommended for
cluster jobs. IDs are sanitized before being used as output filenames.
