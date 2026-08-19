# Training operations

## Stage semantics

v1 uses `slow_ar_only=true`, freezes FastAR, and optimizes SlowAR cross entropy. v2 starts from the
v1 export, freezes SlowAR, unfreezes FastAR, and sets the SlowAR loss weight to zero. v3 and v4
unfreeze both branches and optimize both losses. v4 in this repository is exclusively the mixed
data recipe.

## Outputs and continuation

Trainer checkpoints are written below `OUTPUT_ROOT`; directly loadable Hugging Face packages are
written below `EXPORT_ROOT`. The normal stage chain is:

```text
AUDIO8_INIT_MODEL -> v1_slowar -> v2_fastar -> v3_joint -> v4_mixed
```

`RESUME_MODE=auto` selects the newest trainer checkpoint. `RESUME_MODE=none` starts a new optimizer
state from `MODEL_PATH`. An explicit checkpoint path or the trainer's `model_only` mode can be used
when intentionally discarding optimizer/scheduler state. Do not switch modes without recording it.

## Important overrides

| Variable | Purpose |
| --- | --- |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | Per-rank microbatch |
| `DATALOADER_NUM_WORKERS` | Loader workers; must match code-shard layout for v3/v4 |
| `LEARNING_RATE` | Stage learning rate |
| `NUM_TRAIN_EPOCHS` | Epoch count |
| `SAVE_STEPS` | Checkpoint interval |
| `MODEL_PATH` | Override previous-stage initialization |
| `MASTER_PORT` | Avoid collisions between concurrent jobs |

The shipped values reproduce the original 3-node, 8-GPU-per-node recipes. They are not safe
defaults for arbitrary GPU memory sizes. Validate sequence length and peak memory with a small run.

## Run record

For each experiment retain: repository commit, Fish Speech commit, container digest, dependency
freeze, sanitized environment, host/GPU topology, manifest checksum, shard counts, random seeds,
DeepSpeed config, TensorBoard logs and final export checksum.
