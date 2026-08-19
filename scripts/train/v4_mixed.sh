#!/usr/bin/env bash
# Stage-4 (mixed): joint fine-tune of Audio8 Falcon-H1 dual-AR on
#   20M regen clean + text_tail_v1 regenerated data (chunk-npz layout).
# Same as v4 but data comes from --code_shard_dir (v3 style):
#   CODE_SHARD_DIR = balanced rank_*/worker_*/chunk_*.npz (both datasets)
#   TRAIN_JSONL    = merged chunk-pointer manifest (train_mixed_v4.jsonl)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "$PROJECT_ROOT"

PID_FILE="${PID_FILE:-train_formal_3node_audio8_falcon_joint_v4_mixed.pid}"
printf '%s\n' "$$" > "$PID_FILE"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export USE_LIBUV=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0,bond0,enp,ens,eno
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TMPDIR="${TMPDIR:-/dev/shm/fish_s2pro_tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$TMPDIR/torch_extensions}"
export FISH_DISABLE_EMBED_SCALE="${FISH_DISABLE_EMBED_SCALE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cat > .deepspeed_env <<EOF
PYTHONPATH=$PYTHONPATH
TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM
OMP_NUM_THREADS=$OMP_NUM_THREADS
MKL_NUM_THREADS=$MKL_NUM_THREADS
USE_LIBUV=$USE_LIBUV
NCCL_DEBUG=$NCCL_DEBUG
NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER
TMPDIR=$TMPDIR
TEMP=$TEMP
TMP=$TMP
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
FISH_DISABLE_EMBED_SCALE=$FISH_DISABLE_EMBED_SCALE
PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
EOF
chmod 600 .deepspeed_env

NUM_NODES="${NUM_NODES:-3}"
NUM_GPUS="${NUM_GPUS:-8}"
WORLD_SIZE="$((NUM_NODES * NUM_GPUS))"
HOSTFILE="${HOSTFILE:?Set HOSTFILE to a DeepSpeed hostfile}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "$HOSTFILE")"
fi
MASTER_PORT="${MASTER_PORT:-29639}"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "No hosts found in hostfile: $HOSTFILE" >&2
  exit 1
fi

CODE_SHARD_DIR="${CODE_SHARD_DIR:?Set CODE_SHARD_DIR to mixed v4 shards}"
TRAIN_JSONL="${TRAIN_JSONL:?Set TRAIN_JSONL to the mixed v4 manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/v4_mixed}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/v4_mixed}"
MODEL_PATH="${MODEL_PATH:-$EXPORT_ROOT/v3_joint}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-$DATALOADER_NUM_WORKERS}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
RESUME_MODE="${RESUME_MODE:-none}"

require_file "$TRAIN_JSONL"
require_file "$HOSTFILE"
require_dir "$CODE_SHARD_DIR"

if [[ "$WORKERS_PER_RANK" != "$DATALOADER_NUM_WORKERS" ]]; then
  echo "[joint-v4-mixed] WORKERS_PER_RANK ($WORKERS_PER_RANK) must equal DATALOADER_NUM_WORKERS ($DATALOADER_NUM_WORKERS)" >&2
  exit 1
fi
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "No stage-3 model found (export: $MODEL_PATH)" >&2
  exit 1
fi
echo "Stage-4-mixed starting from $MODEL_PATH"

MIN_RANK_SAMPLES="$("$PYTHON" \
  "$PROJECT_ROOT/scripts/utils/count_audio8_rank_rows.py" \
  --code-shard-dir "$CODE_SHARD_DIR" --num-workers "$DATALOADER_NUM_WORKERS" \
  | sed -n 's/^min_rank_samples=//p')"
if [[ -z "$MIN_RANK_SAMPLES" || "$MIN_RANK_SAMPLES" -le 0 ]]; then
  echo "Failed to compute min rank samples from $CODE_SHARD_DIR" >&2
  exit 1
fi
echo "Training $MIN_RANK_SAMPLES samples on each of $WORLD_SIZE ranks, $NUM_TRAIN_EPOCHS epochs; remainder is dropped"

pids=()
while read -r host rest; do
  if [[ -n "$host" && "$host" != \#* ]]; then
    ssh "$host" "
      set -euo pipefail
      mkdir -p '$TMPDIR' '$TORCH_EXTENSIONS_DIR'
      chmod 1777 '$TMPDIR'
    " &
    pids+=("$!")
  fi
done < "$HOSTFILE"
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[joint-v4-mixed] node preparation failed" >&2
  exit "$status"
fi

DATALOADER_ARGS=(--dataloader_num_workers "$DATALOADER_NUM_WORKERS")
if [[ "$DATALOADER_NUM_WORKERS" -gt 0 ]]; then
  DATALOADER_ARGS+=(--dataloader_prefetch_factor 2 --dataloader_persistent_workers true)
fi

"$PYTHON" -m deepspeed.launcher.runner \
  --hostfile "$HOSTFILE" \
  --num_nodes "$NUM_NODES" \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$PROJECT_ROOT/src/train_audio8_falcon_h1_ds.py" \
  --pretrained_ckpt_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --code_shard_dir "$CODE_SHARD_DIR" \
  --max_train_samples "$MIN_RANK_SAMPLES" \
  --num_codebooks 10 \
  --text_key text \
  --ref_text_key reference_text \
  --use_ref true \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-48}" \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --learning_rate "$LEARNING_RATE" \
  --freeze_fast_ar false \
  --freeze_slow_ar false \
  --slow_ar_only false \
  --base_loss_weight 1.0 \
  --base_loss_weight_final 1.0 \
  --warmup_ratio 0.05 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --logging_steps 10 \
  --report_to tensorboard \
  --logging_dir "$OUTPUT_DIR/tensorboard" \
  --output_dir "$OUTPUT_DIR" \
  --export_dir "$EXPORT_DIR" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --eval_strategy no \
  --do_train \
  --bf16 true \
  --logging_first_step true \
  --resume_mode "$RESUME_MODE" \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --weight_decay 0.0 \
  "${DATALOADER_ARGS[@]}" \
  --remove_unused_columns false \
  --deepspeed "$PROJECT_ROOT/configs/deepspeed/zero2_fp32comm.json" \
  2>&1 | sed -u '/\[data-cache\]/d'
