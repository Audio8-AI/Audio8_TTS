#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=../common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "$PROJECT_ROOT"

PID_FILE="${PID_FILE:-train_formal_3node_audio8_falcon_slowar_v1.pid}"
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

: "${OSS_ENDPOINT:=}"
: "${OSS_REGION:=}"
: "${ARK_OSS_ADDRESSING_STYLE:=virtual}"
: "${OSS_ACCESS_KEY_ID:=}"
: "${OSS_SECRET_ACCESS_KEY:=}"

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
OSS_ENDPOINT=$OSS_ENDPOINT
OSS_REGION=$OSS_REGION
OSS_ACCESS_KEY_ID=$OSS_ACCESS_KEY_ID
OSS_SECRET_ACCESS_KEY=$OSS_SECRET_ACCESS_KEY
ARK_OSS_ADDRESSING_STYLE=$ARK_OSS_ADDRESSING_STYLE
EOF
chmod 600 .deepspeed_env

NUM_NODES="${NUM_NODES:-3}"
NUM_GPUS="${NUM_GPUS:-8}"
WORLD_SIZE="$((NUM_NODES * NUM_GPUS))"
HOSTFILE="${HOSTFILE:?Set HOSTFILE to a DeepSpeed hostfile}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "$HOSTFILE")"
fi
MASTER_PORT="${MASTER_PORT:-29618}"

if [[ -z "$MASTER_ADDR" ]]; then
  echo "No hosts found in hostfile: $HOSTFILE" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${AUDIO8_INIT_MODEL:-}}"
TRAIN_JSONL="${TRAIN_JSONL:?Set TRAIN_JSONL to the v1 manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/v1_slowar}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/v1_slowar}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOCAL_NPY_CACHE_DIR="${LOCAL_NPY_CACHE_DIR:-$CACHE_ROOT/v1_slowar}"
LOCAL_NPY_CACHE_READ_ONLY=false
LOCAL_NPY_CACHE_MAX_FILES="${LOCAL_NPY_CACHE_MAX_FILES:-1000000}"
LOCAL_NPY_CACHE_MAX_GB="${LOCAL_NPY_CACHE_MAX_GB:-20}"
PREPARE_RANK_SHARDS="${PREPARE_RANK_SHARDS:-true}"
RESUME_MODE="${RESUME_MODE:-auto}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LOCAL_NPY_CACHE_SOURCE_PREFIX="${LOCAL_NPY_CACHE_SOURCE_PREFIX:-/}"

require_dir "$MODEL_PATH"
require_file "$TRAIN_JSONL"
require_file "$HOSTFILE"

if [[ "$PREPARE_RANK_SHARDS" == "true" || "$PREPARE_RANK_SHARDS" == "True" || "$PREPARE_RANK_SHARDS" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/utils/shard_jsonl_by_rank.py" \
    --input "$TRAIN_JSONL" \
    --world-size "$WORLD_SIZE"
fi

SHARD_DIR="${TRAIN_JSONL}.shards${WORLD_SIZE}"
if [[ ! -f "$SHARD_DIR/counts.txt" ]]; then
  echo "Missing rank shards: $SHARD_DIR" >&2
  exit 1
fi
RANK_COUNT="$(awk 'NF >= 2 { count += 1 } END { print count + 0 }' "$SHARD_DIR/counts.txt")"
if [[ "$RANK_COUNT" -ne "$WORLD_SIZE" ]]; then
  echo "Expected $WORLD_SIZE rank counts, found $RANK_COUNT: $SHARD_DIR/counts.txt" >&2
  exit 1
fi
MIN_RANK_SAMPLES="$(awk '
  NF >= 2 {
    count = $2 + 0
    if (!seen || count < min_count) min_count = count
    seen = 1
  }
  END {
    if (seen) print min_count
  }
' "$SHARD_DIR/counts.txt")"
if [[ -z "$MIN_RANK_SAMPLES" || "$MIN_RANK_SAMPLES" -le 0 ]]; then
  echo "Invalid rank counts: $SHARD_DIR/counts.txt" >&2
  exit 1
fi
echo "Training $MIN_RANK_SAMPLES samples on each of $WORLD_SIZE ranks; remainder is dropped"

pids=()
while read -r host rest; do
  if [[ -n "$host" && "$host" != \#* ]]; then
    if [[ "$LOCAL_NPY_CACHE_READ_ONLY" == "true" || "$LOCAL_NPY_CACHE_READ_ONLY" == "True" || "$LOCAL_NPY_CACHE_READ_ONLY" == "1" ]]; then
      ssh "$host" "
        set -euo pipefail
        mkdir -p '$TMPDIR' '$TORCH_EXTENSIONS_DIR'
        chmod 1777 '$TMPDIR'
        test -r '$SHARD_DIR/counts.txt'
      " &
    else
      ssh "$host" "
        set -euo pipefail
        mkdir -p '$TMPDIR' '$TORCH_EXTENSIONS_DIR' '$LOCAL_NPY_CACHE_DIR'
        chmod 1777 '$TMPDIR'
        test -r '$SHARD_DIR/counts.txt'
        test -w '$LOCAL_NPY_CACHE_DIR'
      " &
    fi
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
  echo "[cache] node preparation failed" >&2
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
  --output_dir "$OUTPUT_DIR" \
  --export_dir "$EXPORT_DIR" \
  --max_train_samples "$MIN_RANK_SAMPLES" \
  --stream_train_jsonl true \
  --num_codebooks 10 \
  --text_key text \
  --audio_ids_key fish_audio_ids_path \
  --ref_audio_ids_key pair_fish_audio_ids_path \
  --ref_text_key pair_text \
  --local_npy_cache_dir "$LOCAL_NPY_CACHE_DIR" \
  --local_npy_cache_source_prefix "$LOCAL_NPY_CACHE_SOURCE_PREFIX" \
  --local_npy_cache_read_only "$LOCAL_NPY_CACHE_READ_ONLY" \
  --local_npy_cache_rank_subdir false \
  --local_npy_cache_max_files "$LOCAL_NPY_CACHE_MAX_FILES" \
  --local_npy_cache_max_gb "$LOCAL_NPY_CACHE_MAX_GB" \
  --local_npy_cache_delete_on_exit true \
  --local_npy_cache_log_every 5000 \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --learning_rate "$LEARNING_RATE" \
  --freeze_fast_ar true \
  --freeze_slow_ar false \
  --slow_ar_only true \
  --base_loss_weight 1.0 \
  --base_loss_weight_final 1.0 \
  --warmup_ratio 0.05 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --logging_steps 10 \
  --report_to tensorboard \
  --logging_dir "$OUTPUT_DIR/tensorboard" \
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
  --deepspeed "$PROJECT_ROOT/configs/deepspeed/zero2.json" \
  2>&1 | sed -u -e '/\[data-cache\]/d' -e '/\[data-skip\]/d'
