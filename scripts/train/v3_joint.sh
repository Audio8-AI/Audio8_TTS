#!/usr/bin/env bash
# Stage-3: joint fine-tune of the Audio8 Falcon-H1 dual-AR model.
#  - data: 20M Audio8 distillation codec dataset (chunk npz layout), same as
#    train_audio8_3node_fish_ids_full.sh
#  - model: stage-2 fast-AR export (auto-detected below; fallback to newest
#    stage-2 checkpoint, materialized into a complete model folder)
#  - all parameters unfrozen: freeze_fast_ar=false, freeze_slow_ar=false,
#    slow_ar_only=false, base_loss_weight=1.0 (slow + fast AR losses jointly)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "$PROJECT_ROOT"

PID_FILE="${PID_FILE:-train_formal_3node_audio8_falcon_joint_v3.pid}"
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
MASTER_PORT="${MASTER_PORT:-29638}"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "No hosts found in hostfile: $HOSTFILE" >&2
  exit 1
fi

RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-}"
CODE_SHARD_DIR="${CODE_SHARD_DIR:?Set CODE_SHARD_DIR to balanced v3 shards}"
TRAIN_JSONL="${TRAIN_JSONL:?Set TRAIN_JSONL to the v3 manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/v3_joint}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/v3_joint}"
STAGE1_EXPORT_DIR="${STAGE1_EXPORT_DIR:-$EXPORT_ROOT/v1_slowar}"
STAGE2_EXPORT_DIR="${STAGE2_EXPORT_DIR:-$EXPORT_ROOT/v2_fastar}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$OUTPUT_ROOT/v2_fastar}"
MODEL_PATH="${MODEL_PATH:-}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-$DATALOADER_NUM_WORKERS}"
PREPARE_RANK_SHARDS="${PREPARE_RANK_SHARDS:-false}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
RESUME_MODE="${RESUME_MODE:-none}"
BALANCE_CODE_SHARDS_SCRIPT="${BALANCE_CODE_SHARDS_SCRIPT:-}"
BALANCE_STATS_PATH="${BALANCE_STATS_PATH:-$OUTPUT_ROOT/v3_balance_stats.json}"

require_file "$TRAIN_JSONL"
require_file "$HOSTFILE"

if [[ "$WORKERS_PER_RANK" != "$DATALOADER_NUM_WORKERS" ]]; then
  echo "[joint-v3] WORKERS_PER_RANK ($WORKERS_PER_RANK) must equal DATALOADER_NUM_WORKERS ($DATALOADER_NUM_WORKERS)" >&2
  exit 1
fi

# Resolve the stage-2 model: prefer the finished export, otherwise materialize
# the newest stage-2 checkpoint into a complete loadable model folder.
if [[ -z "$MODEL_PATH" ]]; then
  if [[ -f "$STAGE2_EXPORT_DIR/model.safetensors" ]]; then
    MODEL_PATH="$STAGE2_EXPORT_DIR"
  else
    CKPT="$(ls -d "$STAGE2_OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$CKPT" && -f "$CKPT/model.safetensors" ]]; then
      INIT_DIR="$EXPORT_DIR/init_from_$(basename "$CKPT")"
      if [[ ! -f "$INIT_DIR/model.safetensors" ]]; then
        mkdir -p "$INIT_DIR"
        for f in README.md __init__.py chat_template.jinja codec.pth configuration_arktts.py \
                 modeling_arktts.py modeling_arktts_codec.py processing_arktts.py \
                 preprocessor_config.json processor_config.json special_tokens_map.json \
                 tokenizer.json tokenizer_config.json; do
          cp -a "$STAGE1_EXPORT_DIR/$f" "$INIT_DIR/" 2>/dev/null || true
        done
        cp "$CKPT/model.safetensors" "$CKPT/config.json" "$CKPT/generation_config.json" "$INIT_DIR/"
      fi
      MODEL_PATH="$INIT_DIR"
    fi
  fi
fi
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "No stage-2 model found (export: $STAGE2_EXPORT_DIR, output: $STAGE2_OUTPUT_DIR)" >&2
  exit 1
fi
echo "Stage-3 starting from $MODEL_PATH"

if [[ "$PREPARE_RANK_SHARDS" == "true" || "$PREPARE_RANK_SHARDS" == "1" ]]; then
  require_file "$BALANCE_CODE_SHARDS_SCRIPT"
  require_dir "$RAW_OUTPUT_DIR"
  mkdir -p "$CODE_SHARD_DIR"
  "$PYTHON" \
    "$BALANCE_CODE_SHARDS_SCRIPT" \
    "$RAW_OUTPUT_DIR" "$CODE_SHARD_DIR" \
    "$BALANCE_STATS_PATH" \
    --workers "$WORKERS_PER_RANK"
fi
require_dir "$CODE_SHARD_DIR"

MIN_RANK_SAMPLES="$("$PYTHON" \
  "$PROJECT_ROOT/scripts/utils/count_audio8_rank_rows.py" \
  --code-shard-dir "$CODE_SHARD_DIR" --num-workers "$DATALOADER_NUM_WORKERS" \
  | sed -n 's/^min_rank_samples=//p')"
if [[ -z "$MIN_RANK_SAMPLES" || "$MIN_RANK_SAMPLES" -le 0 ]]; then
  echo "Failed to compute min rank samples from $CODE_SHARD_DIR" >&2
  exit 1
fi
echo "Training $MIN_RANK_SAMPLES samples on each of $WORLD_SIZE ranks; remainder is dropped"

pids=()
while read -r host rest; do
  if [[ -n "$host" && "$host" != \#* ]]; then
    ssh "$host" "
      set -euo pipefail
      mkdir -p '$TMPDIR' '$TORCH_EXTENSIONS_DIR'
      chmod 1777 '$TMPDIR'
      test -r '$CODE_SHARD_DIR/counts.txt' 2>/dev/null || true
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
  echo "[joint-v3] node preparation failed" >&2
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
  --output_dir "$OUTPUT_DIR" \
  --export_dir "$EXPORT_DIR" \
  --max_train_samples "$MIN_RANK_SAMPLES" \
  --num_codebooks 10 \
  --text_key text \
  --ref_text_key reference_text \
  --use_ref true \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
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
