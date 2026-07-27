#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-${PROJECT_ROOT}/model/audio8_tts_0_6B_preview}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
EVAL_JSONL="${EVAL_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/audio8_tts_sft}"
EXPORT_DIR="${EXPORT_DIR:-${OUTPUT_DIR}/export}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${TRAIN_JSONL}" ]]; then
  echo "TRAIN_JSONL is required." >&2
  echo "Example: TRAIN_JSONL=data/train.prepared.jsonl bash audio8_tts_sft.sh" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

data_args=(--train_jsonl "${TRAIN_JSONL}")
training_args=(--do_train true)
if [[ -n "${EVAL_JSONL}" ]]; then
  data_args+=(--eval_jsonl "${EVAL_JSONL}")
  training_args+=(--do_eval true --eval_strategy steps --eval_steps "${EVAL_STEPS:-500}")
fi

"${PYTHON}" -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${PROJECT_ROOT}/audio8_tts_sft.py" \
  --model_name_or_path "${MODEL}" \
  "${data_args[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_dir "${EXPORT_DIR}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --warmup_ratio "${WARMUP_RATIO:-0.01}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --bf16 "${BF16:-true}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
  --freeze_slow_ar "${FREEZE_SLOW_AR:-false}" \
  --freeze_fast_ar "${FREEZE_FAST_AR:-false}" \
  --resume_mode "${RESUME_MODE:-none}" \
  --report_to "${REPORT_TO:-tensorboard}" \
  --remove_unused_columns false \
  --deepspeed "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}" \
  "${training_args[@]}" \
  "$@"
