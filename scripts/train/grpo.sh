#!/usr/bin/env bash
# v5: GRPO (ASR + OmniVoice WavLM/ECAPA SIM reward) on the audio8
# Falcon-H1 joint model exported from v3 (export_audio8_falcon_h1_joint_v3_full_regen).
# Training data and all hyperparameters are identical to
# run_fish_slowar_grpo_asr_sim_seedtts_wavlmsim_reward.sh; only the model
# path, run/output names and runtime dirs/ports are changed.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "$PROJECT_ROOT"

# Runtime paths required by the training and reward subprocesses.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export USE_LIBUV=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0,bond0,enp,ens,eno
export TMPDIR="${GRPO_TMPDIR:-$RUNTIME_ROOT/grpo}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
RUNTIME_LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
unset LD_LIBRARY_PATH
export DS_ACCELERATOR=cuda
export DS_ENV_FILE="$TMPDIR/deepspeed.env"

# Python and source paths.
TRAIN_ROOT="$PROJECT_ROOT"

# Distributed training topology. Same 3-node hostfile as the Fast AR job.
NUM_NODES="${NUM_NODES:-3}"
NUM_GPUS="${NUM_GPUS:-8}"
HOSTFILE="${HOSTFILE:?Set HOSTFILE to a DeepSpeed hostfile}"
MASTER_ADDR="${MASTER_ADDR:-$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "$HOSTFILE")}"
MASTER_PORT="${MASTER_PORT:-29642}"

# Seed-TTS training data preparation.
SEEDTTS_ROOT="${SEEDTTS_ROOT:?Set SEEDTTS_ROOT}"
SEEDTTS_EN_META="$SEEDTTS_ROOT/seedtts_testset/en/meta.lst"
SEEDTTS_ZH_META="$SEEDTTS_ROOT/seedtts_testset/zh/meta.lst"
TRAIN_JSONL="${TRAIN_JSONL:?Set TRAIN_JSONL to the GRPO manifest}"
# Model and output paths.
MODEL_PATH="${MODEL_PATH:-$EXPORT_ROOT/v3_joint}"
RUN_NAME="${RUN_NAME:-grpo_audio8_falcon_h1_joint_v5}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"

# Local Fish-code cache.
LOCAL_NPY_CACHE_DIR="${LOCAL_NPY_CACHE_DIR:-$CACHE_ROOT/grpo_ref_codes}"
LOCAL_NPY_CACHE_SOURCE_PREFIX="${LOCAL_NPY_CACHE_SOURCE_PREFIX:-/}"
LOCAL_NPY_CACHE_READ_ONLY=true
LOCAL_NPY_CACHE_RANK_SUBDIR=false
SHARD_TRAIN_BY_RANK=false

# Optimizer and trainer settings.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
SAVE_STEPS="${SAVE_STEPS:-40}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

# GRPO rollout settings.
# rollout_n=8: larger group reduces advantage noise (previous run collapsed
# partly because n=4 plus near-zero rewards amplified random gradients).
GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-4}"
# Cap generation shorter: the previous run drifted toward the 400-token limit
# and WER/CER exploded / SIM collapsed as outputs became long and repetitive.
GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-1024}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-0.8}"
GRPO_TOP_K="${GRPO_TOP_K:-50}"
GRPO_TOP_P="${GRPO_TOP_P:-0.95}"
CODEC_BATCH_SIZE="${CODEC_BATCH_SIZE:-16}"

# ASR + OmniVoice WavLM/ECAPA speaker-similarity reward settings.
GRPO_REWARD_TYPE="${GRPO_REWARD_TYPE:-asr_sim}"
# Linear ASR reward: reward = max(0, 1 - WER/alpha). With alpha=2 the reward
# reaches zero at WER=2 and keeps a uniform gradient across the whole range.
GRPO_REWARD_ALPHA="${GRPO_REWARD_ALPHA:-2.0}"
# Rebalance ASR/SIM: SIM was only 10% and could not pull the policy back once
# ASR reward collapsed. Give SIM 30% and use a linear shape so low-sim outputs
# still receive a meaningful gradient.
GRPO_ASR_WEIGHT="${GRPO_ASR_WEIGHT:-0.3}"
GRPO_SIM_WEIGHT="${GRPO_SIM_WEIGHT:-0.7}"
GRPO_SIM_FLOOR="${GRPO_SIM_FLOOR:-0.0}"
GRPO_SIM_CEIL="${GRPO_SIM_CEIL:-0.85}"
GRPO_SIM_REWARD_SHAPE="${GRPO_SIM_REWARD_SHAPE:-linear}"
GRPO_SIM_REWARD_BETA="${GRPO_SIM_REWARD_BETA:-5.0}"
GRPO_SIM_BACKEND="${GRPO_SIM_BACKEND:-omnivoice}"
GRPO_DECODE_DIR="${GRPO_DECODE_DIR:-$RUNTIME_ROOT/grpo_rewards}"
GRPO_REWARD_TIMEOUT="${GRPO_REWARD_TIMEOUT:-300}"
GRPO_REWARD_KEEP_WAVS="${GRPO_REWARD_KEEP_WAVS:-false}"

# OmniVoice SIM runtime and assets from the Seed-TTS evaluation runbook.
REWARD_PYTHON="${REWARD_PYTHON:-$PYTHON}"
REWARD_EXTRA_PYTHONPATH="${REWARD_EXTRA_PYTHONPATH:-}"
OMNIVOICE_REPO="${OMNIVOICE_REPO:?Set OMNIVOICE_REPO}"
OMNIVOICE_MODEL_DIR="${OMNIVOICE_MODEL_DIR:-$OMNIVOICE_REPO/download/tts_eval_models}"
WAVLM_CHECKPOINT="${WAVLM_CHECKPOINT:-$SEEDTTS_ROOT/wavlm_large_finetune.pth}"

# CV3-style multilingual ASR routing: zh uses ARK-ASR; all other supported
# languages use Whisper-large-v3 with the JSONL language code.
WHISPER_PATH="${WHISPER_PATH:?Set WHISPER_PATH}"
ARK_ASR_PATH="${ARK_ASR_PATH:?Set ARK_ASR_PATH}"
ARK_ASR_MAX_NEW_TOKENS="${ARK_ASR_MAX_NEW_TOKENS:-256}"
ARK_ASR_AUDIO_MAX_SECONDS="${ARK_ASR_AUDIO_MAX_SECONDS:-30.0}"
ARK_ASR_BATCH_SIZE="${ARK_ASR_BATCH_SIZE:-32}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_ROOT/grpo_hf_modules}"

mkdir -p "$TMPDIR"

cat > "$DS_ENV_FILE" <<EOF
PYTHONPATH=$PYTHONPATH
TMPDIR=$TMPDIR
TEMP=$TEMP
TMP=$TMP
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
TRITON_CACHE_DIR=$TRITON_CACHE_DIR
LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH
HF_MODULES_CACHE=$HF_MODULES_CACHE
TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM
OMP_NUM_THREADS=$OMP_NUM_THREADS
MKL_NUM_THREADS=$MKL_NUM_THREADS
USE_LIBUV=$USE_LIBUV
NCCL_DEBUG=$NCCL_DEBUG
NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
DS_ACCELERATOR=$DS_ACCELERATOR
EOF
chmod 600 "$DS_ENV_FILE"

# The mixed-language JSONL is prepared ahead of submission and must never be
# regenerated by a worker node.
[[ -s "$TRAIN_JSONL" ]] || {
  echo "Training JSONL is missing or empty: $TRAIN_JSONL" >&2
  exit 1
}

for required in \
  "$REWARD_PYTHON" \
  "$OMNIVOICE_REPO/omnivoice/eval/models/ecapa_tdnn_wavlm.py" \
  "$OMNIVOICE_MODEL_DIR/speaker_similarity/wavlm_large/hubconf.py" \
  "$OMNIVOICE_MODEL_DIR/speaker_similarity/wavlm_large/wavlm_large.pt" \
  "$WAVLM_CHECKPOINT"; do
  [[ -f "$required" ]] || {
    echo "OmniVoice SIM dependency is missing: $required" >&2
    exit 1
  }
done
PYTHONPATH="$REWARD_EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
  "$REWARD_PYTHON" -c 'import s3prl; assert s3prl.__version__ == "0.4.18", s3prl.__version__'

# Prepare runtime/cache directories before launching DeepSpeed.
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$LOCAL_NPY_CACHE_DIR" "$GRPO_DECODE_DIR" "$HF_MODULES_CACHE"
chmod 1777 "$TMPDIR"

# Launch GRPO training directly. This script does not call another wrapper.
"$PYTHON" -m deepspeed.launcher.runner \
  --hostfile "$HOSTFILE" \
  --num_nodes "$NUM_NODES" \
  --num_gpus "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$TRAIN_ROOT/src/train_arktts_hf_slowar_grpo.py" \
  --pretrained_ckpt_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --max_train_samples 1000000 \
  --output_dir "$OUTPUT_DIR" \
  --export_dir "$EXPORT_DIR" \
  --local_npy_cache_dir "$LOCAL_NPY_CACHE_DIR" \
  --local_npy_cache_source_prefix "$LOCAL_NPY_CACHE_SOURCE_PREFIX" \
  --local_npy_cache_read_only "$LOCAL_NPY_CACHE_READ_ONLY" \
  --local_npy_cache_rank_subdir "$LOCAL_NPY_CACHE_RANK_SUBDIR" \
  --text_key text \
  --ref_audio_ids_key pair_fish_audio_ids_path \
  --ref_text_key pair_text \
  --freeze_fast_ar false \
  --use_ref true \
  --shard_train_by_rank "$SHARD_TRAIN_BY_RANK" \
  --max_length 2048 \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio 0.01 \
  --num_train_epochs 3 \
  --logging_steps "$LOGGING_STEPS" \
  --report_to tensorboard \
  --logging_dir "$TENSORBOARD_DIR" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 10 \
  --do_train \
  --bf16 true \
  --logging_first_step true \
  --resume_mode none \
  --ignore_data_skip false \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --weight_decay 0.0 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --remove_unused_columns false \
  --grpo_reward_type "$GRPO_REWARD_TYPE" \
  --grpo_rollout_n "$GRPO_ROLLOUT_N" \
  --grpo_max_new_tokens "$GRPO_MAX_NEW_TOKENS" \
  --grpo_temperature "$GRPO_TEMPERATURE" \
  --grpo_top_k "$GRPO_TOP_K" \
  --grpo_top_p "$GRPO_TOP_P" \
  --grpo_length_bucket true \
  --grpo_kl_weight 0.01 \
  --grpo_entropy_weight 0.001 \
  --grpo_codec_batch_size "$CODEC_BATCH_SIZE" \
  --grpo_reward_alpha "$GRPO_REWARD_ALPHA" \
  --grpo_asr_weight "$GRPO_ASR_WEIGHT" \
  --grpo_sim_weight "$GRPO_SIM_WEIGHT" \
  --grpo_sim_floor "$GRPO_SIM_FLOOR" \
  --grpo_sim_ceil "$GRPO_SIM_CEIL" \
  --grpo_sim_reward_shape "$GRPO_SIM_REWARD_SHAPE" \
  --grpo_sim_reward_beta "$GRPO_SIM_REWARD_BETA" \
  --grpo_sim_backend "$GRPO_SIM_BACKEND" \
  --grpo_decode_dir "$GRPO_DECODE_DIR" \
  --grpo_seedtts_root "$SEEDTTS_ROOT" \
  --grpo_omnivoice_repo "$OMNIVOICE_REPO" \
  --grpo_omnivoice_model_dir "$OMNIVOICE_MODEL_DIR" \
  --grpo_wavlm_checkpoint "$WAVLM_CHECKPOINT" \
  --grpo_reward_python "$REWARD_PYTHON" \
  --grpo_reward_extra_pythonpath "$REWARD_EXTRA_PYTHONPATH" \
  --grpo_reward_timeout "$GRPO_REWARD_TIMEOUT" \
  --grpo_reward_keep_wavs "$GRPO_REWARD_KEEP_WAVS" \
  --grpo_whisper_path "$WHISPER_PATH" \
  --grpo_ark_asr_path "$ARK_ASR_PATH" \
  --grpo_ark_asr_max_new_tokens "$ARK_ASR_MAX_NEW_TOKENS" \
  --grpo_ark_asr_audio_max_seconds "$ARK_ASR_AUDIO_MAX_SECONDS" \
  --grpo_ark_asr_batch_size "$ARK_ASR_BATCH_SIZE" \
  --grpo_hf_modules_cache "$HF_MODULES_CACHE" \
  --skip_final_save false \
  --deepspeed "$TRAIN_ROOT/configs/deepspeed/zero2_fp32comm.json"

# The GRPO trainer's export only writes weights/config/tokenizer, so copy the
# audio8 runtime files (modeling code, codec, processor) from the v3 export to
# make the final checkpoint directly loadable by batch_infer_arktts.py.
if [[ -d "$EXPORT_DIR" ]]; then
  for f in codec.pth configuration_arktts.py modeling_arktts.py \
           modeling_arktts_codec.py processing_arktts.py \
           preprocessor_config.json processor_config.json \
           special_tokens_map.json tokenizer.json tokenizer_config.json \
           chat_template.jinja README.md; do
    if [[ -f "$MODEL_PATH/$f" ]]; then
      cp -a "$MODEL_PATH/$f" "$EXPORT_DIR/"
    fi
  done
  echo "[v5-export] completed $EXPORT_DIR with audio8 runtime files from $MODEL_PATH"
fi
