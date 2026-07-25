#!/usr/bin/env bash
# Queue scene-disjoint RCL-SGRAF BiGRU lower-LR continuations on GPUs 1/2.
set -euo pipefail

GPU_ID="${1:?usage: queue_roma_bigru_lr5e-5.sh <gpu-id> <shard-index> <num-shards>}"
SHARD_INDEX="${2:?missing shard-index}"
NUM_SHARDS="${3:?missing num-shards}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/Comparison/RCL/RCL-SGRAF/runs/roma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/disk_sda/ktj/cross-modality-results/comparison/rcl_sgraf/runs/roma_lr5e-5_bigru_rerun}"
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/crossmodal/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/disk_sda/ktj/scene_disjoint_v1}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_ROOT}/RoMa/vocab}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-512}"

wait_for_gpu() {
  while true; do
    local used
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${used}" && "${used}" -le "${IDLE_MEMORY_MIB}" ]]; then
      return
    fi
    echo "[$(date --iso-8601=seconds)] gpu=${GPU_ID} busy (${used:-unknown} MiB); waiting" >&2
    sleep 60
  done
}

cells=(scanrefer nr3d 3dllm scenedepict)
for i in "${!cells[@]}"; do
  (( i % NUM_SHARDS == SHARD_INDEX )) || continue
  dataset="${cells[$i]}"
  checkpoint="$(find "${SOURCE_ROOT}/${dataset}/bigru/checkpoint" -maxdepth 1 -type f -name '*model_best*_0.05_*.pth.tar' -print -quit)"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(find "${SOURCE_ROOT}/${dataset}/bigru/checkpoint" -maxdepth 1 -type f -name '*model_best*.pth.tar' -print -quit)"
  fi
  [[ -n "${checkpoint}" ]] || { echo "missing source checkpoint for ${dataset}/bigru" >&2; exit 1; }
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] starting ${dataset}/bigru on GPU ${GPU_ID}" >&2
  GPU_ID="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" DATA_ROOT="${DATA_ROOT}" VOCAB_PATH="${VOCAB_PATH}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" RESUME_CHECKPOINT="${checkpoint}" RESUME_RESET_EPOCH=1 \
    LEARNING_RATE=5e-5 NUM_EPOCHS=30 EARLY_STOP_PATIENCE=15 BATCH_SIZE=8 WORKERS=4 \
    bash "${PROJECT_ROOT}/Comparison/RCL/RCL-SGRAF/scripts/train_roma.sh" "${dataset}" bigru
done
