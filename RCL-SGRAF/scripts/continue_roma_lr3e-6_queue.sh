#!/usr/bin/env bash
# Continue RoMa-adapted RCL-SGRAF weights at 3e-6 without overwriting source runs.
set -euo pipefail

GPU_ID="${1:?usage: continue_roma_lr3e-6_queue.sh <gpu-id> [shard-index] [num-shards]}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/Comparison/RCL/RCL-SGRAF/runs/roma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/disk_sda/ktj/cross-modality-results/comparison/rcl_sgraf/runs/roma_lr3e-6_continue}"
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/crossmodal/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/disk_sda/ktj/scene_disjoint_v1}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_ROOT}/RoMa/vocab}"
BERT_PATH="${BERT_PATH:-/mnt/newdisk/ktj/pretrained/bert-base-uncased}"

cells=(
  'scanrefer bert'
  'nr3d bert'
  '3dllm bert'
  'scenedepict bert'
)
for i in "${!cells[@]}"; do
  (( i % NUM_SHARDS == SHARD_INDEX )) || continue
  read -r dataset encoder <<<"${cells[$i]}"
  # Prefer the latest low-LR continuation checkpoint when restarting a queue;
  # otherwise initialise once from the original high-LR best model.
  checkpoint=""
  resume_reset_epoch=1
  while IFS= read -r candidate; do
    if "${PYTHON_BIN}" - "${candidate}" >/dev/null 2>&1 <<'PY'
import sys
import torch
torch.load(sys.argv[1], map_location='cpu', weights_only=False)
PY
    then
      checkpoint="${candidate}"
      resume_reset_epoch=0
      break
    fi
  done < <(find "${OUTPUT_ROOT}/${dataset}/${encoder}/checkpoint" -maxdepth 1 -type f -name '*checkpoint*.pth.tar' -size +0c -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(find "${SOURCE_ROOT}/${dataset}/${encoder}/checkpoint" -maxdepth 1 -type f -name '*model_best*.pth.tar' -print -quit)"
  fi
  [[ -n "${checkpoint}" ]] || { echo "missing checkpoint for ${dataset}/${encoder}" >&2; exit 1; }
  echo "=== continue ${dataset}/${encoder} on GPU ${GPU_ID} ==="
  GPU_ID="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" DATA_ROOT="${DATA_ROOT}" VOCAB_PATH="${VOCAB_PATH}" BERT_PATH="${BERT_PATH}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" RESUME_CHECKPOINT="${checkpoint}" RESUME_RESET_EPOCH="${resume_reset_epoch}" \
    LEARNING_RATE=3e-6 BERT_LEARNING_RATE=3e-6 LR_UPDATE=30 NUM_EPOCHS=30 EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}" BATCH_SIZE=8 WORKERS=4 \
    bash "${PROJECT_ROOT}/Comparison/RCL/RCL-SGRAF/scripts/train_roma.sh" "${dataset}" "${encoder}"
done
