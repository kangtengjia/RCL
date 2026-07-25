#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: train_roma.sh <dataset> <bigru|bert>}"; TEXT_ENCODER="${2:?usage: train_roma.sh <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"; VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"; BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"; OUTPUT_ROOT="${OUTPUT_ROOT:-runs/roma}"
cd "$(dirname "$0")/.."; EXTRA=(); DEFAULT_EPOCHS=30
if [[ "${TEXT_ENCODER}" == bert ]]; then
  DEFAULT_EPOCHS=15
  EXTRA+=(--bert_path "${BERT_PATH}" --bert_learning_rate "${BERT_LEARNING_RATE:-3e-5}" --bert_warmup_epochs "${BERT_WARMUP_EPOCHS:-2}" --weight_decay "${WEIGHT_DECAY:-1e-4}")
fi
MODEL_DIR="${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/checkpoint"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA+=(--resume "${RESUME_CHECKPOINT}")
  [[ "${RESUME_RESET_EPOCH:-0}" == "1" ]] && EXTRA+=(--resume_reset_epoch)
elif [[ "${AUTO_RESUME:-0}" == "1" ]]; then
  while IFS= read -r checkpoint; do
    if "${PYTHON_BIN}" - "${checkpoint}" >/dev/null 2>&1 <<'PY'
import sys
import torch
torch.load(sys.argv[1], map_location="cpu", weights_only=False)
PY
    then
      EXTRA+=(--resume "${checkpoint}")
      break
    fi
  done < <(find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.pth.tar' -size +0c -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py --data_name "${DATASET}" --data_path "${DATA_ROOT}" --data_root "${DATA_ROOT}" --vocab_path "${VOCAB_PATH}" --text_enc_type "${TEXT_ENCODER}" --img_dim 1024 --num_regions 200 --embed_size 1024 --num_epochs "${NUM_EPOCHS:-${DEFAULT_EPOCHS}}" --early_stop_patience "${EARLY_STOP_PATIENCE:-10}" --batch_size "${BATCH_SIZE:-8}" --workers "${WORKERS:-4}" --learning_rate "${LEARNING_RATE:-5e-4}" --lr_update "${LR_UPDATE:-30}" --lr_schedule "${LR_SCHEDULE:-step}" --lr_min "${LR_MIN:-0}" --lr_cycle_epochs "${LR_CYCLE_EPOCHS:-100}" --model_name "${MODEL_DIR}" --logger_name "${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/log" "${EXTRA[@]}"
