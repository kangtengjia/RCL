#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: train_roma.sh <dataset> <bigru|bert>}"; TEXT_ENCODER="${2:?usage: train_roma.sh <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"; VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"; BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"; OUTPUT_ROOT="${OUTPUT_ROOT:-runs/roma}"
cd "$(dirname "$0")/.."; EXTRA=(); [[ "${TEXT_ENCODER}" == bert ]] && EXTRA+=(--bert_path "${BERT_PATH}")
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py --data_name "${DATASET}" --data_path "${DATA_ROOT}" --data_root "${DATA_ROOT}" --vocab_path "${VOCAB_PATH}" --text_enc_type "${TEXT_ENCODER}" --img_dim 1024 --embed_size 1024 --num_epochs "${NUM_EPOCHS:-30}" --batch_size "${BATCH_SIZE:-8}" --workers "${WORKERS:-4}" --model_name "${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/checkpoint" --logger_name "${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/log" "${EXTRA[@]}"
