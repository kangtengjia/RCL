#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"
BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$(dirname "$0")/.."
"${PYTHON_BIN}" - "${DATA_ROOT}" "${BERT_PATH}" <<'PY'
import sys
from pathlib import Path
from roma import load_roma_bundle
root, bert = sys.argv[1:]
assert (Path(bert) / 'config.json').is_file()
for dataset in ('scenedepict','scanrefer','nr3d','3dllm'):
 for split in ('train','val'):
  bundle=load_roma_bundle(root,dataset,split); print(dataset,split,len(bundle.captions),bundle.features.shape)
PY
