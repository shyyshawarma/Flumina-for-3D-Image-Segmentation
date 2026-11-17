#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Set CHECKPOINT to the trained weights to evaluate.
CHECKPOINT="${CHECKPOINT:-checkpoints/AMOS-2022/Flumina-M.pt}"
# Optional for weights without an adjacent training config.json.
CONFIG="${CONFIG:-}"
CONFIG_ARGS=()
if [[ -n "${CONFIG}" ]]; then
  CONFIG_ARGS=(--config "${CONFIG}")
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

exec "${PYTHON_BIN:-python}" -m inference \
  --dataset AMOS-2022 \
  --model Flumina-M \
  --checkpoint "${CHECKPOINT}" \
  --data-root "${DATA_PATH:-data}" \
  --output-dir "${OUTPUT_DIR:-results}" \
  --device "${DEVICE:-auto}" \
  --metrics "${METRICS:-benchmark}" \
  --inference_augument "${INFERENCE_AUGUMENT:-True}" \
  "${CONFIG_ARGS[@]}" \
  "$@"
