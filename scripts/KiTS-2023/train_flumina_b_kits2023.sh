#!/usr/bin/env bash
# Fixed training recipe; trailing CLI arguments override these defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec "${REPO_ROOT}/scripts/train_model.sh" \
  --model-name flumina \
  --model-variant b \
  --dataset-name KiTS-2023 \
  --data-path "${DATA_PATH:-data}" \
  --run-id "${RUN_ID:-flumina_b_kits2023}" \
  --log-root "${LOG_ROOT:-runs}" \
  --base-lr "${BASE_LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-5e-2}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --crop-size "${CROP_X:-128}" "${CROP_Y:-128}" "${CROP_Z:-128}" \
  --num-epochs "${NUM_EPOCHS:-1000}" \
  --iter-per-epoch "${ITER_PER_EPOCH:-250}" \
  --validation-start-epoch "${VALIDATION_START_EPOCH:-400}" \
  --validation-interval "${VALIDATION_INTERVAL:-10}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-200}" \
  --top-k "${TOP_K:-10}" \
  --evaluation-top-k "${EVALUATION_TOP_K:-10}" \
  --seed "${SEED:-42}" \
  --in-channels 1 \
  --num-classes 4 \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" \
  --compile \
  --adam-beta1 ${ADAM_BETA1:-0.9} \
  --adam-beta2 ${ADAM_BETA2:-0.98} \
  --optimizer-eps ${OPTIMIZER_EPS:-1e-8} \
  --warmup-ratio ${WARMUP_RATIO:-0.05} \
  --warmup-lr ${WARMUP_LR:-1e-5} \
  --min-lr ${MIN_LR:-1e-6} \
  --grad-clip ${GRAD_CLIP:-1} \
  --use-ema \
  --ema-decay "${EMA_DECAY:-0.999}" \
  --inference_augument "${INFERENCE_AUGUMENT:-True}" \
  --progress \
  "$@"
