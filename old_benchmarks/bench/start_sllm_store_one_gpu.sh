#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec /home/ben046/sllm/env/bin/sllm-store start \
  --storage-path /home/ben046/sllm/models/sllm \
  --mem-pool-size "${SLLM_MEM_POOL_SIZE:-32GB}" \
  --num-thread "${SLLM_NUM_THREAD:-4}"
