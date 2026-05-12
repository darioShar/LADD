#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/owt/experiment_train_local.sh \
#   +experiments/owt=mdlm_train \
#   +experiments/owt/variants=mdlm_bl

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZE=${BATCH_SIZE:-32}  # Adjust as needed based on GPU memory and model size

if [ "${NUM_GPUS}" -gt 1 ]; then
  .venv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${BATCH_SIZE} \
    "$@"
else
  .venv/bin/python hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${BATCH_SIZE} \
    "$@"
fi
