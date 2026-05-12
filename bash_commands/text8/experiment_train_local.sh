#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/text8/experiment_train_local.sh \
#   +experiments/text8=mdlm_train \
#   +experiments/text8/variants=mdlm_b

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

batch_size=128  # Adjust as needed based on GPU memory and model size

if [ "${NUM_GPUS}" -gt 1 ]; then
  .venv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${batch_size} \
    "$@"
else
  .venv/bin/python hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${batch_size} \
    "$@"
fi
