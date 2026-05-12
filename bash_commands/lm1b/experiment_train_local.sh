#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/lm1b/experiment_train_local.sh \
#   +experiments/lm1b=mdlm_train \
#   +experiments/lm1b/variants=mdlm_bl

NUM_GPUS=1
MASTER_PORT=${MASTER_PORT:-29500}

batch_size=64


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
