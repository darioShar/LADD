#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/binsaw/experiment_train_local.sh \
#   +experiments/binsaw=mdlm_train

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZE=${BATCH_SIZE:-2048}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-${BATCH_SIZE}}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export OMP_NUM_THREADS

if [ -n "${GPU_IDS:-}" ]; then
  export CUDA_VISIBLE_DEVICES=${GPU_IDS}
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  .venv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${BATCH_SIZE} \
    eval_batch_size=${EVAL_BATCH_SIZE} \
    "$@"
else
  .venv/bin/python hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    batch_size=${BATCH_SIZE} \
    eval_batch_size=${EVAL_BATCH_SIZE} \
    "$@"
fi
