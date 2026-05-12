#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/text8/experiment_sample_local.sh \
#   +experiments/text8=mdlm_sample \
#   +experiments/text8/variants=mdlm_b
#
# To choose a checkpoint manually, add:
#   resume.from_dir=/path/to/lightning/logdir
# or:
#   resume.ckpt_path=/path/to/checkpoint.ckpt

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

if [ "${NUM_GPUS}" -gt 1 ]; then
  .venv/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    "$@"
else
  .venv/bin/python hydra_main.py \
    num_gpu_devices=${NUM_GPUS} \
    "$@"
fi
