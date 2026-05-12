#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash bash_commands/owt/experiment_sample_local.sh \
#   +experiments/owt=mdlm_sample \
#   +experiments/owt/variants=mdlm_bl
#
# To choose a checkpoint manually, add resume.from_dir=... or resume.ckpt_path=...

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
