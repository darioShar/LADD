#!/bin/bash

LATENT_SEQUENCE_LENGTH=${LATENT_SEQUENCE_LENGTH:-16}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}  # Adjust as needed based on GPU memory and model size
TORCH_COMPILE=${TORCH_COMPILE:-max-autotune-no-cudagraphs} # null | default | max-autotune-no-cudagraphs
ATTN_BACKEND=${ATTN_BACKEND:-sdpa} # auto | sdpa | flash , must be set to sdpa if torch_compile enabled

NUM_GPUS=${NUM_GPUS} BATCH_SIZE=${BATCH_SIZE} bash bash_commands/owt/experiment_train_local.sh \
  +experiments/owt=diladd_stage_2_resume_train \
  +experiments/owt/variants=diladd_lat${LATENT_SEQUENCE_LENGTH}_cb8192_dim128 \
  torch_compile=${TORCH_COMPILE} \
  attn_backend=${ATTN_BACKEND}
