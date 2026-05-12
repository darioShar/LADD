#!/usr/bin/env bash
set -euo pipefail

# Simple LM1B Di-LADD SEQ sampling matrix built on the existing Hydra YAML setup.
#
# Edit the lists below by hand to choose which step pairs to launch.
# The two step lists are paired by index inside one Hydra run:
#   NUM_INFERENCE_STEPS_LIST[i] with NUM_INFERENCE_LATENT_STEPS_LIST[i]
#
# Example matched-total reverse-step budgets:
#   NUM_INFERENCE_STEPS_LIST=(1 4 16 64)
#   NUM_INFERENCE_LATENT_STEPS_LIST=(1 4 16 64)
# gives totals of:
#   2, 8, 32, 128
#
# To sweep temperatures like the earlier LM1B runs, update TEMPERATURES as needed.

TEMPERATURES=(1.0)

VARIANTS=(
  diladd_lat32_cb2048_dim128.yaml
  diladd_lat128_cb512_dim32.yaml
)

NUM_INFERENCE_STEPS_LIST=(1 2 4 8 16 32 64 128)
NUM_INFERENCE_LATENT_STEPS_LIST=(8 8 8 8 8 8 8 8)

ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-attn_backend=sdpa}
EXPERIMENT_YAML=${EXPERIMENT_YAML:-diladd_sample.yaml}

if [ ${#NUM_INFERENCE_STEPS_LIST[@]} -ne ${#NUM_INFERENCE_LATENT_STEPS_LIST[@]} ]; then
  echo 'NUM_INFERENCE_STEPS_LIST and NUM_INFERENCE_LATENT_STEPS_LIST must have the same length.' >&2
  exit 1
fi

num_inference_steps_list_str="["
num_inference_steps_latent_list_str="["
for idx in "${!NUM_INFERENCE_STEPS_LIST[@]}"; do
  x_steps=${NUM_INFERENCE_STEPS_LIST[$idx]}
  y_steps=${NUM_INFERENCE_LATENT_STEPS_LIST[$idx]}
  total_reverse_steps=$((x_steps + y_steps))
  echo "Configured pair ${idx}: x_steps=${x_steps}, y_steps=${y_steps}, total=${total_reverse_steps}"

  if [ "$idx" -gt 0 ]; then
    num_inference_steps_list_str+=","
    num_inference_steps_latent_list_str+=","
  fi

  num_inference_steps_list_str+="${x_steps}"
  num_inference_steps_latent_list_str+="${y_steps}"
done
num_inference_steps_list_str+="]"
num_inference_steps_latent_list_str+="]"

for temperature in "${TEMPERATURES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "Launching ${variant} at temperature=${temperature} with x_steps=${num_inference_steps_list_str} and y_steps=${num_inference_steps_latent_list_str}"
    bash bash_commands/lm1b/experiment_sample_local.sh \
      +experiments/lm1b=${EXPERIMENT_YAML} \
      +experiments/lm1b/variants=${variant} \
      temperature=${temperature} \
      ${ADDITIONAL_ARGS} \
      SEQ=true \
      num_inference_steps="${num_inference_steps_list_str}" \
      num_inference_steps_latent="${num_inference_steps_latent_list_str}" \
      group=diladd_seq_sampling \
      postfix=_seq_matrix_temp${temperature} \
      sample_root_dir=logs/models/lm1b/sampling/diladd/seq \
      num_workers=0
  done
done
