#!/usr/bin/env bash
set -euo pipefail

# Co-LADD on Binary SAW.
# Stage 1 is the default. For stage 2, add overrides such as:
#   training_stage=2 freeze_encoder=true use_encoder_latent_sampling=false resume.from_dir=/path/to/stage1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/experiment_train_local.sh" \
  +experiments/binsaw=coladd_train \
  "$@"
