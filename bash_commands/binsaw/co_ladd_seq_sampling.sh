#!/usr/bin/env bash
set -euo pipefail

# Co-LADD SEQ sampling/evaluation on Binary SAW.
# Add resume.from_dir=... or resume.ckpt_path=... to choose a checkpoint.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/experiment_sample_local.sh" \
  +experiments/binsaw=coladd_sample \
  core_model=SEQ \
  "$@"
