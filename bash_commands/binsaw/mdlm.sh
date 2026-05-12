#!/usr/bin/env bash
set -euo pipefail

# MDLM on Binary SAW.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/experiment_train_local.sh" \
  +experiments/binsaw=mdlm_train \
  "$@"
