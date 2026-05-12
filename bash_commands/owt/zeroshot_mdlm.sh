#!/bin/bash

set -euo pipefail

# =========================================================
# 1. Runtime setup (single GPU, local script)
# =========================================================
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

NUM_GPUS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

HF_CACHE_DIR=${HF_CACHE_DIR:-./.hf_cache}
HF_OFFLINE=${HF_OFFLINE:-false}
QWEN_MODEL_ID=${QWEN_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}
QWEN_MODEL_PATH=${QWEN_MODEL_ID}

if [ "${HF_OFFLINE}" = "true" ]; then
  export HF_HOME="${HF_CACHE_DIR}"
  export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
  export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
  export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1

  QWEN_CACHE_DIR="${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-Embedding-0.6B"
  if [ -f "${QWEN_CACHE_DIR}/refs/main" ]; then
    QWEN_SNAPSHOT=$(cat "${QWEN_CACHE_DIR}/refs/main")
  else
    QWEN_SNAPSHOT=$(ls -1 "${QWEN_CACHE_DIR}/snapshots" 2>/dev/null | head -n 1)
  fi
  if [ -n "${QWEN_SNAPSHOT:-}" ] && [ -f "${QWEN_CACHE_DIR}/snapshots/${QWEN_SNAPSHOT}/config.json" ]; then
    QWEN_MODEL_PATH="${QWEN_CACHE_DIR}/snapshots/${QWEN_SNAPSHOT}"
  fi
fi

BASE=mdlm

# =========================================================
# 2. Shared evaluation knobs
# =========================================================
sequence_length=${SEQUENCE_LENGTH:-512}

batch_size=${BATCH_SIZE:-8}
num_workers=${NUM_WORKERS:-8}
val_batch_limit=${VAL_BATCH_LIMIT:-100}
eval_and_log_first_n_batches=${EVAL_AND_LOG_FIRST_N_BATCHES:-0}

torch_compile=${TORCH_COMPILE:-null}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
attn_backend=${ATTN_BACKEND:-auto}

data_config_list=(
  'zeroshot/ptb_qwen'
  'zeroshot/wikitext103_qwen'
  'zeroshot/lm1b_qwen'
  'zeroshot/lambada_qwen'
  'zeroshot/ag_news_qwen'
  'zeroshot/pubmed_qwen'
  'zeroshot/arxiv_qwen'
)

dataset_name_list=(
  'ptb'
  'wikitext103'
  'lm1b'
  'lambada'
  'ag_news'
  'pubmed'
  'arxiv'
)

variant_config_list=('mdlm_bl.yaml')
run_name_list=(mdlm_bl)

# =========================================================
# 4. Launch zero-shot validation runs
# =========================================================
for idx in "${!variant_config_list[@]}"; do
  variant_config=${variant_config_list[$idx]}
  run_name=${run_name_list[$idx]}

  for data_idx in "${!data_config_list[@]}"; do
    data_config=${data_config_list[$data_idx]}
    dataset_name=${dataset_name_list[$data_idx]}
    log_root="logs/models/owt/zeroshot/${BASE}/${dataset_name}/seq_${sequence_length}"
    mkdir -p "${log_root}"

    echo "Running ${run_name} on ${dataset_name}"

    .venv/bin/python hydra_main.py \
      data=${data_config} \
      +experiments/owt=mdlm_sample.yaml \
      +experiments/owt/variants=${variant_config} \
      fork_log=true \
      projectname=owt_zeroshot \
      group=mdlm_zeroshot \
      postfix=_${run_name}_${dataset_name}_seq${sequence_length} \
      num_gpu_devices=${NUM_GPUS} \
      batch_size=${batch_size} \
      eval_batch_size=${batch_size} \
      num_workers=${num_workers} \
      hf_cache_dir=${HF_CACHE_DIR} \
      hf_offline=${HF_OFFLINE} \
      data.params.tokenizer_config.params.pretrained_model_name_or_path=${QWEN_MODEL_PATH} \
      sequence_length=${sequence_length} \
      model.params.eval_and_log_first_n_batches=${eval_and_log_first_n_batches} \
      model.params.enable_entropy=false \
      model.params.enable_generative_perplexity=false \
      model.params.enable_gradient_moment_metric=false \
      model.params.enable_sliced_wasserstein=false \
      model.params.enable_token_distribution_kl=false \
      ++lightning.trainer.limit_val_batches=${val_batch_limit} \
      paths.log_root=${log_root} \
      torch_compile=${torch_compile} \
      gradient_checkpointing=${gradient_checkpointing} \
      attn_backend=${attn_backend} \
      "$@"
  done
done
