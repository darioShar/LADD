# CLAUDE.md - Codebase Context for AI Agent Assistance

## Project Overview

This is a research codebase for **Co-LADD / Di-LADD** models (formerly referred to as LDDMs), focused on text and discrete sequence generation. The codebase is implemented using **PyTorch Lightning** and **Hydra** for configuration management.

**Primary Focus**: Text generation using novel discrete diffusion architectures
**Framework**: PyTorch Lightning 2.4.0, Hydra for config management

---

## Core Architecture

### Main Entry Point
- **File**: [hydra_main.py](hydra_main.py)
- **Purpose**: Main training/validation/prediction script
- **Key Features**:
  - Hydra-based configuration management
  - Multi-GPU support (DDP, FSDP)
  - Checkpoint management and resuming
  - WandB integration for experiment tracking
  - Supports SLURM and torchrun for distributed training

### Lightning Models (dlm/models/)

1. **[baselm.py](dlm/models/baselm.py)** - Base Lightning Module
   - Defines base training/validation steps
   - Model EMA setup
   - Default metrics: entropy, generative perplexity
   - Foundation for all discrete diffusion models

2. **[diffusion.py](dlm/models/diffusion.py)** - MDLM (Baseline)
   - Lightning module for Masked Discrete Diffusion Language Models
   - Strong baseline discrete diffusion model

3. **[latent.py](dlm/models/latent.py)** - Co-LADD (Novel Research)
   - Lightning module for Latent Discrete Diffusion Models
   - Implements augmented process with co-evolved latent embeddings
   - **Two-stage training procedure**:
     - **Stage 1**: Train encoder (x_0 → y_0) and data denoiser (x_t → x_0 given y_t)
     - **Stage 2**: Freeze encoder, train data denoiser and latent denoiser (y_t → y_0)

### Diffusion Module Implementations (dlm/modules/diffusionmodules/)

1. **Classical Masked Discrete Diffusion** (root level files)
   - [masked_loss.py](dlm/modules/diffusionmodules/masked_loss.py) - Loss computation for MDLM
   - [masked_sampling.py](dlm/modules/diffusionmodules/masked_sampling.py) - Sampling procedures
   - [noise_sampling.py](dlm/modules/diffusionmodules/noise_sampling.py) - Noise injection
   - [time_sampling.py](dlm/modules/diffusionmodules/time_sampling.py) - Time step sampling

2. **Co-LADD with Continuous Latents** ([coladd/](dlm/modules/diffusionmodules/coladd/))
   - [loss.py](dlm/modules/diffusionmodules/coladd/loss.py) - Two-stage training loss
   - [infonce_loss.py](dlm/modules/diffusionmodules/coladd/infonce_loss.py) - InfoNCE two-view invariance loss
   - Process: X_0 (discrete) → Y_0 (continuous) via encoder
   - Forward diffusion on (X_t, Y_t) independently
   - Backward diffusion with learned denoisers
   - Optional two-view invariance via encoder input masking (`p_mask_input_enc`) and InfoNCE loss (`use_infonce_loss`)
   - Encoder corruption hook (`use_p_r`) is plumbed through the latent process into encoders

3. **Di-LADD with Discrete Latents** ([diladd/](dlm/modules/diffusionmodules/diladd/))
   - [loss.py](dlm/modules/diffusionmodules/diladd/loss.py) - Discrete latent loss
   - [sampling.py](dlm/modules/diffusionmodules/diladd/sampling.py) - Discrete sampling
   - Uses VQ-VAE style discretization or Gumbel softmax
   - Optional two-view invariance via encoder input masking (`p_mask_input_enc`) and InfoNCE loss (pre-quantization logits/embeds)

### Neural Network Architectures (dlm/transformers/)

1. **DiT (Diffusion Transformer)**
   - [dit.py](dlm/transformers/dit/dit.py) - MDLM neural network
   - [latent_dit.py](dlm/transformers/dit/latent_dit.py) - Co-LADD neural network
   - Hybrid positional strategy for MM-DiT via `use_rope_for_latents`:
     - RoPE on text stream only + learned absolute slot embeddings for latents when disabled
     - Uses [hybrid_flash.py](dlm/transformers/dit/hybrid_flash.py) to split attention blocks and merge with LSE

2. **Qwen3 Integration**
   - [qwen_embedding.py](dlm/transformers/qwen3dit/qwen_embedding.py) - HuggingFace Qwen3 encoder
   - Can be plugged as encoder for Co-LADD/Di-LADD

### Data Modules (dlm/data/)

- [base.py](dlm/data/base.py) - Base datamodule class
- [text8.py](dlm/data/text8.py) - Text8 dataset
- [streaming.py](dlm/data/streaming.py) - Streaming datasets for large corpora
- [custom_tokenizers.py](dlm/data/custom_tokenizers.py) - Custom tokenization
- Multi-GPU support for data loading

### Metrics (dlm/metrics/)
- [token_metrics.py](dlm/metrics/token_metrics.py) - Text modeling metrics

### Callbacks (dlm/callbacks/)
- Custom Lightning callbacks for experiment management

---

## Configuration System (conf/)

**Base Config**: [conf/config.yaml](conf/config.yaml)

### Configuration Hierarchy:
```
conf/
├── config.yaml              # Main config with defaults
├── base/                    # Model family definitions
│   ├── mdlm.yaml           # MDLM baseline (masked process)
│   ├── coladd.yaml         # Co-LADD with continuous latents
│   ├── diladd.yaml         # Di-LADD with discrete latents
│   ├── md4.yaml            # MD4 baseline
│   └── autoregressive.yaml # Autoregressive baseline
├── data/                    # Dataset configurations
│   ├── owt/                # OpenWebText
│   ├── lm1b/               # 1-Billion Word dataset
│   ├── text8/              # Text8
│   ├── binsaw/             # Binary sawtooth
│   └── zeroshot/           # Zero-shot evaluation datasets
├── model/                   # Neural network configs (dataset-specific)
│   ├── text8/
│   │   ├── mdlm.yaml
│   │   ├── coladd.yaml
│   │   └── diladd.yaml
│   ├── lm1b/
│   │   ├── mdlm.yaml
│   │   ├── coladd.yaml
│   │   └── diladd.yaml
│   ├── owt/
│   ├── binsaw/
│   ├── mdlm_base.yaml      # Base MDLM architecture
│   ├── coladd_base.yaml    # Base Co-LADD architecture
│   └── diladd_base.yaml
├── lightning/
│   ├── trainer/            # Trainer configurations
│   ├── callbacks/          # Callback configurations
│   └── strategy/           # Distributed strategy (DDP, FSDP)
└── resume/                  # Resume configurations
    ├── from_null.yaml
    ├── from_dir.yaml
    └── from_ckpt.yaml
```

### Key Configuration Parameters:
- `mode`: train | val | pred
- `debug`: Enable debug mode (moves logs to debug_runs/)
- `projectname`: WandB project name
- `base`: Model family (mdlm, coladd, diladd, md4, autoregressive)
- Diffusion process selection is class-target driven in config (`sampler_config.target` and `loss_config.target`), not via a `diffusion_type` flag.
- `data`: Dataset configuration path
- `model`: Neural network configuration path
- `resume.from_dir`: Resume from checkpoint directory or file
- `fork_log`: Create new run when resuming (vs continue same run)
- `use_infonce_loss`, `infonce_weight`, `infonce_temperature`, `p_mask_input_enc`: two-view invariance loss for encoder
- `use_rope_for_latents`: Hybrid positional strategy toggle for MM-DiT (RoPE on x only + slot embeddings on y)

### Config Factoring Conventions

Dataset configs use shared per-dataset common files where possible, for example `conf/data/lm1b/_common.yaml` and
`conf/data/owt/_common.yaml`. Variant data configs should keep only tokenizer, sequence length, packing, cache path, or
dataset-specific overrides.

Experiment configs also use shared common files to reduce drift:
- `conf/experiments/{dataset}/_train_common.yaml`
- `conf/experiments/{dataset}/_sample_common.yaml`
- `conf/experiments/{dataset}/_coladd_common.yaml`
- `conf/experiments/{dataset}/_diladd_common.yaml`

For Hydra defaults ordering, common files should usually be loaded before `_self_`, so the local experiment or variant
file remains the final override layer. Some model-family commons own their `base`, `data`, and `model` defaults directly;
experiment YAMLs can then import the family common instead of repeating those three overrides.

Runtime/device config:
- `${auto_num_devices:}` resolves to visible CUDA device count, or `1` for MPS/CPU.
- `${cpu_count:}` is a portable CPU-count resolver and should be used instead of Linux-only `os.sched_getaffinity`.
- `lightning.trainer.accelerator` defaults to `auto`; `hydra_main.py` resolves CUDA/MPS/CPU at runtime.
- On MPS or single-device runs, Lightning strategy is disabled and Di-LADD `ddp_sync_mode` is forced to `none`.

---

## Running Experiments

### Basic Command Structure:
```bash
# Compose experiment YAMLs with optional variants
uv run python hydra_main.py \
  +experiments/owt=mdlm_train \
  +experiments/owt/variants=mdlm_bl

# Debug mode (single GPU, offline WandB)
uv run python hydra_main.py \
  +experiments/text8=mdlm_train \
  +experiments/text8/variants=mdlm_b \
  debug=true

# Resume training
uv run python hydra_main.py \
  +experiments/owt=mdlm_train \
  +experiments/owt/variants=mdlm_bl \
  resume.from_dir=/path/to/checkpoint

# Override specific parameters
uv run python hydra_main.py \
  +experiments/text8=coladd_train \
  +experiments/text8/variants=coladd_all_zero_no_rope \
  lightning.trainer.max_steps=10000 \
  batch_size=64
```

### Example Bash Scripts (bash_commands/)
Public launch wrappers are organized by dataset:
- [bash_commands/text8/experiment_train_local.sh](bash_commands/text8/experiment_train_local.sh)
- [bash_commands/text8/experiment_sample_local.sh](bash_commands/text8/experiment_sample_local.sh)
- [bash_commands/lm1b/experiment_train_local.sh](bash_commands/lm1b/experiment_train_local.sh)
- [bash_commands/lm1b/experiment_sample_local.sh](bash_commands/lm1b/experiment_sample_local.sh)
- [bash_commands/owt/experiment_train_local.sh](bash_commands/owt/experiment_train_local.sh)
- [bash_commands/owt/experiment_sample_local.sh](bash_commands/owt/experiment_sample_local.sh)
- [bash_commands/binsaw/experiment_train_local.sh](bash_commands/binsaw/experiment_train_local.sh)
- [bash_commands/binsaw/experiment_sample_local.sh](bash_commands/binsaw/experiment_sample_local.sh)

**Script Structure**:
- GPU and rendezvous configuration
- Optional batch-size overrides
- `hydra_main.py "$@"` forwarding
- Hydra command invocation

### Experiment Config Strategy

Prefer putting experiment identity in Hydra YAMLs and keeping bash scripts as plain launch wrappers.

This is implemented for text and synthetic sequence experiments as:
- `conf/experiments/<dataset_or_group>/*_{train,sample}.yaml`: dataset/model-family/stage configs, e.g. `mdlm_train`, `mdlm_sample`, `coladd_train`.
- `conf/experiments/<dataset_or_group>/variants/*.yaml`: reusable variant records, e.g. `mdlm_bl`, `coladd_lat64_adaptive`, `diladd_lat64_cb8192_dim128`.
- `bash_commands/<dataset_or_group>/experiment_train_local.sh` and `experiment_sample_local.sh`: local launch wrappers.

The bash scripts should stay simple: cluster/local setup plus `hydra_main.py "$@"`. Do not encode experiment matrices in bash arrays. Add or modify experiments by editing YAML files, then pass those YAMLs to the launcher as Hydra overrides.

OWT training example, local:
```bash
bash bash_commands/owt/experiment_train_local.sh \
  +experiments/owt=mdlm_train \
  +experiments/owt/variants=mdlm_bl
```

OWT Di-LADD staged training example, local:
```bash
bash bash_commands/owt/experiment_train_local.sh \
  +experiments/owt=diladd_stage_1_train \
  +experiments/owt/variants=diladd_lat64_cb8192_dim128

bash bash_commands/owt/experiment_train_local.sh \
  +experiments/owt=diladd_stage_2_resume_train \
  +experiments/owt/variants=diladd_lat64_cb8192_dim128
```

OWT sampling example, local:
```bash
bash bash_commands/owt/experiment_sample_local.sh \
  +experiments/owt=mdlm_sample \
  +experiments/owt/variants=mdlm_bl
```

Binary SAW training example, local:
```bash
bash bash_commands/binsaw/experiment_train_local.sh \
  +experiments/binsaw=mdlm_train
```

Generic multi-GPU or multi-node scheduler launch:
```bash
export NUM_GPUS=${NUM_GPUS:-8}
export NUM_NODES=${NUM_NODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

srun .venv/bin/torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${NUM_NODES} \
  --node_rank=${SLURM_NODEID} \
  --rdzv_backend=c10d \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  hydra_main.py \
  num_gpu_devices=${NUM_GPUS} \
  num_nodes=${NUM_NODES} \
  +experiments/owt=mdlm_train \
  +experiments/owt/variants=mdlm_bl
```

For offline scheduler jobs, pre-download datasets/models and pass `hf_offline=true`.

Sampling resume behavior:
- Sample YAMLs set `resume.from_dir: ${train_log_dir}` by default.
- Some training YAMLs can also set `resume.from_dir`; for example OWT Di-LADD stage 2 resumes from the matching stage-1 log root.
- In [hydra_main.py](hydra_main.py), if `resume.from_dir` is a parent directory containing timestamped run subdirectories (`YYYY-MM-DDTHH-MM-SS_*`), the latest timestamped child is selected automatically.
- If `resume.from_dir` is already a run directory containing `checkpoints/`, it is used directly.
- If `resume.from_dir` points to a `.ckpt` file, that exact checkpoint is used and the run logdir is inferred from its grandparent directory.
- `resume.ckpt_path=/path/to/checkpoint.ckpt` has priority over `resume.from_dir`.

To add a new experiment:
1. Add or edit a stage config in `conf/experiments/<dataset_or_group>/`.
2. Add or edit a variant in `conf/experiments/<dataset_or_group>/variants/`.
3. Run it directly with `hydra_main.py`, or with the local wrapper by passing `+experiments/<dataset_or_group>=...` and `+experiments/<dataset_or_group>/variants=...`.

---

## Development Environment

### Python Version: 3.12+
### CUDA: 12.3 (H100 GPUs used in experiments)

### Key Dependencies:
- **PyTorch**: 2.5.1 (with CUDA 12.1)
- **Lightning**: 2.4.0
- **Transformers**: 4.51.0
- **Hydra-core**: 1.3.2+
- **WandB**: 0.19.8
- **Flash-attn**: 2.7.4.post1 (Linux only)
- **DeepSpeed**: 0.16.4 (Linux only)
- **vLLM**: 0.7.3 (Linux only)
- **vector-quantize-pytorch**: Required for lucidrains vector quantizer

### Installation:
```bash
# Using uv package manager
uv sync --no-install-package flash-attn
uv sync

# Login to services
uv run huggingface-cli login
uv run wandb login
```

**Important**: The project uses a uv-managed virtual environment located in `.venv/`. To run Python scripts or commands, use `uv run python` or `.venv/bin/python`.

### Code Formatting (Ruff):
```toml
[tool.ruff.format]
quote-style = "single"
line-ending = "auto"
indent-style = "space"
docstring-code-format = true
docstring-code-line-length = "dynamic"
line-length = 125
```

**Important**: Always maintain single-quote style and 125 character line length.

---

## Key Implementation Details

### Model Instantiation
Uses `instantiate_from_config()` utility ([dlm/utils.py](dlm/utils.py)):
- Automatically loads classes from config paths
- Handles nested configuration objects
- Integrates with Hydra's OmegaConf

### Checkpointing
- **Directory Structure**: `logs/models/{experiment_name}/{timestamp}/checkpoints/`
- **Checkpoint Naming**: Managed by `get_checkpoint_name()` utility
- **Resume Logic**:
  - `resume.from_dir`: Can be logdir OR checkpoint file
  - `fork_log=true`: Load weights but start new run
  - `fork_log=false`: Continue same WandB run

### Distributed Training
- **Strategies**: DDP (default), FSDP (via custom wrapper)
- **SLURM Support**: Automatic detection via environment variables
- **Torchrun Support**: Compatible with torchrun launcher
- **Multi-GPU**: Batch size accumulation for large effective batch sizes

### Logging
- **WandB**: Online/offline mode, automatic run resuming
- **CSV Logger**: Local metrics storage
- **Callbacks**: Custom callbacks for saving WandB IDs, model checkpoints

### Encoder Corruption & Two-View Invariance
- **InfoNCE loss**: [infonce_loss.py](dlm/modules/diffusionmodules/coladd/infonce_loss.py) implements masked mean pooling, cosine similarity, and temperature scaling
- **Two-view masking**: `p_mask_input_enc` creates two corrupted encoder views; losses in [coladd/loss.py](dlm/modules/diffusionmodules/coladd/loss.py) and [diladd/loss.py](dlm/modules/diffusionmodules/diladd/loss.py) add InfoNCE when enabled
- **p_r corruption**: Routed through [latentprocess.py](dlm/modules/diffusionmodules/coladd/latentprocess.py) into encoders; `EncoderDiTModel` and `QwenEmbeddingModel` can zero masked token embeddings (Qwen requires `pooling_strategy=last_layer`)

### Hybrid Positional Attention (MM-DiT)
- **use_rope_for_latents=false**: Latents get learned slot embeddings while text retains RoPE; implemented in [latent_dit.py](dlm/transformers/dit/latent_dit.py)
- **Block-wise flash attention**: [hybrid_flash.py](dlm/transformers/dit/hybrid_flash.py) splits text/text vs cross/latent blocks and merges outputs with LSE

## Project Structure Overview

```
LADD/
├── hydra_main.py           # Main entry point
├── conf/                   # Hydra configurations
├── dlm/                    # Main package
│   ├── models/             # Lightning modules
│   ├── modules/            # Model components
│   │   └── diffusionmodules/
│   │       ├── coladd/           # Co-LADD (continuous)
│   │       └── diladd/           # Di-LADD (discrete)
│   ├── transformers/       # Neural networks
│   │   ├── dit/           # DiT architectures
│   │   └── qwen3dit/      # Qwen integration
│   ├── data/              # Data modules
│   ├── metrics/           # Evaluation metrics
│   ├── callbacks/         # Lightning callbacks
│   ├── lightning/         # Lightning utilities (FSDP)
│   └── utils.py           # Utility functions
├── bash_commands/         # Example run scripts
│   ├── text8/
│   ├── lm1b/
│   ├── owt/
│   └── binsaw/
├── download_hf_dataset.py # Dataset cache helper
└── download_pretrained_models.py # Model cache helper
```

---

## Common Tasks & Patterns

### Adding a New Model Variant:
1. Create Lightning module in [dlm/models/](dlm/models/)
2. Implement loss/sampling in [dlm/modules/diffusionmodules/](dlm/modules/diffusionmodules/)
3. Define neural network in [dlm/transformers/](dlm/transformers/)
4. Create config in [conf/base/](conf/base/)
5. Add dataset-specific model configs in [conf/model/](conf/model/)
6. Add experiment and variant YAMLs under [conf/experiments/](conf/experiments/)
7. Add or update a bash wrapper only when a repeated launch pattern needs one

### Debugging:
```bash
# Debug mode: Single GPU, offline WandB, logs move to debug_runs/
uv run python hydra_main.py +experiments/text8=mdlm_train +experiments/text8/variants=mdlm_b debug=true
```

### Resuming Training:
```bash
# Resume same run
uv run python hydra_main.py resume.from_dir=/path/to/logdir

# Resume from specific checkpoint
uv run python hydra_main.py resume.from_dir=/path/to/checkpoint.ckpt

# Fork: Load weights but create new run
uv run python hydra_main.py resume.from_dir=/path/to/logdir fork_log=true
```

### Hyperparameter Sweeps:
Use Hydra's multirun mode or WandB sweeps. Override parameters via command line:
```bash
uv run python hydra_main.py \
  +experiments/text8=coladd_train \
  +experiments/text8/variants=coladd_all_zero_no_rope \
  optimizer_config.params.lr=1e-4,5e-4,1e-3 \
  --multirun
```

---

## Important Notes for AI Assistance

When you are done modifying code, do not automatically create a summary document explaining the changes made.

### When Modifying Code:
1. **Always read files before editing** - Understand context before changes
2. **Maintain code style**: Single quotes, 125 char lines, Ruff formatting
3. **Respect two-stage training**: Co-LADD has distinct Stage 1 and Stage 2 logic
4. **Preserve checkpoint compatibility**: Be careful with state_dict changes
5. **Test with debug mode first**: Use `debug=true` for quick validation

### When Adding Features:
1. **Follow existing patterns**: Check similar implementations first
2. **Use instantiate_from_config**: Maintain config-based instantiation
3. **Support multi-GPU**: Consider DDP/FSDP compatibility
4. **Add configs**: Create corresponding YAML configs in [conf/](conf/)
5. **Document in bash scripts**: Provide example usage in [bash_commands/](bash_commands/)

### Common Pitfalls:
- **Checkpoint path remapping**: [hydra_main.py](hydra_main.py) includes path remapping logic for backward compatibility
- **Tokenizer passing**: Data modules expose `tokenizer` attribute, models consume it
- **EMA models**: Base LM manages EMA, check [baselm.py](dlm/models/baselm.py)
- **Distributed initialization**: Different logic for SLURM vs torchrun
- **Config resolution**: Use `OmegaConf.to_container(cfg, resolve=True)` for interpolation

### Testing:
- Python smoke check: `uv run python -c "import dlm.data; import dlm.metrics; import dlm.models"`
- Syntax check: `uv run python -m compileall -q dlm download_hf_dataset.py download_pretrained_models.py hydra_main.py`
- Config check: `uv run python hydra_main.py --cfg job +experiments/text8=mdlm_train debug=true`

---

## Research Context

### Co-LADD Innovation:
The key innovation is the **latent factoring** approach:
1. Discrete tokens X_0 are encoded to latent Y_0
2. Both X_t and Y_t are diffused independently
3. Denoising leverages cross-information (X_t ← Y_t, Y_t ← X_t)
4. This provides richer signal than pure discrete diffusion

### Two-Stage Training Rationale:
- **Stage 1**: Learn good encoder and basic denoising
- **Stage 2**: Refine joint denoising while keeping encoder stable

### Discrete vs Continuous Latents:
- **Continuous** ([coladd/](dlm/modules/diffusionmodules/coladd/)): Richer representations, harder to train
- **Discrete** ([diladd/](dlm/modules/diffusionmodules/diladd/)): VQ-VAE style, more stable, potentially lower capacity

---

## Current Task / Specific Modifications

**[User will append their specific task description below this line]**

---
