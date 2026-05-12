# Model Configuration Guide

This directory contains model configurations for different datasets and architectures.

## DiT Model Sizing

All model configurations now use a centralized sizing system defined in `dit_sizes.yaml`.

### Available Model Sizes

Choose from four standard Diffusion Transformer sizes:

| Size | Params (backbone only) | Hidden Size | Layers | Heads | Cond Hidden |
|------|----------------------|-------------|---------|-------|-------------|
| **S** (Small) | ~33M | 384 | 12 | 6 | 128 |
| **B** (Base) | ~130M | 768 | 12 | 12 | 128 |
| **L** (Large) | ~458M | 1024 | 24 | 16 | 256 |
| **XL** (Extra Large) | ~675M | 1152 | 28 | 16 | 256 |

### How to Use

In any model config file, simply set the `model_size` parameter:

```yaml
# @package _global_

defaults:
  - dit_sizes

# Model size: S, B, L, or XL
model_size: B  # Change this to switch model size
```

### What Gets Configured

When you set `model_size` and `model_head_size`, the following parameters are automatically configured:

#### Main DiT Backbone (controlled by `model_size`)
- `dit_hidden_size`
- `dit_num_hidden_layers`
- `dit_num_attention_heads`
- `dit_cond_hidden_size`

#### Encoder (controlled by `model_size`)
- `encoder_hidden_size`
- `encoder_num_hidden_layers`
- `encoder_num_attention_heads`
- `encoder_cond_hidden_size`

#### Model Head (controlled by `model_head_size`)
For latent models with RAE-style heads:
- `dit_head_hidden_size`
- `dit_head_num_layers`
- `dit_head_num_attention_heads`

### Example: Switching from Small to Large

**Before:**
```yaml
model_size: S
```

**After:**
```yaml
model_size: L
```

That's it! All model dimensions will automatically scale.

### Custom Configurations

If you need custom sizes not covered by S/B/L/XL, you can either:

1. **Override specific parameters** after importing dit_sizes:
   ```yaml
   defaults:
     - dit_sizes

   model_size: B
   dit_num_hidden_layers: 18  # Custom override
   ```

2. **Add a new size** to `dit_sizes.yaml`:
   ```yaml
   dit_sizes:
     M:  # Medium
       hidden_size: 512
       num_hidden_layers: 16
       # ...
   ```

### Head Sizes for Latent Models

Head sizes are based on the Representation Auto-Encoder (RAE) paper. The RAE paper found that **wide and shallow heads** (2-layer, 2048-dim 'G' head) are most effective:

> "A 2-layer, 2048-dim (G) head outperforms a 6-layer, 1152-dim (XL) head by a large margin, despite having similar GFlops."

All head configurations use **2 layers** and vary only in width:

| Head Size | Hidden Size | Layers | Heads | Use Case |
|-----------|-------------|--------|-------|----------|
| **B** (Base) | 768 | 2 | 12 | Small experiments |
| **H** (Large) | 1536 | 2 | 16 | Medium models |
| **G** (Giant) | 2048 | 2 | 16 | **Default (RAE recommendation)** |
| **T** (Tremendous) | 2688 | 2 | 21 | Large encoders |

The default is `model_head_size: G`, which can be overridden in your config:

```yaml
defaults:
  - dit_sizes

model_size: B
model_head_size: G  # Change to B, H, or T if needed
```

### Dataset-Specific Configs

Each dataset has its own model configurations:

- `text8/` - Text8 dataset models
- `lm1b/` - LM1B dataset models
- `owt/` - OpenWebText dataset models
- `binsaw/` - Binary SAW dataset models

Within each dataset folder:
- `mdlm_small.yaml` - MDLM baseline model
- `coladd*.yaml` - Co-LADD variants
- `diladd*.yaml` - Di-LADD variants (VQ-VAE/Gumbel)

### Notes

- All configs now use `model_size: S` by default
- Dropout rates and other hyperparameters remain dataset-specific
- Encoder type selection (DiT vs Qwen) is preserved in relevant configs
- Dict-based head configurations (e.g., for Qwen encoders) are preserved
