import torch
import torch.nn as nn

try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_forward,
        _flash_attn_varlen_backward,
    )
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


class FlashAttnVarLenWithLSE(torch.autograd.Function):
    """
    Custom Autograd for VarLen Flash Attention that exposes LSE for merging.
    """
    @staticmethod
    def forward(
        ctx, 
        q, k, v, 
        cu_seqlens_q, cu_seqlens_k, 
        max_seqlen_q, max_seqlen_k,
        dropout_p=0.0, softmax_scale=None, causal=False, 
        window_size_left=-1, window_size_right=-1, softcap=0.0, 
        alibi_slopes=None, deterministic=False
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # out: (Total_Tokens, H, D)
        # lse: (Batch, H, MaxSeqLen) <--- This is PADDED
        out, lse, _, rng_state = _flash_attn_varlen_forward(
            q, k, v, 
            cu_seqlens_q, cu_seqlens_k, 
            max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal, 
            window_size_left, window_size_right, softcap, 
            alibi_slopes, 
            False # return_softmax
        )

        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes, rng_state)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes, rng_state = ctx.saved_tensors
        
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        _flash_attn_varlen_backward(
            dout.contiguous(), 
            q, k, v, out, lse, 
            dq, dk, dv, 
            cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.dropout_p, ctx.softmax_scale, ctx.causal, 
            ctx.window_size_left, ctx.window_size_right, ctx.softcap, 
            alibi_slopes, ctx.deterministic, rng_state
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=False):
    return FlashAttnVarLenWithLSE.apply(
        q, k, v, 
        cu_seqlens_q, cu_seqlens_k, 
        max_seqlen_q, max_seqlen_k,
        0.0, softmax_scale, causal, 
        -1, -1, 0.0, None, False
    )


def lse_merge(out1, lse1, out2, lse2):
    """Merge two packed attention outputs based on LSE."""
    # Ensure LSEs are float32 for stability (they usually are from flash_attn)
    lse1 = lse1.to(torch.float32).unsqueeze(-1) # (Total, H, 1)
    lse2 = lse2.to(torch.float32).unsqueeze(-1)
    
    # Compute weights in Float32 to avoid overflow/underflow
    lse_max = torch.maximum(lse1, lse2)
    # Avoid -inf issues
    lse_max = lse_max.masked_fill(lse_max == float('-inf'), 0.0)

    w1 = torch.exp(lse1 - lse_max)
    w2 = torch.exp(lse2 - lse_max)
    
    # Determine the target dtype from inputs (e.g. bfloat16)
    target_dtype = out1.dtype
    
    # Cast chunks to Float32 for the weighted sum to maintain precision
    out1 = out1.to(torch.float32)
    out2 = out2.to(torch.float32)

    # Merge
    out_merged = (w1 * out1 + w2 * out2) / (w1 + w2 + 1e-6)
    
    # === FIX: Cast back to original dtype (BFloat16) ===
    return out_merged.to(target_dtype)


class HybridFlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        # Ensure clean division first
        assert dim % num_heads == 0, f"Hidden dim {dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim = dim // num_heads
        
        # --- ADD THESE ASSERTIONS ---
        assert self.head_dim % 8 == 0, (
            f"Flash Attention requires head_dim to be a multiple of 8. "
            f"Got dim={dim}, heads={num_heads} -> head_dim={self.head_dim}."
        )
        assert self.head_dim <= 256, (
            f"Flash Attention typically supports head_dim up to 256. "
            f"Got {self.head_dim}."
        )
        # ----------------------------

        self.scale = self.head_dim**-0.5

    def apply_rope(self, x, cos, sin):
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        seq_len = x.shape[1]
        # Slice for current sequence length and Q/K frequency index (0)
        curr_cos = cos[:, :seq_len, 0, :, :].to(x.dtype)
        curr_sin = sin[:, :seq_len, 0, :, :].to(x.dtype)
        
        return (x * curr_cos) + (rotate_half(x) * curr_sin)

    def _prep_latent_cu_seqlens(self, B, L, device):
        return torch.arange(0, (B + 1) * L, step=L, device=device, dtype=torch.int32)

    def _gather_lse(self, lse, seqlens=None):
        """
        Convert LSE to Packed format (Total_Valid_Tokens, H).
        
        Args:
            lse: (Batch, Heads, MaxSeqLen) [Standard] OR (Heads, Total) [Edge case]
            seqlens: Tensor of shape (Batch,) containing valid lengths.
        """
        # Case 1: LSE is already 2D (Heads, Total) or (Batch*MaxSeqLen, H) ?
        # Check if it matches the 'packed' assumption
        if lse.dim() == 2:
            # If shape is (H, N), transpose to (N, H)
            if lse.shape[0] == self.num_heads:
                return lse.transpose(0, 1).contiguous()
            return lse

        # Case 2: Standard Padded 3D output (Batch, Heads, MaxSeqLen)
        # We must slice out the valid data for each batch element.
        # This mirrors how 'varlen' attention packs the 'out' tensor.
        
        B, H, S = lse.shape
        
        # Permute to (Batch, MaxSeqLen, Heads) for easier slicing
        lse = lse.permute(0, 2, 1) # (B, S, H)
        
        if seqlens is None:
            # Assume full sequences if no lengths provided
            return lse.reshape(-1, H)

        # Iterate and collect valid segments
        # This is CPU-side loop but B is small, so it's negligible overhead 
        # compared to the massive GPU compute of attention.
        lse_packed_list = []
        for b in range(B):
            length = seqlens[b].item()
            # Slice valid tokens: [0 : length]
            lse_packed_list.append(lse[b, :length, :])
            
        # Concatenate into (Total_Valid, Heads)
        lse_packed = torch.cat(lse_packed_list, dim=0)
            
        return lse_packed

    def forward(self, qkv_text, qkv_latent, rope_cos, rope_sin, attn_mask):
        B, T, _, H, D = qkv_text.shape
        L = qkv_latent.shape[1]

        # ------------------------------------------------------------------
        # 1. Preprocessing: Unbinding
        # ------------------------------------------------------------------
        
        q_text_raw, k_text_raw, v_text = qkv_text.unbind(2) 
        q_lat, k_lat, v_lat = qkv_latent.unbind(2)

        # ------------------------------------------------------------------
        # 2. Apply RoPE ONLY for Text-to-Text
        # ------------------------------------------------------------------
        q_text_rot = self.apply_rope(q_text_raw, rope_cos, rope_sin)
        k_text_rot = self.apply_rope(k_text_raw, rope_cos, rope_sin)

        # ------------------------------------------------------------------
        # 3. Packing & Seqlens
        # ------------------------------------------------------------------

        # Calculate Text Seqlens from Mask
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask != 0
            seqlens_text = attn_mask.sum(dim=1, dtype=torch.int32)
            indices = torch.nonzero(attn_mask.flatten(), as_tuple=False).flatten()
        else:
            seqlens_text = torch.full((B,), T, device=qkv_text.device, dtype=torch.int32)
            indices = None

        cu_seqlens_text = torch.zeros(B + 1, device=qkv_text.device, dtype=torch.int32)
        cu_seqlens_text[1:] = torch.cumsum(seqlens_text, dim=0)
        max_seqlen_text = int(seqlens_text.max().item())

        cu_seqlens_lat = self._prep_latent_cu_seqlens(B, L, qkv_text.device)
        max_seqlen_lat = L

        # helper to pack a tensor (B, T, H, D) -> (Total, H, D)
        def pack(x):
            if indices is not None:
                return x.reshape(-1, H, D).index_select(0, indices)
            return x.reshape(-1, H, D)

        # Pack RAW Text (for Cross Attn)
        q_text_raw_packed = pack(q_text_raw)
        k_text_raw_packed = pack(k_text_raw)
        v_text_packed     = pack(v_text)

        # Pack ROTATED Text (for Self Attn)
        q_text_rot_packed = pack(q_text_rot)
        k_text_rot_packed = pack(k_text_rot)

        # Pack Latent (trivial flatten)
        q_lat_packed = q_lat.reshape(-1, H, D)
        k_lat_packed = k_lat.reshape(-1, H, D)
        v_lat_packed = v_lat.reshape(-1, H, D)

        # ------------------------------------------------------------------
        # 4. Attention Blocks
        # ------------------------------------------------------------------

        # --- Block 1: Output for TEXT tokens ---
        
        # 1A: Text -> Text
        out_tt, lse_tt_padded = flash_attn_varlen_func(
            q_text_rot_packed, k_text_rot_packed, v_text_packed,
            cu_seqlens_text, cu_seqlens_text, max_seqlen_text, max_seqlen_text,
            softmax_scale=self.scale
        )

        # 1B: Text -> Latent
        out_tl, lse_tl_padded = flash_attn_varlen_func(
            q_text_raw_packed, k_lat_packed, v_lat_packed,
            cu_seqlens_text, cu_seqlens_lat, max_seqlen_text, max_seqlen_lat,
            softmax_scale=self.scale
        )

        # === FIX: GATHER LSEs ===
        # Use indices to pack text LSEs
        lse_tt = self._gather_lse(lse_tt_padded, seqlens=seqlens_text)
        lse_tl = self._gather_lse(lse_tl_padded, seqlens=seqlens_text)

        # Merge & Unpack Text
        out_text_packed = lse_merge(out_tt, lse_tt, out_tl, lse_tl)
        
        out_text = torch.zeros(B * T, H, D, device=qkv_text.device, dtype=qkv_text.dtype)
        if indices is not None:
            out_text.index_copy_(0, indices, out_text_packed)
        else:
            out_text = out_text_packed
        out_text = out_text.view(B, T, H * D)

        # --- Block 2: Output for LATENT tokens ---
        
        # 2A: Latent -> Text
        out_lt, lse_lt_padded = flash_attn_varlen_func(
            q_lat_packed, k_text_raw_packed, v_text_packed,
            cu_seqlens_lat, cu_seqlens_text, max_seqlen_lat, max_seqlen_text,
            softmax_scale=self.scale
        )

        # 2B: Latent -> Latent
        out_ll, lse_ll_padded = flash_attn_varlen_func(
            q_lat_packed, k_lat_packed, v_lat_packed,
            cu_seqlens_lat, cu_seqlens_lat, max_seqlen_lat, max_seqlen_lat,
            softmax_scale=self.scale
        )

        # === FIX: GATHER LSEs ===
        # Latents are usually dense/full, so seqlens is just [L, L, L...]
        # We can create a seqlens tensor for latents or just rely on LSE being full.
        seqlens_lat = torch.full((B,), L, device=qkv_text.device, dtype=torch.int32)
        lse_lt = self._gather_lse(lse_lt_padded, seqlens=seqlens_lat)
        lse_ll = self._gather_lse(lse_ll_padded, seqlens=seqlens_lat)

        # Merge & Reshape Latent
        out_lat_packed = lse_merge(out_lt, lse_lt, out_ll, lse_ll)
        out_lat = out_lat_packed.view(B, L, H * D)

        return out_text, out_lat