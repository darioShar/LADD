import torch


def get_anneal_attn_mask(input_ids, attn_mask_ratio, dtype=torch.float32):
    """
    Implementation of the annealing attention mask from DiffuLLaMA (https://arxiv.org/abs/2410.17891).

    Reference:
    - https://github.com/HKUNLP/DiffuLLaMA/blob/main/LLaMA-Factory/src/llamafactory/train/ddm/trainer.py#L642
    """
    bsz, seq_len = input_ids.size()
    mask = torch.full((seq_len, seq_len), 0, device=input_ids.device)
    mask_cond = torch.arange(mask.size(-1), device=input_ids.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)

    random_mask = torch.bernoulli(
        torch.full((seq_len, seq_len), 0.0, device=input_ids.device) + attn_mask_ratio,
    )

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min,
    )  # shape: (bsz, 1, seq_len, seq_len)


# copied from https://github.com/dvruette/gidd/blob/main/gidd/utils.py#L37
@torch.no_grad()
def sample_categorical(probs, generator=None):
    # Convert to float64 for maximum precision in final sampling operation
    dtype = torch.float64 if probs.device.type != 'mps' else torch.float32
    probs_float64 = probs.to(dtype)
    uniform = torch.rand(
        probs_float64.shape[:-1],
        dtype=dtype,
        device=probs_float64.device,
        generator=generator,
    ).unsqueeze(-1)
    cumprobs = probs_float64.cumsum(-1)
    cumprobs[..., -1] = 1 + 1e-4
    return torch.searchsorted(cumprobs, uniform, right=True).squeeze(-1)



def sample_categorical_gumbel(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)
