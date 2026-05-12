import torch
from torchmetrics import Metric


class PerplexityMetric(Metric):
    """PyTorch Lightning metric for computing perplexity.

    Properly aggregates loss across batches and then computes perplexity,
    instead of averaging perplexities which is mathematically incorrect.

    This metric accumulates the total loss and total number of tokens,
    then computes perplexity as exp(total_loss / total_tokens).
    """

    def __init__(self):
        super().__init__()
        # Use add_state to register metric state for distributed training
        self.add_state('total_loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_tokens', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, elbo: torch.Tensor, num_tokens: int):
        """Update the metric state.

        Args:
            elbo: Total evidence lower bound for the batch (sum of token losses)
            num_tokens: Number of tokens in the batch
        """
        self.total_loss += elbo
        self.total_tokens += num_tokens

    def compute(self):
        """Compute the perplexity.

        Returns:
            Perplexity as exp(average_loss_per_token)
        """
        if self.total_tokens == 0:
            return torch.tensor(float('inf'))

        average_loss_per_token = self.total_loss / self.total_tokens
        return torch.exp(average_loss_per_token)


class BitsPerCharacterMetric(Metric):
    """PyTorch Lightning metric for computing bits per character.

    Properly aggregates loss across batches and then computes BPC.
    """

    def __init__(self):
        super().__init__()
        self.add_state('total_loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_tokens', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, elbo: torch.Tensor, num_tokens: int):
        """Update the metric state.

        Args:
            elbo: Total evidence lower bound for the batch (sum of token losses)
            num_tokens: Number of tokens in the batch
        """
        self.total_loss += elbo
        self.total_tokens += num_tokens

    def compute(self):
        """Compute the bits per character.

        Returns:
            BPC as average_loss / ln(2)
        """
        if self.total_tokens == 0:
            return torch.tensor(float('inf'))

        average_loss = self.total_loss / self.total_tokens
        return average_loss / torch.log(torch.tensor(2.0))


class EntropyMetric(Metric):
    """PyTorch Lightning metric for computing token entropy over generated samples.

    Computes the entropy of token distributions across all generated tokens,
    providing a measure of diversity in the generated text.

    Entropy = -sum(p(token) * log(p(token))) where p(token) is the probability
    of each token computed from the frequency distribution over all samples.
    """

    def __init__(self, pad_token_id: int | None = None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.add_state(
            'sequence_entropy',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state('total_sequences', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, token_ids: torch.Tensor):
        for sentence_tokens in token_ids:
            if self.pad_token_id is not None:
                sentence_tokens = sentence_tokens[sentence_tokens != self.pad_token_id]
            total_tokens = sentence_tokens.numel()
            if total_tokens == 0:
                continue
            _, counts = torch.unique(sentence_tokens, return_counts=True)
            probs = counts.float() / float(total_tokens)
            entropy = -torch.sum(probs * torch.log(probs))
            self.sequence_entropy += entropy
            self.total_sequences += 1

    def compute(self):
        """Compute the entropy.

        Returns:
            Entropy over token distribution
        """
        return self.sequence_entropy / self.total_sequences.clamp(min=1)


class TokenDistributionKLMetric(Metric):
    """KL(P_real || P_generated) between empirical token distributions.

    MPS note: torch.index_add_ does not accept int64 sources on MPS, so counts are stored as float.
    """

    def __init__(self, vocab_size: int | None = None, pad_token_id: int | None = None, eps: float = 1e-8):
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else None
        self.pad_token_id = pad_token_id
        self.eps = float(eps)

        initial_size = self.vocab_size if self.vocab_size is not None else 10

        # IMPORTANT: counts in float (MPS-friendly), not long
        dtype = torch.float64 if not torch.backends.mps.is_available() else torch.float32
        self.add_state('real_token_counts', default=torch.zeros(initial_size, dtype=dtype), dist_reduce_fx='sum')
        self.add_state(
            'generated_token_counts',
            default=torch.zeros(initial_size, dtype=dtype),
            dist_reduce_fx='sum',
        )
        self.add_state('total_real_tokens', default=torch.tensor(0.0, dtype=dtype), dist_reduce_fx='sum')
        self.add_state('total_generated_tokens', default=torch.tensor(0.0, dtype=dtype), dist_reduce_fx='sum')

    def _resize_counts_(self, new_size: int) -> None:
        if new_size <= self.real_token_counts.size(0):
            return
        device = self.real_token_counts.device
        dtype = self.real_token_counts.dtype
        old = self.real_token_counts.size(0)

        new_real = torch.zeros(new_size, dtype=dtype, device=device)
        new_gen = torch.zeros(new_size, dtype=dtype, device=device)
        new_real[:old] = self.real_token_counts
        new_gen[:old] = self.generated_token_counts
        self.real_token_counts = new_real
        self.generated_token_counts = new_gen

    @torch.no_grad()
    def update(self, real_token_ids: torch.Tensor, generated_token_ids: torch.Tensor) -> None:
        dev = self.real_token_counts.device

        real_flat = real_token_ids.reshape(-1).to(dev)
        gen_flat = generated_token_ids.reshape(-1).to(dev)

        if self.pad_token_id is not None:
            real_flat = real_flat[real_flat != self.pad_token_id]
            gen_flat = gen_flat[gen_flat != self.pad_token_id]

        if real_flat.numel() == 0 and gen_flat.numel() == 0:
            return

        if real_flat.numel() > 0:
            real_unique, real_counts = torch.unique(real_flat, return_counts=True)
            real_unique = real_unique.to(torch.long)  # indices must be int64
            real_counts = real_counts.to(self.real_token_counts.dtype)  # source must be float on MPS
        else:
            real_unique = torch.empty((0,), device=dev, dtype=torch.long)
            real_counts = torch.empty((0,), device=dev, dtype=self.real_token_counts.dtype)

        if gen_flat.numel() > 0:
            gen_unique, gen_counts = torch.unique(gen_flat, return_counts=True)
            gen_unique = gen_unique.to(torch.long)
            gen_counts = gen_counts.to(self.generated_token_counts.dtype)
        else:
            gen_unique = torch.empty((0,), device=dev, dtype=torch.long)
            gen_counts = torch.empty((0,), device=dev, dtype=self.generated_token_counts.dtype)

        max_token = 0
        if real_unique.numel() > 0:
            max_token = max(max_token, int(real_unique.max().item()))
        if gen_unique.numel() > 0:
            max_token = max(max_token, int(gen_unique.max().item()))

        if max_token >= self.real_token_counts.size(0):
            if self.vocab_size is not None:
                raise ValueError(
                    f'Token id {max_token} exceeds configured vocab_size={self.vocab_size}. '
                    'Pass the correct vocab size to TokenDistributionKLMetric.',
                )
            self._resize_counts_(max_token + 1)

        if real_unique.numel() > 0:
            self.real_token_counts.index_add_(0, real_unique, real_counts)
            self.total_real_tokens += float(real_flat.numel())

        if gen_unique.numel() > 0:
            self.generated_token_counts.index_add_(0, gen_unique, gen_counts)
            self.total_generated_tokens += float(gen_flat.numel())

    def compute(self) -> torch.Tensor:
        dev = self.real_token_counts.device
        if self.total_real_tokens.item() == 0.0 or self.total_generated_tokens.item() == 0.0:
            return torch.tensor(0.0, device=dev)

        p_counts = self.real_token_counts
        q_counts = self.generated_token_counts

        support = p_counts > 0
        p = p_counts[support]
        q = q_counts[support]

        p = p / p.sum()
        q = q + self.eps
        q = q / q.sum()

        return torch.sum(p * (torch.log(p) - torch.log(q)))


class SequenceScalarMetricBase(Metric):
    """Base for metrics that compute a scalar per sequence and then
    aggregate mean/variance across sequences.
    Subclasses must implement `_sequence_stat(y_b: Tensor) -> Tensor scalar`.
    """

    def __init__(self):
        super().__init__()
        self.add_state('sum_vals', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_sq_vals', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count_seq', default=torch.tensor(0), dist_reduce_fx='sum')

    def _sequence_stat(self, y_b: torch.Tensor) -> torch.Tensor:
        """Return a scalar tensor for one sequence. To be overridden."""
        raise NotImplementedError

    def update(self, y: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Args:
            y: (B, S) long tensor of codes.
            attention_mask: (B, S) bool/int tensor, 1 for valid tokens (optional).
        """
        B, S = y.shape
        for b in range(B):
            if attention_mask is not None:
                mask_b = attention_mask[b].bool()
                y_b = y[b][mask_b]
            else:
                y_b = y[b]

            if y_b.numel() == 0:
                continue

            v = self._sequence_stat(y_b).to(self.sum_vals.device)
            self.sum_vals += v
            self.sum_sq_vals += v * v
            self.count_seq += 1

    def _mean_and_var(self):
        if self.count_seq == 0:
            zero = torch.tensor(0.0, device=self.sum_vals.device)
            return zero, zero
        n = self.count_seq.to(self.sum_vals.dtype)
        mean = self.sum_vals / n
        mean_sq = self.sum_sq_vals / n
        var = (mean_sq - mean * mean).clamp_min(0.0)
        return mean, var


# ----------------- Unique-code ratio: |unique(y)| / len(y) -----------------


class UniqueCodeRatioBase(SequenceScalarMetricBase):
    """Base: per-sequence unique-code ratio."""

    def _sequence_stat(self, y_b: torch.Tensor) -> torch.Tensor:
        # y_b: (L,) long
        length = y_b.numel()
        if length == 0:
            return torch.tensor(0.0, device=y_b.device)
        num_unique = y_b.unique().numel()
        return torch.tensor(float(num_unique) / float(length), device=y_b.device)


class UniqueCodeRatioMeanMetric(UniqueCodeRatioBase):
    """Mean of per-sequence unique-code ratios."""

    def compute(self) -> torch.Tensor:
        mean, _ = self._mean_and_var()
        return mean


class UniqueCodeRatioVarMetric(UniqueCodeRatioBase):
    """Variance of per-sequence unique-code ratios."""

    def compute(self) -> torch.Tensor:
        _, var = self._mean_and_var()
        return var


# --------------- Self-transition rate: #same / (len(y)-1) -----------------


class SelfTransitionRateBase(SequenceScalarMetricBase):
    """Base: per-sequence self-transition rate."""

    def _sequence_stat(self, y_b: torch.Tensor) -> torch.Tensor:
        # y_b: (L,)
        length = y_b.numel()
        if length <= 1:
            return torch.tensor(0.0, device=y_b.device)
        same = (y_b[1:] == y_b[:-1]).float()
        return same.mean()  # scalar tensor on y_b.device


class SelfTransitionRateMeanMetric(SelfTransitionRateBase):
    """Mean of per-sequence self-transition rates."""

    def compute(self) -> torch.Tensor:
        mean, _ = self._mean_and_var()
        return mean


class SelfTransitionRateVarMetric(SelfTransitionRateBase):
    """Variance of per-sequence self-transition rates."""

    def compute(self) -> torch.Tensor:
        _, var = self._mean_and_var()
        return var


# ----------------- VQ-VAE Embedding Space Collapse Metrics -----------------


class CodebookUtilizationMetric(Metric):
    """Measures the fraction of codebook entries that are actively used.

    Codebook collapse is indicated when only a small fraction of codes are used.
    This metric tracks the global utilization across all batches.

    Returns:
        Float in [0, 1]: fraction of codebook entries with at least one usage.
    """

    def __init__(self, num_embeddings: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.add_state(
            'used_codes',
            default=torch.zeros(num_embeddings, dtype=torch.bool),
            dist_reduce_fx='max',
        )

    def update(self, indices: torch.Tensor) -> None:
        """Update with discrete code indices.

        Args:
            indices: (B, S) or (B*S,) long tensor of code indices.
        """
        flat = indices.reshape(-1).long()
        # Mark codes as used
        valid = (flat >= 0) & (flat < self.num_embeddings)
        used = flat[valid]
        self.used_codes[used] = True

    def compute(self) -> torch.Tensor:
        return self.used_codes.float().mean()


class MinCodebookPairwiseDistanceMetric(Metric):
    """Measures the minimum pairwise distance between codebook embeddings.

    Low values indicate codebook collapse where embeddings are too close together.
    This should be computed on the codebook embeddings directly, not on indices.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.add_state('min_dist', default=torch.tensor(float('inf')), dist_reduce_fx='min')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, embeddings: torch.Tensor) -> None:
        """Update with codebook embeddings.

        Args:
            embeddings: (K, D) tensor of codebook embeddings.
        """
        K = embeddings.shape[0]
        if K < 2:
            return

        # Compute pairwise squared L2 distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        sq_norms = (embeddings**2).sum(dim=1)  # (K,)
        dists_sq = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * embeddings @ embeddings.t()

        # Set diagonal to inf to ignore self-distances
        dists_sq.fill_diagonal_(float('inf'))

        # Get minimum distance (take sqrt for actual L2 distance)
        min_dist_sq = dists_sq.min()
        min_dist = torch.sqrt(min_dist_sq.clamp(min=self.eps))

        if min_dist < self.min_dist:
            self.min_dist = min_dist
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.min_dist


class MeanCodebookPairwiseDistanceMetric(Metric):
    """Measures the mean pairwise distance between codebook embeddings.

    Low values indicate potential codebook collapse.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.add_state('sum_mean_dist', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, embeddings: torch.Tensor) -> None:
        """Update with codebook embeddings.

        Args:
            embeddings: (K, D) tensor of codebook embeddings.
        """
        K = embeddings.shape[0]
        if K < 2:
            return

        # Compute pairwise squared L2 distances
        sq_norms = (embeddings**2).sum(dim=1)
        dists_sq = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * embeddings @ embeddings.t()

        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(K, K, device=embeddings.device, dtype=torch.bool), diagonal=1)
        dists = torch.sqrt(dists_sq[mask].clamp(min=self.eps))

        self.sum_mean_dist += dists.mean()
        self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_mean_dist.device)
        return self.sum_mean_dist / self.count
