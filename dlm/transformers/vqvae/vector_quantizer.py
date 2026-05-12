import torch
import torch.nn.functional as F
from torch import nn

class VectorQuantizer(nn.Module):
    """
    VQ-VAE style vector quantizer with optional EMA codebook updates,
    L2 normalization (cosine similarity), and Codebook Restart (dead code revival).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        # New parameters
        normalize: bool = False,       # L2 normalize inputs and codes (Cosine Similarity)
        restart_threshold: float | None = None # Usage threshold to consider a code 'dead'
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps
        self.normalize = normalize
        self.use_restart = False
        self.restart_threshold = restart_threshold
        if self.restart_threshold is not None:
            self.use_restart = True

        if self.use_ema:
            # Codebook as buffers (no grads)
            embedding = torch.randn(num_embeddings - 1, embedding_dim)
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)
            
            self.register_buffer('embedding', embedding)
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings - 1))
            self.register_buffer('ema_embedding', embedding.clone())
        else:
            # Codebook as parameters (learned by gradient)
            self.embedding = nn.Embedding(num_embeddings - 1, embedding_dim)
            nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # embedding for the masked vector is learned anyway
        # Note: We usually do NOT normalize the mask token or restart it;
        # it is a special semantic marker.
        self.z_mask = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.uniform_(self.z_mask, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def codebook(self) -> torch.Tensor:
        w = self.embedding if self.use_ema else self.embedding.weight
        if self.normalize:
            # Always return unit-norm codes for distance calculation
            w = F.normalize(w, p=2, dim=-1)
        return w

    def forward(self, z_e: torch.Tensor, mask_positions: torch.Tensor | None = None, return_one_hot: bool = False,):
        """
        Args:
            z_e: encoder output, shape (B, ..., D).
        """
        z_e_shape = z_e.shape
        assert z_e_shape[-1] == self.embedding_dim
        flat_z_e = z_e.reshape(-1, self.embedding_dim)  # (N, D)

        # 1. Normalization (Optional)
        # If we normalize, Euclidean distance is equivalent to Cosine Distance
        if self.normalize:
            flat_z_e = F.normalize(flat_z_e, p=2, dim=-1)
            # Embedding weight is normalized in codebook()

        E = self.codebook()  # (K_no_mask, D)
        K_no_mask = E.shape[0]
        K = self.num_embeddings  # total codes = K_no_mask + 1 (mask)

        # 2. Distance Calculation
        # Squared L2 distances: (N, K_no_mask)
        z_e_sq = (flat_z_e**2).sum(dim=1, keepdim=True)  # (N, 1)
        e_sq = (E**2).sum(dim=1)  # (K_no_mask,)
        distances = z_e_sq - 2 * flat_z_e @ E.t() + e_sq  # (N, K_no_mask)

        # 3. Nearest code indices
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        encodings = F.one_hot(encoding_indices, K).type(flat_z_e.dtype)  # (N, K)
        encodings_no_mask = encodings[:, :K_no_mask]  # (N, K_no_mask)

        dead_codes_count = 0

        # 4. Codebook Restart (Dead Code Revival)
        # Only applied during training and if using EMA (safest context)
        if self.training and self.use_restart and self.use_ema:
            with torch.no_grad():
                # Check usage in current batch
                cluster_usage = encodings_no_mask.sum(0) # (K_no_mask,)
                dead_indices = torch.nonzero(cluster_usage < self.restart_threshold).flatten()
                dead_codes_count = int(dead_indices.numel())

                if dead_indices.numel() > 0:
                    # Randomly sample input vectors from the batch to replace dead codes
                    # We pick random indices from [0, N-1]
                    rand_idx = torch.randint(0, flat_z_e.size(0), (dead_indices.numel(),), device=flat_z_e.device)
                    replacement_vectors = flat_z_e[rand_idx].detach()
                    
                    if self.normalize:
                        replacement_vectors = F.normalize(replacement_vectors, p=2, dim=-1)

                    # Update buffer (embedding) and EMA stats
                    self.embedding[dead_indices].copy_(replacement_vectors)
                    self.ema_embedding[dead_indices].copy_(replacement_vectors)
                    # Reset cluster size to small value (or usage threshold) so it has 'fresh' history
                    self.ema_cluster_size[dead_indices].fill_(self.restart_threshold)

                    # Re-compute quantization with new codes. 
                    # must refresh 'E' so the current batch uses the revived codes
                    E = self.codebook() 
                    # Re-calculate distances with new E
                    e_sq = (E**2).sum(dim=1)
                    distances = z_e_sq - 2 * flat_z_e @ E.t() + e_sq
                    encoding_indices = torch.argmin(distances, dim=1)
                    encodings = F.one_hot(encoding_indices, K).type(flat_z_e.dtype)
                    encodings_no_mask = encodings[:, :K_no_mask]

        # 5. Quantize
        # Note: We must use the updated E from self.embedding if restart happened
        # but for gradient flow consistency in this step, using the E computed earlier is fine.
        flat_z_q = encodings_no_mask @ E  # (N, D)
        z_q = flat_z_q.reshape(z_e_shape)  # (B, ..., D)

        # 6. EMA Updates (Standard VQ-VAE)
        if self.use_ema and self.training:
            with torch.no_grad():
                ema_cluster_size = self.ema_cluster_size * self.decay + (1.0 - self.decay) * encodings_no_mask.sum(0)
                self.ema_cluster_size.copy_(ema_cluster_size)

                n = ema_cluster_size.sum()
                cluster_size = (ema_cluster_size + self.eps) / (n + K_no_mask * self.eps) * n

                ema_embedding = self.ema_embedding * self.decay + (1.0 - self.decay) * (encodings_no_mask.t() @ flat_z_e)
                self.ema_embedding.copy_(ema_embedding)

                # Update the actual embedding buffer
                cluster_size = torch.clamp(cluster_size, min=1.0)
                new_embedding = ema_embedding / cluster_size.unsqueeze(1)
                
                if self.normalize:
                    new_embedding = F.normalize(new_embedding, p=2, dim=-1)
                
                self.embedding.copy_(new_embedding)

        # 7. Loss & Straight-Through
        z_q_st = z_e + (z_q - z_e).detach()

        if self.use_ema:
            vq_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + commitment_loss

        # Reshape indices
        indices = encoding_indices.reshape(*z_e_shape[:-1])
        one_hot = None
        if return_one_hot:
            one_hot = encodings.reshape(*z_e_shape[:-1], K)
        
        # return a dictionnary with 
        # z_q_st: quantized output with straight-through estimator
        # vq_loss: vector quantization loss
        # indices: indices of the selected codes
        # one_hot: one-hot encoding of the selected codes
        # dead_codes_count: how many restart each step
        # ema_cluster_size.min(), .median(), .max()
        # embedding.norm().min/max (even with normalize, check for inf/nan)
        # prop_top_1_token: % of tokens assigned to top-1 code (collapse indicator)
        
        with torch.no_grad():
            usage_counts = encodings_no_mask.sum(0).float()
            usage_total = usage_counts.sum()
            prop_top_1_token = (usage_counts.max() / usage_total).item() if usage_total > 0 else 0.0

            if self.use_ema:
                ema_cluster = self.ema_cluster_size.detach()
                ema_cluster_min = ema_cluster.min().item()
                ema_cluster_median = ema_cluster.median().item()
                ema_cluster_max = ema_cluster.max().item()
            else:
                ema_cluster_min = float('nan')
                ema_cluster_median = float('nan')
                ema_cluster_max = float('nan')

            embedding_norm = E.detach().norm(p=2, dim=-1)
            embedding_norm_min = embedding_norm.min().item()
            embedding_norm_max = embedding_norm.max().item()

        vq_return_dict = {
            'z_q_st': z_q_st,
            'vq_loss': vq_loss,
            'indices': indices,
            'one_hot': one_hot,
            'dead_codes_count': dead_codes_count,
            'ema_cluster_size_min': ema_cluster_min,
            'ema_cluster_size_median': ema_cluster_median,
            'ema_cluster_size_max': ema_cluster_max,
            'embedding_norm_min': embedding_norm_min,
            'embedding_norm_max': embedding_norm_max,
            'prop_top_1_token': prop_top_1_token,
        }

        return z_q_st, vq_loss, indices, one_hot