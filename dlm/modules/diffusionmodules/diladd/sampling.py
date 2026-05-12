import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel

from dlm.transformers.dit import ContinuousDiTModel, JointDiscreteDiTModel
from dlm.transformers.vqvae.lucidrain_vector_quantizer import LucidRainVectorQuantizer
from dlm.utils import is_rank_zero

from ..masked_sampling import AbsorbingSampler
from ..noise_sampling import NoiseOutput


def _decode_indices_to_latents(
    vq,  # either your VectorQuantizer or LucidRainVectorQuantizer wrapper
    indices: torch.Tensor,  # (B, S) int
) -> torch.Tensor:
    """
    Returns z of shape (B, S, D) in *model latent dim* D.
    Prefers lucidrains decode helpers if they exist (safe for codebook_dim != dim).
    """
    # lucidrains wrappers: vq.vqvae exists
    vqvae = getattr(vq, 'vqvae', None)

    if vqvae is not None:
        # Preferred decode path (documented for SimVQ / LFQ)  [oai_citation:2‡GitHub](https://github.com/lucidrains/vector-quantize-pytorch)
        if hasattr(vqvae, 'indices_to_codes'):
            return vqvae.indices_to_codes(indices)

        # Residual LFQ exposes get_output_from_indices (documented)  [oai_citation:3‡GitHub](https://github.com/lucidrains/vector-quantize-pytorch)
        if hasattr(vqvae, 'get_output_from_indices'):
            return vqvae.get_output_from_indices(indices)

        # If neither exists, you *may* be able to access the raw codebook weights,
        # but this can be WRONG if codebook_dim != dim.
        raise RuntimeError(
            "Lucidrains quantizer has no indices_to_codes / get_output_from_indices. "
            "Either add a decode helper or avoid codebook_dim != dim and manually access codebook weights.",
        )

    # fallback: your own class
    E = vq._get_embedding_weight()  # (K_no_mask, D)
    return F.embedding(indices, E)



class DiscreteLatentDDMSampler:
    """Sampler for Discrete Latent Discrete Diffusion Models.

    This sampler performs joint denoising of a token sequence `x` and a
    discrete latent vector `y`.
    """

    def __init__(
        self,
        use_precomputed_latent: bool = False,
        use_encoder_latent: bool = False,
        encoder_strategy: str = 'gumbel',
        temperature: float | None = None,
        x_kwargs: dict | None = None,
        y_kwargs: dict | None = None,
        SEQ: bool = False,
        autoregressive: bool = False,
    ):
        """Initialize the LatentDDMSampler.

        Args:
            core_model (dict): The core model to use.
            SEQ (bool): If True, generate latents first, then generate data conditionally.
            autoregressive (bool): If True, generate data autoregressively token-by-token.
                                  - If SEQ=True: Generate latents first, then x autoregressively conditioned on y.
                                  - If SEQ=False: Joint generation with x autoregressive and y using diffusion.
                                    Requires num_inference_steps = num_inference_steps_latent = sequence_length.

        """
        x_kwargs = dict(x_kwargs) if x_kwargs is not None else {}
        y_kwargs = dict(y_kwargs) if y_kwargs is not None else {}
        if temperature is not None and 'temperature' not in x_kwargs:
            x_kwargs['temperature'] = temperature

        self.x_sampler = AbsorbingSampler(**x_kwargs)
        self.y_sampler = AbsorbingSampler(**y_kwargs)
        self.use_precomputed_latent = use_precomputed_latent
        self.num_inference_steps = self.x_sampler.num_inference_steps
        self.num_inference_steps_latent = self.y_sampler.num_inference_steps
        self.use_encoder_latent = use_encoder_latent
        self.encoder_strategy = encoder_strategy
        self.temperature = temperature
        self.SEQ = SEQ
        self.autoregressive = autoregressive

    def get_timesteps(
        self,
        model_config: dict,
        alphas: NoiseOutput,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Get the timesteps for the token sequence `x`.

        Args:
            model_config: The config of the model.
            alphas: The alphas of the noise sampler.
            t: The timesteps of the noise sampler.

        """
        time_type = getattr(model_config, 'x_time_type', 'time')
        if time_type == 'noise':
            timesteps = alphas.alpha
        elif time_type == 'time':
            timesteps = t
        elif time_type == 'none':
            timesteps = None
        else:
            raise NotImplementedError(f'Unknown time_type: {time_type}')
        return timesteps

    def _one_hot_logits(self, indices: torch.Tensor, num_classes: int, neg_infty: float) -> torch.Tensor:
        logits = torch.full(
            (*indices.shape, num_classes),
            fill_value=neg_infty,
            device=indices.device,
            dtype=torch.float32,
        )
        return logits.scatter_(-1, indices.unsqueeze(-1), 0.0)

    def _subs_parameterization_sampling(self, x0_logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
        """Apply subs parameterization for sampling (without setting unmasked positions).

        During sampling, we don't have the true x_0, so we only:
        1. Set mask token logit to -infty
        2. Apply logsumexp normalization

        Args:
            x0_logits: Raw logits from model (B, S, vocab_size)
            mask_token_id: ID of the mask token

        Returns:
            Normalized logits (B, S, vocab_size)
        """
        neg_infty = self.x_sampler.masked_process.neg_infty
        # Set mask token logit to -infty
        x0_logits = x0_logits.clone()
        # Convert neg_infty to the dtype of x0_logits to avoid overflow
        neg_infty_tensor = torch.tensor(neg_infty, dtype=x0_logits.dtype, device=x0_logits.device)
        x0_logits[..., mask_token_id] = neg_infty_tensor
        # Normalize with logsumexp
        x0_logits = x0_logits - torch.logsumexp(x0_logits, dim=-1, keepdim=True)
        return x0_logits

    def model_forward(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: PreTrainedModel | None,
        x,
        y,
        t_x,
        t_y,
    ):
        t_x = self.get_timesteps(model_config=joint_denoiser.config, alphas=self.x_sampler.noise_sampler(t_x), t=t_x)
        t_y = self.get_timesteps(model_config=joint_denoiser.config, alphas=self.y_sampler.noise_sampler(t_y), t=t_y)
        joint_denoiser_output = joint_denoiser(x, y, t_x, t_y)
        return joint_denoiser_output.logits, joint_denoiser_output.y_pred

    def encode_x_0_to_y_0(
        self,
        x_0: torch.Tensor,
        encoder: torch.nn.Module,
        vector_quantizer: LucidRainVectorQuantizer,
    ):
        """Encode the token sequence `x_0` to the discrete latent vector `y_0`.

        Args:
            x_0: The token sequence to encode.
            encoder: The encoder model.
            vector_quantizer: The vector quantizer model.

        Returns:
            The discrete latent vector `y_0`.
        """
        z_q_st, y_0, y_0_one_hot = None, None, None
        if self.encoder_strategy == 'gumbel':
            y_0_logits = encoder(x_0).y_pred  # logits
            latent_mask_id = self.y_sampler.mask_token_id
            y_0_logits[..., latent_mask_id] = -1e6
            y_0_logits = y_0_logits - torch.logsumexp(y_0_logits, dim=-1, keepdim=True)
            y_0_prob = y_0_logits.exp()
            y_0 = torch.argmax(y_0_prob, dim=-1)
            y_0_one_hot = F.one_hot(y_0, num_classes=y_0_prob.shape[-1]).float()
            # precomputed_y_0 = sample_categorical(y_0_prob)  # (B, S)
        elif self.encoder_strategy == 'vqvae':
            encoded_y_0 = encoder(x_0).y_pred  # (B, S, D) or (B, S*D) depending on encoder architecture
            encoded_y_0 = encoded_y_0.view(encoded_y_0.size(0), -1, vector_quantizer.vqvae_latent_dim)  # ensures (B, S, D)
            z_q_st, vq_loss, y_0, y_0_one_hot = vector_quantizer(encoded_y_0, mask_positions=None, return_one_hot=True)  # (B, S)
        else:
            raise NotImplementedError(f'Unknown encoder_strategy: {self.encoder_strategy}')
        neg_infty = self.y_sampler.masked_process.neg_infty
        num_classes = y_0_one_hot.shape[-1]
        precomputed_y_0_logits = self._one_hot_logits(y_0, num_classes, neg_infty)
        return z_q_st, y_0, y_0_one_hot, precomputed_y_0_logits

    def _build_input_from_y_t(
        self,
        y_t: torch.Tensor,  # (B, S)
        joint_denoiser: JointDiscreteDiTModel,
        vector_quantizer: LucidRainVectorQuantizer,  # your class
    ) -> torch.Tensor:
        if self.encoder_strategy == 'vqvae':
            """
            Map discrete codes y_t to continuous latents z_t.
            Non-mask codes use VQ embedding; mask code uses z_mask.
            """
            assert vector_quantizer is not None
            
            # E = vector_quantizer._get_embedding_weight()  # (K_no_mask, D)
            # K_no_mask, D = E.shape

            mask_token_id = self.y_sampler.mask_token_id  # should be K_no_mask

            # mask positions
            mask = y_t == mask_token_id  # (B, S), bool

            # for non-masked positions, indices are < K_no_mask
            # safe: send masked indices to 0 before embedding
            y_embed_idx = y_t.masked_fill(mask, 0)  # (B, S)
            z_t = _decode_indices_to_latents(vector_quantizer, y_embed_idx)  # (B, S, D)


            # Overwrite masked positions with learnable mask vector
            D = z_t.shape[-1]
            z_mask = vector_quantizer.z_mask.view(1, 1, D)
            z_t = torch.where(mask.unsqueeze(-1), z_mask.expand_as(z_t), z_t)
            y_input = z_t  # (B, S, D)
        else:
            y_input = torch.nn.functional.one_hot(
                y_t,
                num_classes=joint_denoiser.patch_dim,
            ).float()

        return y_input

    def _sample_latents_only(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: ContinuousDiTModel | None,
        vector_quantizer: LucidRainVectorQuantizer | None,
        y_T: torch.Tensor,
        num_inference_steps_latent: int,
        x_seq_len: int,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Sample latents y_0 from noise y_T using the latent denoiser.

        Args:
            joint_denoiser: The joint denoiser model.
            latent_denoiser: Optional separate latent denoiser.
            vector_quantizer: Vector quantizer for discrete latents.
            y_T: Initial noise for latents (B, S_y).
            num_inference_steps_latent: Number of sampling steps for latents.
            x_seq_len: Length of the x sequence used during training (for RoPE alignment).
            verbose: Whether to show progress bar.

        Returns:
            y_0: Denoised latents (B, S_y).
        """
        device = y_T.device
        ts_latent = self.y_sampler.discretization(num_inference_steps_latent, reverse=True).to(device)
        y_t = y_T

        # Create a dummy x_t (all mask tokens) matching the training x length,
        # so RoPE positions for y match training-time offsets.
        num_samples = y_T.size(0)
        dummy_x_t = torch.full(
            (num_samples, x_seq_len),
            fill_value=self.x_sampler.mask_token_id,
            device=device,
            dtype=torch.long,
        )

        for i, (t, next_t) in tqdm(
            enumerate(zip(ts_latent[:-1], ts_latent[1:], strict=False)),
            desc='Sampling latents only',
            disable=not verbose or not is_rank_zero(),
            total=len(ts_latent) - 1,
        ):
            # Get latent input
            y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)

            # Forward through model to get y_logits
            # Use dummy x_t and t_x=0 since we only care about latents
            t_x = torch.zeros_like(t)
            t_y = t
            _, y_logits = self.model_forward(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x=dummy_x_t,
                y=y_input,
                t_x=t_x,
                t_y=t_y,
            )

            # Sample next latent state
            alphas_y = self.y_sampler.noise_sampler(t).alpha
            next_alphas_y = self.y_sampler.noise_sampler(next_t).alpha

            dtype = torch.float64 if y_logits.device.type != 'mps' else torch.float32
            y_t = self.y_sampler.masked_process.sample_bridge(
                xt=y_t,
                x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
                alpha_t=alphas_y.type(dtype),
                alpha_s=next_alphas_y.type(dtype),
                shift_logits=self.y_sampler.shift_logits,
            )

        # Final denoising step if noise_removal is enabled
        if self.y_sampler.noise_removal:
            y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
            t_x = torch.zeros_like(ts_latent[-1])
            t_y = ts_latent[-1]
            _, y_logits = self.model_forward(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x=dummy_x_t,
                y=y_input,
                t_x=t_x,
                t_y=t_y,
            )

            dtype = torch.float64 if y_logits.device.type != 'mps' else torch.float32
            y_t = self.y_sampler.masked_process.sample_last_step(
                xt=y_t,
                x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
            )

        return y_t

    @torch.no_grad()
    def __call__(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        encoder: torch.nn.Module,
        latent_denoiser: ContinuousDiTModel | None,
        vector_quantizer: LucidRainVectorQuantizer | None,
        input_ids: torch.Tensor | None = None,
        batch: dict[str, torch.Tensor] | None = None,
        num_inference_steps: int | None = None,
        num_inference_steps_latent: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        token_proportion_to_generate: float | None = None,
        sample_latents_only: bool = False,
        all_input_ids: torch.Tensor | None = None,
        verbose: bool = True,
        **kwargs: dict,
    ):
        def _scalar_or_mean(t: torch.Tensor) -> float:
            return float(t.item()) if t.numel() == 1 else float(t.mean().item())

        def _print_nonfinite_logits(
            name: str,
            logits: torch.Tensor,
            step: int,
            t_x: torch.Tensor,
            next_t_x: torch.Tensor,
            t_y: torch.Tensor,
            next_t_y: torch.Tensor,
        ) -> None:
            if torch.isfinite(logits).all():
                return
            num_total = logits.numel()
            num_nan = torch.isnan(logits).sum().item()
            num_inf = torch.isinf(logits).sum().item()
            num_finite = torch.isfinite(logits).sum().item()
            finite_vals = logits[torch.isfinite(logits)]
            if num_finite > 0:
                finite_min = finite_vals.min().item()
                finite_max = finite_vals.max().item()
                finite_mean = finite_vals.mean().item()
            else:
                finite_min = None
                finite_max = None
                finite_mean = None
            rank = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            prefix = f'[rank{rank}] ' if rank is not None else ''
            print(
                f'{prefix}Non-finite {name} logits in DiscreteLatentDDMSampler at step={step}: '
                f'shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device} '
                f'nonfinite={num_total - num_finite}/{num_total} nan={num_nan} inf={num_inf} '
                f'finite_min={finite_min} finite_max={finite_max} finite_mean={finite_mean} '
                f't_x={_scalar_or_mean(t_x)} next_t_x={_scalar_or_mean(next_t_x)} '
                f't_y={_scalar_or_mean(t_y)} next_t_y={_scalar_or_mean(next_t_y)}',
            )

        # 1. Setup sampling parameters
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if num_inference_steps_latent is None:
            num_inference_steps_latent = self.num_inference_steps_latent
        if temperature is None:
            temperature = self.temperature
        self.x_sampler.temperature = temperature
        if max_new_tokens is None:
            max_new_tokens = self.x_sampler.max_new_tokens
            if max_new_tokens is None:
                raise ValueError('max_new_tokens cannot be None')

        # Determine num_samples from input_ids, batch, or kwargs
        # When use_encoder_latent=True, we need to match the batch size since we'll encode batch['input_ids']
        if input_ids is not None:
            num_samples = input_ids.size(0)
        elif batch is not None and 'input_ids' in batch:
            num_samples = batch['input_ids'].size(0)
        else:
            num_samples = kwargs.get('num_samples', 1)

        device = joint_denoiser.device

        # 2. Initialize x_T and y_T
        x_T = self.x_sampler.init_xT(input_ids, num_samples, max_new_tokens, all_input_ids).to(device)
        y_T = self.y_sampler.init_xT(None, num_samples, self.y_sampler.max_new_tokens, all_input_ids).to(device)

        # 3. Get time discretization
        ts = self.x_sampler.discretization(num_inference_steps, reverse=True).to(device)
        ts_latent = self.y_sampler.discretization(num_inference_steps_latent, reverse=True).to(device)
        if token_proportion_to_generate is not None:
            ts = ts[int(token_proportion_to_generate * len(ts)) :]
            ts_latent = ts_latent[int(token_proportion_to_generate * len(ts_latent)) :]

        # if latent_process is not None, sample all y_t at once
        precomputed_y_0_logits = None
        if self.use_precomputed_latent:
            precomputed_y_0 = batch['latent']
            num_classes = joint_denoiser.patch_dim if self.encoder_strategy == 'gumbel' else vector_quantizer.codebook_size
            neg_infty = self.y_sampler.masked_process.neg_infty
            precomputed_y_0_logits = self._one_hot_logits(precomputed_y_0, num_classes, neg_infty)
        if self.use_encoder_latent:
            _, _, _, precomputed_y_0_logits = self.encode_x_0_to_y_0(
                x_0=batch['input_ids'],
                encoder=encoder,
                vector_quantizer=vector_quantizer,
            )

        # Autoregressive mode without SEQ: joint generation with x autoregressive, y diffusion
        if self.autoregressive and not self.SEQ:
            # In this mode, num_inference_steps must equal sequence_length
            seq_length = x_T.size(1)
            assert num_inference_steps == seq_length, (
                f'For autoregressive mode with SEQ=False, num_inference_steps ({num_inference_steps}) '
                f'must equal sequence_length ({seq_length})'
            )
            assert num_inference_steps_latent == seq_length, (
                f'For autoregressive mode with SEQ=False, num_inference_steps_latent ({num_inference_steps_latent}) '
                f'must equal sequence_length ({seq_length})'
            )

            x_t = x_T.clone()
            y_t = y_T.clone()

            # Each step: denoise one position of x autoregressively, and one step of y diffusion
            for token_idx, (t, next_t) in tqdm(
                enumerate(zip(ts[:-1], ts[1:], strict=False)),
                desc='Autoregressive generation (joint with latent diffusion)',
                disable=not verbose or not is_rank_zero(),
                total=len(ts) - 1,
            ):
                # Get latent input
                y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)

                # Forward pass
                t_latent = ts_latent[token_idx]
                x_logits, y_logits = self.model_forward(
                    joint_denoiser=joint_denoiser,
                    latent_denoiser=latent_denoiser,
                    x=x_t,
                    y=y_input,
                    t_x=t,
                    t_y=t_latent,
                )

                # For x: apply subs parameterization and sample token at position token_idx
                x_logits = self._subs_parameterization_sampling(x_logits, self.x_sampler.mask_token_id)
                token_logits = x_logits[:, token_idx, :]  # (B, vocab_size)
                token_logits = self.x_sampler._apply_logit_processors(token_logits)
                probs = F.softmax(token_logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                x_t[:, token_idx] = sampled_token

                # For y: normal diffusion bridge sampling
                next_t_latent = ts_latent[min(token_idx + 1, len(ts_latent) - 1)]
                alphas_y = self.y_sampler.noise_sampler(t_latent).alpha
                next_alphas_y = self.y_sampler.noise_sampler(next_t_latent).alpha

                dtype = torch.float64 if y_logits.device.type != 'mps' else torch.float32
                y_t = self.y_sampler.masked_process.sample_bridge(
                    xt=y_t,
                    x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
                    alpha_t=alphas_y.type(dtype),
                    alpha_s=next_alphas_y.type(dtype),
                    shift_logits=self.y_sampler.shift_logits,
                )

            # Final denoising for y if noise_removal is enabled
            if self.y_sampler.noise_removal:
                y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
                x_logits, y_logits = self.model_forward(
                    joint_denoiser=joint_denoiser,
                    latent_denoiser=latent_denoiser,
                    x=x_t,
                    y=y_input,
                    t_x=ts[-1],
                    t_y=ts_latent[-1],
                )

                dtype = torch.float64 if y_logits.device.type != 'mps' else torch.float32
                y_t = self.y_sampler.masked_process.sample_last_step(
                    xt=y_t,
                    x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
                )

            return x_t, y_t

        # SEQ mode: generate latents first, then data conditionally
        if self.SEQ:
            # Step 1: Generate latents y_0 first
            if precomputed_y_0_logits is None:
                y_t = self._sample_latents_only(
                    joint_denoiser=joint_denoiser,
                    latent_denoiser=latent_denoiser,
                    vector_quantizer=vector_quantizer,
                    y_T=y_T,
                    num_inference_steps_latent=num_inference_steps_latent,
                    x_seq_len=x_T.size(1),
                    verbose=verbose,
                )
                # Convert y_t to precomputed_y_0_logits for conditioning
                num_classes = joint_denoiser.patch_dim if self.encoder_strategy == 'gumbel' else vector_quantizer.codebook_size
                neg_infty = self.y_sampler.masked_process.neg_infty
                precomputed_y_0_logits = self._one_hot_logits(y_t, num_classes, neg_infty)
            else:
                # Use precomputed latents
                y_t = torch.argmax(precomputed_y_0_logits, dim=-1)

            # Step 2: Generate data x_0 conditionally on y_0
            if self.autoregressive:
                # Autoregressive generation: one token at a time, left-to-right
                x_t = x_T.clone()
                seq_length = x_T.size(1)

                for token_idx in tqdm(
                    range(seq_length),
                    desc='Autoregressive generation',
                    disable=not verbose or not is_rank_zero(),
                ):
                    # Single forward pass with current state (previous tokens unmasked, future masked)
                    y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
                    t_x = torch.zeros(x_t.size(0), device=device)  # Clean timestep for data
                    t_y = torch.zeros(x_t.size(0), device=device)  # Clean timestep for latents

                    x_logits, _ = self.model_forward(
                        joint_denoiser=joint_denoiser,
                        latent_denoiser=latent_denoiser,
                        x=x_t,
                        y=y_input,
                        t_x=t_x,
                        t_y=t_y,
                    )

                    # Apply subs parameterization: set mask token logit to -infty and logsumexp normalize
                    x_logits = self._subs_parameterization_sampling(x_logits, self.x_sampler.mask_token_id)

                    # Sample token at position token_idx from logits
                    token_logits = x_logits[:, token_idx, :]  # (B, vocab_size)
                    token_logits = self.x_sampler._apply_logit_processors(token_logits)

                    # Sample from categorical distribution
                    probs = F.softmax(token_logits, dim=-1)
                    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

                    # Update x_t: unmask the current position
                    x_t[:, token_idx] = sampled_token

                return x_t, y_t

            # Non-autoregressive: generate all data tokens jointly conditioned on y_0
            x_t = x_T
            for i, (t, next_t) in tqdm(
                enumerate(zip(ts[:-1], ts[1:], strict=False)),
                desc=f'Sampling data conditioned on latents for {num_inference_steps} steps',
                disable=not verbose or not is_rank_zero(),
                total=len(ts) - 1,
            ):
                y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
                x_logits, _ = self.model_forward(
                    joint_denoiser=joint_denoiser,
                    latent_denoiser=latent_denoiser,
                    x=x_t,
                    y=y_input,
                    t_x=t,
                    t_y=torch.zeros_like(t),  # Latents are clean (t_y=0)
                )

                alphas_x = self.x_sampler.noise_sampler(t).alpha
                next_alphas_x = self.x_sampler.noise_sampler(next_t).alpha
                _print_nonfinite_logits('x', x_logits, i, t, next_t, torch.zeros_like(t), torch.zeros_like(t))

                dtype = torch.float64 if x_logits.device.type != 'mps' else torch.float32
                x_t = self.x_sampler.masked_process.sample_bridge(
                    xt=x_t,
                    x0_logits=self.x_sampler._apply_logit_processors(x_logits.type(dtype)),
                    alpha_t=alphas_x.type(dtype),
                    alpha_s=next_alphas_x.type(dtype),
                    shift_logits=self.x_sampler.shift_logits,
                )

            if self.x_sampler.noise_removal:
                y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
                x_logits, _ = self.model_forward(
                    joint_denoiser=joint_denoiser,
                    latent_denoiser=latent_denoiser,
                    x=x_t,
                    y=y_input,
                    t_x=ts[-1],
                    t_y=torch.zeros_like(ts[-1]),
                )

                dtype = torch.float64 if x_logits.device.type != 'mps' else torch.float32
                x_t = self.x_sampler.masked_process.sample_last_step(
                    xt=x_t,
                    x0_logits=self.x_sampler._apply_logit_processors(x_logits.type(dtype)),
                )

            return x_t, y_t

        # Default: Joint sampling (original behavior)
        T = ts[0]
        t0 = ts[-1]
        next_ts = ts[1:]
        x_t = x_T
        y_t = y_T
        for i, (t, next_t) in tqdm(
            enumerate(zip(ts[:-1], next_ts, strict=False)),
            desc=f'Sampling with {self.__class__.__name__} for {num_inference_steps} steps',
            disable=not verbose or not is_rank_zero(),
            total=len(ts) - 1,
        ):
            # get latent input to denoiser, depending on encoder_strategy
            y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
            x_logits, y_logits = self.model_forward(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x=x_t,
                y=y_input,
                t_x=ts[i],
                t_y=ts_latent[min(i, len(ts_latent) - 1)],
            )
            y_logits = precomputed_y_0_logits if precomputed_y_0_logits is not None else y_logits

            alphas_x = self.x_sampler.noise_sampler(t).alpha
            next_alphas_x = self.x_sampler.noise_sampler(next_t).alpha
            t_latent = ts_latent[min(i, len(ts_latent) - 1)]
            next_t_latent = ts_latent[min(i + 1, len(ts_latent) - 1)]
            alphas_y = self.y_sampler.noise_sampler(t_latent).alpha
            next_alphas_y = self.y_sampler.noise_sampler(next_t_latent).alpha
            _print_nonfinite_logits('x', x_logits, i, t, next_t, t_latent, next_t_latent)
            _print_nonfinite_logits('y', y_logits, i, t, next_t, t_latent, next_t_latent)

            dtype = torch.float64 if x_logits.device.type != 'mps' else torch.float32
            x_t = self.x_sampler.masked_process.sample_bridge(
                xt=x_t,
                x0_logits=self.x_sampler._apply_logit_processors(x_logits.type(dtype)),
                alpha_t=alphas_x.type(dtype),
                alpha_s=next_alphas_x.type(dtype),
                shift_logits=self.x_sampler.shift_logits,
            )
            y_t = self.y_sampler.masked_process.sample_bridge(
                xt=y_t,
                x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
                alpha_t=alphas_y.type(dtype),
                alpha_s=next_alphas_y.type(dtype),
                shift_logits=self.y_sampler.shift_logits,
            )

        if self.x_sampler.noise_removal:
            y_input = self._build_input_from_y_t(y_t, joint_denoiser, vector_quantizer)
            x_logits, y_logits = self.model_forward(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x=x_t,
                y=y_input,
                t_x=ts[i],
                t_y=ts_latent[min(i, len(ts_latent) - 1)],
            )

            y_logits = precomputed_y_0_logits if precomputed_y_0_logits is not None else y_logits

            dtype = torch.float64 if x_logits.device.type != 'mps' else torch.float32
            x_t = self.x_sampler.masked_process.sample_last_step(
                xt=x_t,
                x0_logits=self.x_sampler._apply_logit_processors(x_logits.type(dtype)),
            )
            y_t = self.y_sampler.masked_process.sample_last_step(
                xt=y_t,
                x0_logits=self.y_sampler._apply_logit_processors(y_logits.type(dtype)),
            )

        return x_t, y_t
