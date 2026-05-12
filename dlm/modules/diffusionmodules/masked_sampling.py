import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from ...utils import instantiate_from_config, is_rank_zero
from ..logit_process import get_logit_processors
from .discretization import TimeDiscretization
from .masked_process import MaskedDiffusionProcess
from .noise_sampling import NoiseOutput, NoiseSampling
from .utils import sample_categorical


class BaseSampler:
    def __init__(
        self,
        num_inference_steps: int = 1000,
        max_new_tokens: int | None = None,
        discretization_config: dict | None = None,
        noise_sampler_config: dict | None = None,
        logit_processors: dict | list[dict] | None = None,
        # e.g. {"target": "ddm.modules.diffusionmodules.logit_process.Temperature", "params": {"temperature": 1.0}}
        shift_logits: bool = False,  # Proposed in DiffuGPT (https://arxiv.org/abs/2410.17891)
        enable_caching: bool = False,  # Proposed in MDLM (https://arxiv.org/abs/2406.07524)
        noise_removal: bool = False,  # do noise removal after the last step
        temperature: float | None = None,  # Temperature sampling on normalized logits
        verbose: bool = False,
    ):
        if discretization_config is None:
            # memo: MD4 recommends using cosine discretization for models trained with linear schedule
            discretization_config = {
                'target': 'dlm.modules.diffusionmodules.discretization.UniformDiscretization',
            }
        self.discretization: TimeDiscretization = instantiate_from_config(
            discretization_config,
        )
        if noise_sampler_config is None:
            noise_sampler_config = {
                'target': 'dlm.modules.diffusionmodules.noise_sampling.LinearSampling',
            }
        self.noise_sampler: NoiseSampling = instantiate_from_config(
            noise_sampler_config,
        )
        self.max_new_tokens = max_new_tokens
        self.num_inference_steps = num_inference_steps
        self.logit_processors = get_logit_processors(logit_processors) if logit_processors else None
        self.shift_logits = shift_logits
        self.enable_caching = enable_caching
        self.noise_removal = noise_removal
        self.temperature = temperature
        self.verbose = verbose
        self.pad_token_id = None

    def get_alpha(self, t) -> torch.Tensor:
        # sample noise
        alphas: NoiseOutput = self.noise_sampler(t)
        return alphas.alpha

    def get_dalpha(self, t) -> torch.Tensor:
        # sample noise
        alphas: NoiseOutput = self.noise_sampler(t)
        return alphas.dalpha

    def init_xT(
        self,
        input_ids: torch.Tensor | None,
        sequence_length: int,
        num_samples: int = 1,
        all_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _apply_logit_processors(self, logits: torch.Tensor) -> torch.Tensor:
        """
        applied to x_0 = p_{0|t} before sampling from p_{s|t}
        """
        if self.temperature is not None and self.temperature != 1.0:
            if self.temperature <= 0:
                raise ValueError(f'temperature must be > 0, got {self.temperature}')
            original_dtype = logits.dtype
            compute_dtype = (
                torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype
            )
            logits_float = logits.to(dtype=compute_dtype)
            logits = logits_float - torch.logsumexp(logits_float, dim=-1, keepdim=True)
            logits = (logits / self.temperature).to(dtype=original_dtype)

        if self.logit_processors is None:
            return logits
        return self.logit_processors(logits)

    def sampler_step(self, model_output: CausalLMOutput, t, next_t, x) -> torch.Tensor:
        """Perform a single step of the reverse process."""
        raise NotImplementedError

    def last_sampler_step(self, model_output, x) -> torch.Tensor:
        raise NotImplementedError

    def model_forward(self, model, time_type, x, t, noise, **model_kwargs: dict):
        if time_type == 'noise':
            timesteps = self.get_alpha(t)
        elif time_type == 'time':
            timesteps = t
        elif time_type == 'none':
            timesteps = None
        else:
            raise NotImplementedError(f'Unknown time_type: {time_type}')

        if time_type != 'none':
            model_kwargs['timesteps'] = timesteps * torch.ones(x.shape[0], device=x.device)
        
        model_kwargs['noise'] = noise

        return model(x, **model_kwargs)

    @torch.no_grad()
    def __call__(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        max_new_tokens: int | None = None,
        reverse: bool = True,
        num_samples: int = 1,
        verbose=None,
        all_input_ids: torch.Tensor | None = None,
        **model_kwargs,
    ):
        if self.pad_token_id is None:
            assert hasattr(model.config, 'pad_token_id'), 'pad_token_id must be provided'
            self.pad_token_id = model.config.pad_token_id
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens or model.config.max_position_embeddings
        verbose = verbose if verbose is not None else self.verbose
        time_type = getattr(model.config, 'time_type', 'noise')

        x = self.init_xT(
            input_ids,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            all_input_ids=all_input_ids,
        ).to(model.device)
        
        noise = x.clone()
        
        ts = self.discretization(num_inference_steps, reverse=reverse).to(model.device)

        model_output = None
        for i in tqdm(
            range(num_inference_steps),
            desc=f'Sampling with {self.__class__.__name__} for {num_inference_steps} steps',
            disable=not verbose or not is_rank_zero(),
        ):
            t = ts[i]
            next_t = ts[i + 1]
            # Get the model's output
            if model_output is None:
                model_output = self.model_forward(model, time_type, x, t, noise, **model_kwargs)
            x_next, noise = self.sampler_step(model_output, self.get_alpha(t), self.get_alpha(next_t), x, noise)
            # If no new tokens become unmasked or the model is time-independent,
            # reuse the model output
            if not (self.enable_caching and torch.allclose(x_next, x) and (time_type == 'none')):
                model_output = None
            x = x_next

        if self.noise_removal:
            model_output = self.model_forward(model, time_type, x, ts[-1], noise, **model_kwargs)
            x, noise = self.last_sampler_step(model_output, x, noise)
        if self.shift_logits:
            x = x[:, 1:]

        return x


class AbsorbingSampler(BaseSampler):
    """Start from a sequence with all tokens masked."""

    def __init__(self, mask_token_id: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.vocab_size = None
        self.masked_process = MaskedDiffusionProcess(mask_token_id=mask_token_id)

    def init_xT(
        self,
        input_ids: torch.Tensor | None,
        num_samples: int,
        max_new_tokens: int,
        all_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if all_input_ids is not None:
            x_T = all_input_ids
            # replace half of x_t by the mask token
            for i in range(input_ids.shape[0]):
                mask = torch.randperm(x_T.shape[1], device=x_T.device)[: x_T.shape[1] // 2]
                x_T[i, mask] = self.mask_token_id
                return x_T

        if input_ids is None:
            return torch.full(
                (num_samples, max_new_tokens),
                self.mask_token_id,
                dtype=torch.long,
            )
        # prompt_ids might be padded with pad_token_id on the left
        # We want to remove all of the pad_tokens and append mask tokens to the rest of the sequence
        # e.g. pad_token_id = 0, mask_token_id = 1, sequence_length = 5
        # [[0, 0, 4], [4, 5, 6]] -> [[4, 1, 1, 1, 1], [4, 5, 6, 1]]
        sequence_length = input_ids.shape[1] + max_new_tokens
        x_T = torch.full(
            (input_ids.shape[0], sequence_length),
            self.mask_token_id,
            dtype=torch.long,
        )

        # extract the non-pad tokens
        for i in range(input_ids.shape[0]):
            assert self.pad_token_id is not None, 'pad_token_id must be provided'
            non_pad_tokens = input_ids[i][input_ids[i] != self.pad_token_id]
            # append mask tokens to the rest of the sequence
            x_T[i, : non_pad_tokens.shape[0]] = non_pad_tokens
        return x_T

    def sampler_step(self, model_output, alpha, next_alpha, x, noise):
        """Sample a single step of the diffusion process."""
        dtype = torch.float64 if model_output.logits.device.type != 'mps' else torch.float32
        return self.masked_process.sample_bridge(
            xt=x,
            x0_logits=self._apply_logit_processors(model_output.logits.type(dtype)),
            alpha_t=alpha.type(dtype),
            alpha_s=next_alpha.type(dtype),
            shift_logits=self.shift_logits,
        ), noise

    def last_sampler_step(self, model_output, x, noise):
        """Sample the last step of the diffusion process."""
        dtype = torch.float64 if model_output.logits.device.type != 'mps' else torch.float32
        return self.masked_process.sample_last_step(
            xt=x,
            x0_logits=self._apply_logit_processors(model_output.logits.type(dtype)),
        ), noise

    def __call__(self, model, **kwargs):
        if self.mask_token_id is None:
            assert hasattr(model.config, 'mask_token_id'), 'mask_token_id must be provided'
            self.mask_token_id = model.config.mask_token_id
        if self.vocab_size is None:
            self.vocab_size = model.config.vocab_size
        return super().__call__(model, **kwargs)


class AncestralSampler(AbsorbingSampler):
    """Proposed in MD4, Shi et al. (2024)"""

    def sampler_step(self, model_output, alpha, next_alpha, x, noise):
        """Reference:
        - https://github.com/google-deepmind/md4/blob/main/md4/models/diffusion/md4.py#L283
        """
        # Sample for only the masked tokens
        mask_indices = x == self.mask_token_id

        # 1. Get mean of predicted x_0
        logits = self._apply_logit_processors(model_output.logits)
        mu = self.masked_process.subs_parameterization(
            x=x,
            x0_logits=logits,
            mask_indices=mask_indices,
            return_probs=True,
        )

        # 2. calculate the probabilities
        unmask_probs = (next_alpha - alpha) / (1 - alpha)
        one_hot_m = torch.zeros_like(mu, device=mu.device, dtype=mu.dtype)
        one_hot_m[..., self.mask_token_id] = 1
        probs = unmask_probs * mu + (1 - unmask_probs) * one_hot_m
        to_unmask = sample_categorical(probs)
        return torch.where(mask_indices, to_unmask, x), noise
        # sanity check to see if the number of tokens that are the same is equal to the number of tokens that are not masked
        # (x[~mask_indices] == next_x[~mask_indices]).sum(), (~mask_indices).sum()


class AncestralGIDDSampler(AbsorbingSampler):
    """Proposed in GIDD (https://arxiv.org/abs/2503.04482)"""

    def get_probs_at_t(self, probs, alpha):
        """https://github.com/dvruette/gidd/blob/main/gidd/diffusion_process.py#L147"""
        mask_probs = 1 - alpha
        probs = alpha[..., None, None] * probs
        probs[..., self.mask_token_id] = mask_probs
        return probs

    def sampler_step(self, model_output, alpha, next_alpha, x, noise):
        """x_{t-1} ~ q(x_t | x_{t-1}) frac{q(x_{t-1} | hat{x}_0)}{q(x_t | hat{x}_0)}"""
        # Sample for only the masked tokens
        mask_indices = x == self.mask_token_id

        # 1. Get mean of predicted x_0
        logits = self._apply_logit_processors(model_output.logits)
        mu = self.masked_process.subs_parameterization(
            x=x,
            x0_logits=logits,
            mask_indices=mask_indices,
            return_probs=True,
        )

        # 2. forward process
        q_t = self.get_probs_at_t(mu, alpha)
        q_t_1 = self.get_probs_at_t(mu, next_alpha)
        q_t = q_t.gather(-1, x.unsqueeze(-1))

        # compute q(x_t | x_{t-1})
        alpha_next = alpha / next_alpha
        one_hot_m = torch.zeros_like(mu, device=mu.device, dtype=mu.dtype)
        one_hot_m[..., self.mask_token_id] = 1
        q_next = alpha_next * F.one_hot(x, num_classes=mu.shape[-1]) + (1 - alpha_next) * one_hot_m

        probs = q_next * q_t_1 / q_t
        to_unmask = sample_categorical(probs)
        return torch.where(mask_indices, to_unmask, x), noise


class FirstHittingSampler(AbsorbingSampler):
    """Proposed in https://arxiv.org/abs/2409.02908"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = None

    def sampler_step(self, model_output, alpha, next_alpha, x, noise):
        pass
