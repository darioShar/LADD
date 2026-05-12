import torch

from dlm.transformers.dit import ContinuousDiTModel, JointDiscreteDiTModel

from ....utils import instantiate_from_config
from ..masked_sampling import AbsorbingSampler
from .latentprocess import LatentProcess


class LatentDDMSampler(AbsorbingSampler):
    """Sampler for Latent Discrete Diffusion Models.

    This sampler performs joint denoising of a token sequence `x` and a
    continuous latent vector `y`.
    """

    def __init__(
        self,
        core_model_config: dict,
        use_precomputed_latent: bool = False,
        num_inference_steps_latent: int | None = None,
        use_encoder_latent: bool = False,
        temperature: float | None = None,
        **kwargs,
    ):
        """Initialize the LatentDDMSampler.

        Args:
            core_model (dict): The core model to use.

        """
        super().__init__(**kwargs)
        self.core_model: LatentProcess = instantiate_from_config(core_model_config)
        self.use_precomputed_latent = use_precomputed_latent
        self.num_inference_steps_latent = (
            num_inference_steps_latent if num_inference_steps_latent is not None else self.num_inference_steps
        )
        self.use_encoder_latent = use_encoder_latent
        self.temperature = temperature

    @torch.no_grad()
    def __call__(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        encoder: torch.nn.Module,
        latent_denoiser: ContinuousDiTModel | None,
        input_ids: torch.Tensor | None = None,
        batch: dict[str, torch.Tensor] | None = None,
        num_inference_steps: int | None = None,
        num_inference_steps_latent: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        token_proportion_to_generate: float | None = None,
        sample_latents_only: bool = False,
        all_input_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        try:
            # 1. Setup sampling parameters
            if num_inference_steps is None:
                num_inference_steps = self.num_inference_steps
            if num_inference_steps_latent is None:
                num_inference_steps_latent = self.num_inference_steps_latent
            if temperature is None:
                temperature = self.temperature
            if max_new_tokens is None:
                max_new_tokens = self.max_new_tokens
                if max_new_tokens is None:
                    raise ValueError('max_new_tokens cannot be None')

            num_samples = input_ids.size(0) if input_ids is not None else kwargs.get('num_samples', 1)
            device = joint_denoiser.device

            # 2. Initialize x_T and y_T
            x_T = self.init_xT(input_ids, num_samples, max_new_tokens, all_input_ids).to(device)

            # y_T is pure Gaussian noise
            y_T = torch.randn(num_samples, self.core_model.y_latent_dim, device=device, dtype=joint_denoiser.dtype)

            # 3. Get time discretization
            ts = self.discretization(num_inference_steps, reverse=True).to(device)
            ts_latent = self.discretization(num_inference_steps_latent, reverse=True).to(device)
            if token_proportion_to_generate is not None:
                ts = ts[int(token_proportion_to_generate * len(ts)) :]
                ts_latent = ts_latent[int(token_proportion_to_generate * len(ts_latent)) :]

            # if latent_process is not None, sample all y_t at once
            precomputed_y_0 = None
            if self.use_precomputed_latent:
                precomputed_y_0 = batch['latent']
            if self.use_encoder_latent:
                encoded_y_0 = self.core_model.get_y_0_encoded(encoder, batch['input_ids'])
                precomputed_y_0 = encoded_y_0['mean'] + encoded_y_0['std'] * torch.randn_like(encoded_y_0['std'])

            x_0, y_0 = self.core_model.sample_backward(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x_T=x_T,
                y_T=y_T,
                ts=ts,
                ts_latent=ts_latent,
                x_sampler_step=self.sampler_step,
                x_last_sampler_step=self.last_sampler_step,
                precomputed_y_0=precomputed_y_0,
                sample_latents_only=sample_latents_only,
                temperature=temperature,
            )

            return x_0, y_0

        except Exception as e:
            print(f'ERROR in LatentDDMSampler: {type(e).__name__}: {e}')
            import traceback

            traceback.print_exc()
            return None, None
