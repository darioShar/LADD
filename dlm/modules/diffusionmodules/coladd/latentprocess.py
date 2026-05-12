import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PretrainedConfig

from dlm.transformers.dit import ContinuousDiTModel, JointDiscreteDiTModel, Output
from dlm.utils import is_rank_zero

from ....utils import instantiate_from_config
from ..continuous_noise_sampling import ContinuousNoiseSampling
from ..ddpm import DDPMProcess
from ..masked_process import MaskedDiffusionProcess
from ..noise_sampling import NoiseOutput, NoiseSampling


class LatentProcess:
    def __init__(
        self,
        y_latent_dim: int,
        # Config dictionaries for noise samplers
        x_noise_sampler_config: dict,
        # Config dictionary for continuous noise sampler
        y_noise_sampler_config: dict,
        mask_token_id,
        training_stage,
        latent_strategy: dict,
        prediction_type: str = 'y_0',  # 'y_0' or 'eps'
        ddim: bool = False,
        all_zero_latent: bool = False,
        neg_infty=-1e6,
    ):
        self.y_latent_dim = y_latent_dim
        # Config dictionaries that need to be instantiated
        self.x_noise_sampler: NoiseSampling = instantiate_from_config(x_noise_sampler_config)
        self.y_noise_sampler: ContinuousNoiseSampling = instantiate_from_config(y_noise_sampler_config)
        self.mask_token_id = mask_token_id
        self.training_stage = training_stage
        self.neg_infty = neg_infty
        self.masked_process = MaskedDiffusionProcess(
            mask_token_id=mask_token_id,
            neg_infty=neg_infty,
        )
        self.ddpm = DDPMProcess(continuous_noise_sampling=self.y_noise_sampler)
        self.latent_strategy = latent_strategy
        self.prediction_type = prediction_type
        self.ddim = ddim
        self.all_zero_latent = all_zero_latent

    def get_y_0_encoded(self, encoder, x_0, **encoder_kwargs):
        y_0_encoded = {}
        if self.all_zero_latent:
            # make y_pred the same type as the encoder parameters
            dtype = next(encoder.parameters()).dtype
            y_pred = torch.zeros(x_0.size(0), self.y_latent_dim, device=x_0.device, dtype=dtype)
            y_std = torch.zeros_like(y_pred)
        else:
            encoder_output = encoder(x_0, **encoder_kwargs)
            y_pred = encoder_output.y_pred
            y_std = encoder_output.y_std

        y_0_encoded['mean'] = y_pred
        if self.latent_strategy['variance'] == 'learnable':
            y_0_encoded['std'] = y_std
        elif self.latent_strategy['variance'] == 'fixed':
            y_0_encoded['std'] = self.latent_strategy['fixed_sigma_latent'] * torch.ones_like(y_0_encoded['mean'])
        elif self.latent_strategy['variance'] is None:
            y_0_encoded['std'] = torch.zeros_like(y_0_encoded['mean'])
        else:
            raise NotImplementedError(f'Unknown latent variance strategy: {self.latent_strategy["variance"]}')
        return y_0_encoded

    def encode_to_latent(
        self,
        batch,
        encoder,
        x_0: torch.Tensor,
        use_p_r: bool = False,
        use_precomputed_latent: bool = False,
        use_encoder_supervision: bool = False,
        masked_tokens_pos: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        training: bool = True,
    ):
        y_0 = {}
        y_0_encoded = {}
        y_0_real = {}
        encode_x_0 = True

        p_r = None
        if use_p_r:
            assert masked_tokens_pos is not None, 'masked_tokens_pos must be provided when use_p_r is True'
            # if in train mode, sample p_r uniform[0, 0.9] for each sequence, yielding a vector of size x_0.shape[0].
            # We will zero the corresponding masked inputs/ encoder outputs.
            p_r = (
                torch.rand(x_0.shape[0], device=x_0.device) * 0.9
                if training
                else torch.ones(x_0.shape[0], device=x_0.device)
            )

        if use_precomputed_latent:
            y_0_real['mean'] = batch['latent']
            y_0_real['std'] = torch.zeros_like(y_0_real['mean'])
            B, S = x_0.shape
            # post-process the embedding
            if hasattr(encoder, 'post_process_hidden_states'):
                y_0['mean'] = encoder.post_process_hidden_states(y_0['mean'].reshape(B, S, -1), attention_mask)
                y_0['mean'] = y_0['mean'].reshape(B, -1)
            y_0 = y_0_real.copy()
            encode_x_0 = use_encoder_supervision
        if encode_x_0:
            y_0_encoded = self.get_y_0_encoded(
                encoder,
                x_0,
                masked_tokens_pos=masked_tokens_pos,
                p_r=p_r,
                attention_mask=attention_mask,
            )
            if not use_precomputed_latent:
                y_0 = y_0_encoded
            # SANITY CHECK: output of encoder == precompute_latent
            # assert torch.allclose(y_0_real, y_0_encoded), 'Encoder output does not match precomputed latent'

        return {'y_0': y_0, 'y_0_encoded': y_0_encoded, 'y_0_real': y_0_real}

    def get_latent_loss_t(self, y_0_mean, y_0_std, y_0, model_pred_mean, model_pred_std, t, z_t, eps):
        # compute L_{t-1}^y loss
        loss_dict_y = {}

        # handle the case where t = 0 (L_0^y + L_{encoder})
        t_zero_mask = t < eps
        loss_weight_t_0_y_0 = None
        loss_weight_t_0_eps = None
        if t_zero_mask.any() and (y_0_std is not None) and (self.latent_strategy['variance'] is not None):
            if self.latent_strategy['variance'] == 'fixed':
                loss_weight_t_0_y_0 = 1 / (2 * y_0_std[t_zero_mask] ** 2)
                abar_sq = self.y_noise_sampler.alpha_bar_squared(t[t_zero_mask])
                sbar_sq = self.y_noise_sampler.sigma_bar_squared(t[t_zero_mask])
                loss_weight_t_0_eps = (abar_sq / sbar_sq) * loss_weight_t_0_y_0
            else:
                raise NotImplementedError(
                    f'Loss weight for t=0 with non-fixed variance is not implemented. Got {self.latent_strategy["variance"]}',
                )

        if self.prediction_type == 'eps':
            mse = F.mse_loss(model_pred_mean, z_t, reduction='none')
            loss_weight = self.y_noise_sampler.loss_weight_eps_pred(
                t,
                t - eps,
            )  # emulates T= 1 / eps
            # replace all loss_weight associated to t < eps with loss_weight_t_0_eps
            if loss_weight_t_0_eps is not None:
                loss_weight[t_zero_mask] = loss_weight_t_0_eps
        elif self.prediction_type == 'y_0':
            mse = F.mse_loss(model_pred_mean, y_0, reduction='none')
            loss_weight = self.y_noise_sampler.loss_weight_y_0_pred(
                t,
                t - eps,
            )  # emulates T= 1 / eps
            # replace all loss_weight associated to t < eps with loss_weight_t_0_y_0
            if loss_weight_t_0_y_0 is not None:
                loss_weight[t_zero_mask] = loss_weight_t_0_y_0
        else:
            raise NotImplementedError(f'Unknown prediction type: {self.prediction_type}')

        y_latent_dim = mse.shape[1]
        nll = mse.sum(dim=-1) * loss_weight
        nll[t_zero_mask] -= 0.5 * y_latent_dim  # for t = 0, add the constant term
        nll = nll.mean()
        elbo_t = nll * ((1 - eps) / eps)
        loss_t = elbo_t / y_latent_dim
        loss_dict_y['loss_y_t'] = mse.mean()  # loss_t
        loss_dict_y['elbo_y_t'] = elbo_t  # must multiply by T - 1 (sum instead of monte-carlo averaging)
        return loss_dict_y

    def get_timesteps_x(
        self,
        model_config: dict,
        alphas_x: NoiseOutput,
        t_x: torch.Tensor,
    ) -> torch.Tensor:
        """Get the timesteps for the token sequence `x`.

        Args:
            model_config: The config of the model.
            alphas: The alphas of the noise sampler.
            t_x: The timesteps of the noise sampler.

        """
        x_time_type = getattr(model_config, 'x_time_type', 'time')
        if x_time_type == 'noise':
            timesteps_x = alphas_x.alpha
        elif x_time_type == 'time':
            timesteps_x = t_x
        elif x_time_type == 'none':
            timesteps_x = None
        else:
            raise NotImplementedError(f'Unknown x_time_type: {x_time_type}')
        return timesteps_x

    def get_timesteps_y(
        self,
        t_y: torch.Tensor,
    ) -> torch.Tensor:
        return t_y

    def _joint_denoiser_forward_with_temperature(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        timesteps_x: torch.Tensor | None,
        timesteps_y: torch.Tensor,
        temperature: float | None = None,
    ) -> Output:
        output: Output = joint_denoiser(
            x_t,
            y_t,
            timesteps_x=timesteps_x,
            timesteps_y=timesteps_y,
        )
        
        if temperature is None or temperature == 1.0:
            return output

        log_probs = F.log_softmax(output.logits.float(), dim=-1)
        scaled_log_probs = log_probs / temperature
        
        return Output(
            logits=scaled_log_probs,
            y_pred=output.y_pred,
            y_std=output.y_std,
        )

    def sample_forward(
        self,
        batch,
        encoder,
        x_0: torch.Tensor,
        t_x,
        t_y,
        maskable_mask: torch.Tensor,
        use_p_r: bool = False,
        use_precomputed_latent: bool = False,
        use_encoder_supervision: bool = False,
        model_config: PretrainedConfig | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        alphas_x = self.x_noise_sampler(t_x)
        x_t = self.masked_process.sample_forward(x_0, alphas_x, maskable_mask, model_config)
        attention_mask = batch.get('attention_mask')
        encode_output = self.encode_to_latent(
            batch,
            encoder,
            x_0,
            masked_tokens_pos= (x_0 != x_t),
            use_p_r=use_p_r,
            use_precomputed_latent=use_precomputed_latent,
            use_encoder_supervision=use_encoder_supervision,
            attention_mask=attention_mask,
            training=training,
        )
        y_0_mean = encode_output['y_0']['mean']
        y_0_std = encode_output['y_0']['std']
        y_0 = y_0_mean + y_0_std * torch.randn_like(y_0_mean)
        y_t, z_t = self.ddpm.sample_forward(y_0, t_y)
        return x_t, y_0, encode_output, y_t, z_t

    def model_forward(
        self,
        joint_denoiser,
        latent_denoiser,
        x: torch.Tensor,
        y: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        y_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            joint_denoiser: The joint denoiser model.
            latent_denoiser: The latent denoiser model.
            x: The input token sequence.
            y: The latent representation.
            t: The timesteps.
            y_0: The initial latent representation (optional).

        Returns:
            logits: The logits for the token sequence.
            y_pred: The predicted latent representation.

        """
        raise NotImplementedError('model_forward method is not implemented.')

    def sample_backward(
        self,
        joint_denoiser,
        latent_denoiser,
        x_T,
        y_T,
        ts,
        ts_latent,
        x_sampler_step,
        x_last_sampler_step,
        precomputed_y_0,
        sample_latents_only,
        temperature: float | None = None,
    ):
        raise NotImplementedError('sample_backward method is not implemented.')

    def y_sampler_step(self, y_t, model_pred, t, next_t, precomputed_y_0=None, shift_logits=None):
        # y_0_pred = precomputed_y_0 if precomputed_y_0 is not None else joint_denoiser_output.y_pred
        if self.prediction_type == 'eps':
            eps = model_pred if (precomputed_y_0 is None) else self.ddpm.y_0_to_eps(y_t, precomputed_y_0, t)
            if self.ddim:
                y_t = self.ddpm.sample_bridge_eps_pred_ddim(y_t, eps, t, next_t)
            else:
                y_t = self.ddpm.sample_bridge_eps_pred(y_t, eps, t, next_t)
            # y_0 = self.ddpm.eps_to_y_0(y_t, eps, t)
            # y_t = self.ddpm.sample_bridge_y_0_pred(y_t, y_0, t, next_t)
        elif self.prediction_type == 'y_0':
            y_0 = model_pred if (precomputed_y_0 is None) else precomputed_y_0
            if self.ddim:
                y_t = self.ddpm.sample_bridge_y_0_pred_ddim(y_t, y_0, t, next_t)
            else:
                y_t = self.ddpm.sample_bridge_y_0_pred(y_t, y_0, t, next_t)
        else:
            raise NotImplementedError(f'Unknown prediction type: {self.prediction_type}')
        return y_t

    def y_sampler_last_step(self, y_t, model_pred, t, precomputed_y_0=None, shift_logits=None):
        # y_0_pred = precomputed_y_0 if precomputed_y_0 is not None else joint_denoiser_output.y_pred
        if self.prediction_type == 'eps':
            eps = model_pred if (precomputed_y_0 is None) else self.ddpm.y_0_to_eps(y_t, precomputed_y_0, t)
            y_t = self.ddpm.sample_last_step_eps_pred(y_t, eps, t)
            # y_0 = self.ddpm.eps_to_y_0(y_t, eps, t)
            # y_t = self.ddpm.sample_last_step_y_0_pred(y_t, y_0, t)
        elif self.prediction_type == 'y_0':
            y_0 = model_pred if (precomputed_y_0 is None) else precomputed_y_0
            y_t = self.ddpm.sample_last_step_y_0_pred(y_t, y_0, t)
        else:
            raise NotImplementedError(f'Unknown prediction type: {self.prediction_type}')
        return y_t

    def sample_bridge_x(self, x, y, x_logits, y_pred, t, s, shift_logits=None):
        raise NotImplementedError('sample_bridge_x method is not implemented.')

    def sample_bridge_y(self, x, y, x_logits, y_pred, t, s, shift_logits=None):
        raise NotImplementedError('sample_bridge_y method is not implemented.')

    def sample_last_step_x(self, x, y, x_logits, y_pred):
        raise NotImplementedError('sample_last_step_x method is not implemented.')

    def sample_last_step_y(self, x, y, x_logits, y_pred):
        raise NotImplementedError('sample_last_step_y method is not implemented.')

    # def get_latent_loss_0(self, y_0_mean, y_0_std, y_0, y_0_pred_mean_time_0, y_0_pred_std_time_0):
    #     assert False, 'This function is deprecated and should not be used.'
    #     loss_dict_y_0 = {}
    #     # compute L_{latent} loss, this time with learnable or fixed variance
    #     if self.latent_strategy['variance'] == 'fixed':
    #         rescaled_y_0_pred_mean = y_0_pred_mean_time_0 / self.latent_strategy['fixed_sigma_latent']
    #         rescaled_y_0 = y_0 / self.latent_strategy['fixed_sigma_latent']
    #         elbo_y_0 = 0.5 * F.mse_loss(rescaled_y_0_pred_mean, rescaled_y_0, reduction='none') - 0.5
    #         elbo_y_0 = elbo_y_0.sum(dim=-1).mean()
    #     elif self.latent_strategy['variance'] == 'learnable':
    #         rescaled_y_0_pred_mean = y_0_pred_mean_time_0 / y_0_pred_std_time_0
    #         rescaled_y_0 = y_0 / y_0_pred_std_time_0
    #         elbo_y_0 = 0.5 * F.mse_loss(rescaled_y_0_pred_mean, rescaled_y_0, reduction='none') - 0.5
    #         elbo_y_0 += torch.log(y_0_pred_std_time_0 / y_0_std)
    #         elbo_y_0 = elbo_y_0.sum(dim=-1).mean()
    #     else:
    #         raise NotImplementedError(f'Unknown latent variance strategy: {self.latent_strategy["variance"]}')
    #     loss_dict_y_0['elbo_y_0'] = elbo_y_0
    #     loss_dict_y_0['loss_y_0'] = loss_dict_y_0['elbo_y_0']
    #     return loss_dict_y_0


class JointProcess(LatentProcess):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def model_forward(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: ContinuousDiTModel,
        x,
        y,
        t_x,
        t_y,
        y_0,
        attention_mask: torch.Tensor | None = None,
    ):
        t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t_x), t_x=t_x)
        t_y = self.get_timesteps_y(t_y)
        joint_denoiser_output = joint_denoiser(x, y, t_x, t_y, attention_mask=attention_mask)
        return joint_denoiser_output.logits, joint_denoiser_output.y_pred, joint_denoiser_output.y_std

    def sample_backward(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: ContinuousDiTModel,
        x_T,
        y_T,
        ts,
        ts_latent,
        x_sampler_step,
        x_last_sampler_step,
        precomputed_y_0=None,
        noise_removal=True,
        sample_latents_only=False,
        temperature: float | None = None,
    ):
        t0 = ts[-1]
        next_ts = ts[1:]
        x_t = x_T
        noise_t = x_T.clone()
        y_t = y_T
        num_samples = x_t.shape[0]
        for _, (t, next_t) in tqdm(enumerate(zip(ts[:-1], next_ts, strict=False)), total=len(ts)-1, desc='Sampling with Joint process', disable=not is_rank_zero()):
            t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t), t_x=t)
            t_y = self.get_timesteps_y(t)
            joint_denoiser_output = self._joint_denoiser_forward_with_temperature(
                joint_denoiser=joint_denoiser,
                x_t=x_t,
                y_t=y_t,
                timesteps_x=t_x.expand(num_samples) if t_x is not None else None,
                timesteps_y=t_y.expand(num_samples),
                temperature=temperature,
            )
            y_t = self.y_sampler_step(y_t, joint_denoiser_output.y_pred, t, next_t, precomputed_y_0)
            alphas_x = self.x_noise_sampler(t).alpha
            next_alphas_x = self.x_noise_sampler(next_t).alpha
            x_t, noise_t = x_sampler_step(joint_denoiser_output, alphas_x, next_alphas_x, x_t, noise_t)
        if noise_removal:
            t = t0
            t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t), t_x=t)
            t_y = self.get_timesteps_y(t)
            joint_denoiser_output = self._joint_denoiser_forward_with_temperature(
                joint_denoiser=joint_denoiser,
                x_t=x_t,
                y_t=y_t,
                timesteps_x=t_x.expand(num_samples) if t_x is not None else None,
                timesteps_y=t_y.expand(num_samples),
                temperature=temperature,
            )
            y_t = self.y_sampler_last_step(y_t, joint_denoiser_output.y_pred, t, precomputed_y_0)
            x_t, noise_t = x_last_sampler_step(joint_denoiser_output, x_t, noise_t)
        return x_t, y_t


class SEQProcess(LatentProcess):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def model_forward(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: ContinuousDiTModel,
        x,
        y,
        t_x,
        t_y,
        y_0,
        attention_mask: torch.Tensor | None = None,
    ):
        t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t_x), t_x=t_x)
        t_y_zero = self.get_timesteps_y(torch.zeros_like(t_y))
        joint_denoiser_output = joint_denoiser(x, y_0, t_x, t_y_zero, attention_mask=attention_mask)
        t_y = self.get_timesteps_y(t_y)
        latent_denoiser_output = latent_denoiser(y, t_y)
        return joint_denoiser_output.logits, latent_denoiser_output.y_pred, latent_denoiser_output.y_std

    def sample_backward(
        self,
        joint_denoiser: JointDiscreteDiTModel,
        latent_denoiser: ContinuousDiTModel,
        x_T,
        y_T,
        ts,
        ts_latent,
        x_sampler_step,
        x_last_sampler_step,
        precomputed_y_0=None,
        noise_removal=True,
        sample_latents_only=False,
        temperature: float | None = None,
    ):
        t0 = ts[-1]
        next_ts = ts[1:]
        t0_latent = ts_latent[-1]
        next_ts_latent = ts_latent[1:]
        x_t = x_T
        noise_t = x_T.clone()
        y_t = y_T
        num_samples = x_t.shape[0]
        if precomputed_y_0 is not None:
            y_t = precomputed_y_0
        else:
            for _, (t, next_t) in tqdm(
                enumerate(zip(ts_latent[:-1], next_ts_latent, strict=False)),
                total=len(ts_latent) - 1,
                desc='Sampling y_t',
                disable=not is_rank_zero(),
            ):
                t_y = self.get_timesteps_y(t)
                latent_output: Output = latent_denoiser(
                    y_t,
                    timesteps_y=t_y.expand(num_samples),
                )
                y_t = self.y_sampler_step(y_t, latent_output.y_pred, t, next_t, precomputed_y_0)
            if noise_removal:
                t = t0_latent
                t_y = self.get_timesteps_y(t)
                latent_output: Output = latent_denoiser(
                    y_t,
                    timesteps_y=t_y.expand(num_samples),
                )
                y_t = self.y_sampler_last_step(y_t, latent_output.y_pred, t, precomputed_y_0)
        if sample_latents_only:
            return x_t, y_t
        for _, (t, next_t) in tqdm(
            enumerate(zip(ts[:-1], next_ts, strict=False)),
            total=len(ts) - 1,
            desc='Sampling with SEQ process',
            disable=not is_rank_zero(),
        ):
            t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t), t_x=t)
            t_y = self.get_timesteps_y(torch.zeros_like(t))
            joint_denoiser_output = self._joint_denoiser_forward_with_temperature(
                joint_denoiser=joint_denoiser,
                x_t=x_t,
                y_t=y_t,
                timesteps_x=t_x.expand(num_samples) if t_x is not None else None,
                timesteps_y=t_y.expand(num_samples),
                temperature=temperature,
            )
            alphas_x = self.x_noise_sampler(t).alpha
            next_alphas_x = self.x_noise_sampler(next_t).alpha
            x_t, noise_t = x_sampler_step(joint_denoiser_output, alphas_x, next_alphas_x, x_t, noise_t)
        if noise_removal:
            t_x = self.get_timesteps_x(model_config=joint_denoiser.config, alphas_x=self.x_noise_sampler(t0), t_x=t0)
            t_y = self.get_timesteps_y(torch.zeros_like(t0))
            joint_denoiser_output = self._joint_denoiser_forward_with_temperature(
                joint_denoiser=joint_denoiser,
                x_t=x_t,
                y_t=y_t,
                timesteps_x=t_x.expand(num_samples) if t_x is not None else None,
                timesteps_y=t_y.expand(num_samples),
                temperature=temperature,
            )
            x_t, noise_t = x_last_sampler_step(joint_denoiser_output, x_t, noise_t)
        return x_t, y_t
