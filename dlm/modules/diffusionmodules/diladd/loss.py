from typing import Any
import time

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch import nn

from dlm.transformers.dit import JointDiscreteDiTModel

from ..coladd.infonce_loss import InfoNCELoss
from ..masked_loss import MaskedDiffusionLoss
from ..nonfinite_utils import summarize_nonfinite_tensor
from ..noise_sampling import NoiseOutput, NoiseSampling
from ....utils import print_rank_zero


class DiscreteLatentDDMLoss(nn.Module):
    """Loss function for Discrete Latent Discrete Diffusion Model (LatentDDM).

    This loss combines:
    1. The standard masked diffusion loss for the token sequence `x`.
    2. The standard masked diffusion loss for the token sequence `y`.
    """

    def __init__(
        self,
        # Config for the core model to use
        ratio_x_to_y_stage_1: float | None = None,
        ratio_x_to_y_stage_2: float | None = None,
        data_loss_weight_stage_1: float = 1.0,
        data_loss_weight_stage_2: float = 1.0,
        latent_loss_weight_stage_1: float = 1.0,
        latent_loss_weight_stage_2: float = 1.0,
        latent_loss_weight_schedule: str | None = None,
        latent_loss_weight_schedule_steps: int | None = None,
        training_stage: int | None = None,
        encoder_strategy: str = 'gumbel',  # vqvae, gumbel or reinforce
        # New options for using target labels
        use_precomputed_latent: bool = False,
        use_encoder_supervision: bool = False,
        encoder_supervision_weight: float = 1.0,
        p_cfg: float = 0.0,  # p_tau
        use_p_r: bool = False,  # p_r
        kl_regularization_weight: float = 0.0,
        conditional_independence_loss: bool = False,
        conditional_independence_kl_direction: str = 'forward',  # 'forward' for D_KL(P || Q), 'backward' for D_KL(Q || P)
        conditional_independence_detach_p: bool = True,  # whether to detach P in the KL loss
        independent_timesteps: bool = False,
        use_data_loss_elbo: bool = True,
        use_latent_loss_elbo: bool = False,
        # InfoNCE two-view invariance parameters
        use_infonce_loss: bool = False,
        infonce_weight: float = 1.0,
        infonce_temperature: float = 0.1,
        p_mask_input_enc: float = 0.0,  # probability of masking tokens for two-view corruption
        measure_wall_clock_time: bool = False,
        x_kwargs: dict[str, Any] = {},
        y_kwargs: dict[str, Any] = {},
    ):
        super().__init__()
        self.x_loss = MaskedDiffusionLoss(**x_kwargs)
        self.y_loss = MaskedDiffusionLoss(**y_kwargs)

        assert self.x_loss.time_sampler.eps > 0, 'eps must be larger than zero, will be used to determine step size'
        assert self.y_loss.time_sampler.eps > 0, 'eps must be larger than zero, will be used to determine step size'

        self.x_noise_sampler: NoiseSampling = self.x_loss.noise_sampler
        self.y_noise_sampler: NoiseSampling = self.y_loss.noise_sampler

        self.ratio_x_to_y_stage_1 = ratio_x_to_y_stage_1
        self.ratio_x_to_y_stage_2 = ratio_x_to_y_stage_2
        self.data_loss_weight_stage_1 = data_loss_weight_stage_1
        self.data_loss_weight_stage_2 = data_loss_weight_stage_2
        self.latent_loss_weight_stage_1 = latent_loss_weight_stage_1
        self.latent_loss_weight_stage_2 = latent_loss_weight_stage_2
        self.latent_loss_weight_schedule = latent_loss_weight_schedule
        self.latent_loss_weight_schedule_steps = latent_loss_weight_schedule_steps

        self.total_x_y_steps_stage_1 = sum(self.ratio_x_to_y_stage_1) if self.ratio_x_to_y_stage_1 is not None else 0
        self.total_x_y_steps_stage_2 = sum(self.ratio_x_to_y_stage_2) if self.ratio_x_to_y_stage_2 is not None else 0
        self.total_x_y_loss_steps = self.total_x_y_steps_stage_1 + self.total_x_y_steps_stage_2
        self.training_stage = training_stage

        assert self.training_stage in [1, 2], f'got training_stage = {self.training_stage}'

        self.encoder_strategy = encoder_strategy
        
        self.register_buffer('latent_loss_weight_training_steps', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_iter', torch.tensor(0, dtype=torch.long))

        self.use_precomputed_latent = use_precomputed_latent
        self.use_encoder_supervision = use_encoder_supervision
        self.encoder_supervision_weight = encoder_supervision_weight
        self.p_cfg = p_cfg
        self.use_p_r = use_p_r
        self.kl_regularization_weight = kl_regularization_weight
        self.conditional_independence_loss = conditional_independence_loss
        self.conditional_independence_kl_direction = conditional_independence_kl_direction
        self.conditional_independence_detach_p = conditional_independence_detach_p
        self.independent_timesteps = independent_timesteps
        self.use_latent_loss_elbo = use_latent_loss_elbo
        self.use_data_loss_elbo = use_data_loss_elbo

        # InfoNCE two-view invariance
        self.use_infonce_loss = use_infonce_loss
        self.infonce_weight = infonce_weight
        self.p_mask_input_enc = p_mask_input_enc
        self.measure_wall_clock_time = measure_wall_clock_time
        if self.use_infonce_loss:
            self.infonce_loss_fn = InfoNCELoss(
                temperature=infonce_temperature,
                normalize=True,
            )

    @staticmethod
    def _sync_device_for_timing(device: torch.device) -> None:
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        elif device.type == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def manage_gumbel_temperature(self):
        # manage gumbel temperature if encoder_strategy is gumbel
        # decay temperature logaritmically from .1 to 1e-4 over 200k steps
        min_temp = 1e-5
        max_temp = 1.0
        decay_steps = 200000
        if self.total_iter.item() > decay_steps:
            return torch.tensor(min_temp)
        temp = torch.exp(
            torch.log(torch.tensor(max_temp))
            - (torch.log(torch.tensor(max_temp)) - torch.log(torch.tensor(min_temp))) * (self.total_iter / decay_steps),
        )
        return temp

    def _log_nonfinite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            return
        if torch.isfinite(tensor).all():
            return
        stats = summarize_nonfinite_tensor(tensor)
        print_rank_zero(
            f'[DiscreteLatentDDMLoss] Non-finite {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} '
            f'device={tensor.device} nan={stats["num_nan"]} inf={stats["num_inf"]} '
            f'finite_min={stats["finite_min"]} finite_max={stats["finite_max"]} '
            f'finite_mean={stats["finite_mean"]}',
        )

    def _log_prediction_context(
        self,
        *,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        attention_mask: torch.Tensor,
        x_0_pred_logits: torch.Tensor,
        y_0_pred_logits: torch.Tensor,
    ) -> None:
        masked_fraction_x = (x_t == self.x_loss.mask_token_id).float().mean().item()
        masked_fraction_y = (y_t == self.y_loss.mask_token_id).float().mean().item()
        attention_keep_fraction = attention_mask.float().mean().item()
        print_rank_zero(
            '[DiscreteLatentDDMLoss] Prediction context: '
            f't_x[min={t_x.min().item():.6f}, max={t_x.max().item():.6f}, mean={t_x.mean().item():.6f}] '
            f't_y[min={t_y.min().item():.6f}, max={t_y.max().item():.6f}, mean={t_y.mean().item():.6f}] '
            f'x_masked_frac={masked_fraction_x:.6f} y_masked_frac={masked_fraction_y:.6f} '
            f'attention_keep_frac={attention_keep_fraction:.6f} '
            f'x_vocab={x_0_pred_logits.shape[-1]} y_vocab={y_0_pred_logits.shape[-1]}',
        )
        self._log_nonfinite('x_t', x_t)
        self._log_nonfinite('y_t', y_t)
        self._log_nonfinite('x_0_pred_logits', x_0_pred_logits)
        self._log_nonfinite('y_0_pred_logits', y_0_pred_logits)

    def manage_training_stage(self, batch_size, batch_idx, device):
        x_loss_enabled = torch.ones(batch_size, dtype=torch.int32, device=device, requires_grad=False)
        y_loss_enabled = torch.ones(batch_size, dtype=torch.int32, device=device, requires_grad=False)
        # stage 1
        if self.training_stage == 1:
            # if latent_denoiser is not None:
            # latent denoiser is frozen
            # set_grad([latent_denoiser], requires_grad=False)
            y_loss_enabled = 0 * y_loss_enabled
        # stage 2
        elif self.total_x_y_steps_stage_2 > 0:
            batch_idx_stage_2 = batch_idx % self.total_x_y_steps_stage_2
            x_loss_enabled = int(batch_idx_stage_2 < self.ratio_x_to_y_stage_2[0]) * x_loss_enabled
            y_loss_enabled = 1 - x_loss_enabled
        return x_loss_enabled, y_loss_enabled

    def model_forward(
        self,
        model: JointDiscreteDiTModel,
        x_t: torch.Tensor,
        y_t_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        alphas_x: NoiseOutput | None,
        alphas_y: NoiseOutput | None,
        t_x: torch.Tensor | None,
        t_y: torch.Tensor | None,
        **model_kwargs: dict[str, Any],
    ):
        model_kwargs['attention_mask'] = attention_mask
        x_time_type = getattr(model.config, 'x_time_type', 'none')
        y_time_type = getattr(model.config, 'y_time_type', 'none')
        if x_time_type == 'noise':
            timesteps_x = alphas_x.alpha
        elif x_time_type == 'time':
            timesteps_x = t_x
        elif x_time_type == 'none':
            timesteps_x = None
        else:
            raise NotImplementedError(f'Unknown time_type: {x_time_type}')
        if y_time_type == 'noise':
            timesteps_y = alphas_y.alpha
        elif y_time_type == 'time':
            timesteps_y = t_y
        elif y_time_type == 'none':
            timesteps_y = None
        else:
            raise NotImplementedError(f'Unknown time_type: {y_time_type}')
        model_kwargs['timesteps_x'] = timesteps_x
        model_kwargs['timesteps_y'] = timesteps_y
        if not self.measure_wall_clock_time:
            return model(x_t, y_t_vec, **model_kwargs)

        self._sync_device_for_timing(x_t.device)
        start_time = time.perf_counter()
        output = model(x_t, y_t_vec, **model_kwargs)
        self._sync_device_for_timing(x_t.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print_rank_zero(f'[DiscreteLatentDDMLoss] model_forward wall clock: {elapsed_ms:.3f} ms')
        return output

    def get_data_loss_t(self, x_0_pred_logits, x_0, x_t, t_x, attention_mask, labels, example_mask=None):
        x_loss_weight = self.x_noise_sampler(t_x).loss_weight
        loss_dict_x = self.x_loss.get_loss(
            logits=x_0_pred_logits,
            x_0=x_0,
            x_t=x_t,
            attention_mask=attention_mask,
            labels=labels,
            loss_weight=x_loss_weight,
            use_data_loss_elbo=self.use_data_loss_elbo,
            example_mask=example_mask,
        )
        loss_dict_x_t = {f'{k}_t': v for k, v in loss_dict_x.items() if '_x' in k}
        return loss_dict_x_t, loss_dict_x

    def get_latent_loss_t(self, y_0_pred_logits, y_0, y_t, t_y, attention_mask, labels):
        y_loss_weight = self.y_noise_sampler(t_y).loss_weight
        # no gradient operation on y_0 and y_t; will only select relevant positions in y_0_pred_logits
        loss_dict_y = self.y_loss.get_loss(
            logits=y_0_pred_logits,
            x_0=y_0,
            x_t=y_t,
            attention_mask=attention_mask,
            labels=labels,
            loss_weight=y_loss_weight,
            use_data_loss_elbo=self.use_data_loss_elbo,
        )
        loss_dict_y_t = {
            'loss_y_t': loss_dict_y['loss'],
            'loss_elbo_y_t': loss_dict_y['elbo'],
        }
        return loss_dict_y_t, loss_dict_y

    def compute_latent_loss_weight(self):
        latent_loss_weight = self.latent_loss_weight_stage_1 if self.training_stage == 1 else self.latent_loss_weight_stage_2
        if self.latent_loss_weight_schedule is None:
            return latent_loss_weight
        if self.latent_loss_weight_schedule == 'warmup':
            assert self.latent_loss_weight_schedule_steps is not None, (
                'latent_loss_weight_schedule_steps must be set for warmup'
            )
            current_step = min(self.latent_loss_weight_training_steps.item(), self.latent_loss_weight_schedule_steps)
            self.latent_loss_weight_training_steps += 1
            latent_loss_weight = latent_loss_weight * current_step / self.latent_loss_weight_schedule_steps
            return latent_loss_weight
        if self.latent_loss_weight_schedule == 'delayed_warmup':
            assert self.latent_loss_weight_schedule_steps is not None, (
                'latent_loss_weight_schedule_steps must be set for delayed_warmup'
            )
            delay_steps = int(0.2 * self.latent_loss_weight_schedule_steps)
            warmup_steps = self.latent_loss_weight_schedule_steps - delay_steps
            current_step = min(self.latent_loss_weight_training_steps.item(), self.latent_loss_weight_schedule_steps)
            self.latent_loss_weight_training_steps += 1
            if current_step < delay_steps:
                latent_loss_weight = 0.0
            else:
                latent_loss_weight = latent_loss_weight * (current_step - delay_steps) / warmup_steps
            return latent_loss_weight
        raise ValueError(f'Unknown latent_loss_weight_schedule: {self.latent_loss_weight_schedule}')

    def compute_data_loss_weight(self):
        data_loss_weight = self.data_loss_weight_stage_1 if self.training_stage == 1 else self.data_loss_weight_stage_2
        return data_loss_weight

    def forward(
        self,
        lightning_module: L.LightningModule,
        joint_denoiser: JointDiscreteDiTModel,
        encoder: torch.nn.Module,
        latent_denoiser: torch.nn.Module | None,
        vector_quantizer: torch.nn.Module | None,
        batch: dict[str, Any],
        batch_idx: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # 1. Get inputs from batch
        x_0 = batch['input_ids']
        labels = batch.get('labels', x_0.detach().clone())  # old labels
        attention_mask = batch.get('attention_mask', torch.ones_like(x_0))
        maskable_mask = labels != -100

        # manage stages
        x_loss_enabled, y_loss_enabled = self.manage_training_stage(x_0.shape[0], batch_idx, x_0.device)

        # 3. Sample timesteps and noise
        t_x = self.x_loss.time_sampler(x_0.size(0)).to(x_0.device)
        t_y = self.y_loss.time_sampler(x_0.size(0)).to(x_0.device) if self.independent_timesteps else t_x
        
        x_0_view1 = x_0
        if (self.p_mask_input_enc > 0.) and self.training:
            mask_view1 = torch.rand(x_0.shape, device=x_0.device) < self.p_mask_input_enc
            x_0_view1 = x_0.clone()
            x_0_view1[mask_view1 & maskable_mask] = self.x_loss.mask_token_id
        
        # encode to y_0 logits and forbid probability mass on the latent mask token
        if self.encoder_strategy == 'gumbel':
            y_0_logits = encoder(x_0_view1, attention_mask=attention_mask).y_pred
            z_e_view1 = y_0_logits.clone()  # for InfoNCE loss later
            latent_mask_id = self.y_loss.mask_token_id
            y_0_logits[..., latent_mask_id] = self.y_loss.masked_process.neg_infty
            y_0_logits = y_0_logits - torch.logsumexp(y_0_logits, dim=-1, keepdim=True)
            tau = self.manage_gumbel_temperature()
            y_0_onehot = F.gumbel_softmax(
                y_0_logits,
                tau=tau,
                hard=True,
                dim=-1,
            )  # hard=True -> straight-through; mask token has (near) zero probability
            # Use integer tokens for the discrete diffusion process
            y_0 = y_0_onehot.argmax(dim=-1).long()  # ATTENTION: breaks gradient graph
        elif self.encoder_strategy == 'vqvae':
            assert vector_quantizer is not None, 'vector_quantizer must be provided for vqvae encoder_strategy'
            z_e = encoder(x_0_view1, attention_mask=attention_mask).y_pred  # (B, S, D) or (B, S*D) depending on encoder architecture
            z_e = z_e.view(z_e.size(0), -1, vector_quantizer.vqvae_latent_dim)  # ensures (B, S, D)
            z_e_view1 = z_e.clone()  # for InfoNCE loss later
            z_q_st, vq_loss, y_0, _ = vector_quantizer(z_e, mask_positions=None, return_one_hot=False)
        else:
            raise NotImplementedError(f'Encoder strategy {self.encoder_strategy} not implemented yet')

        # For y we allow all positions to be maskable (independent of x mask)
        maskable_mask_y = torch.ones_like(y_0, dtype=torch.bool)
        attention_mask_y = torch.ones_like(y_0).long()

        # sample x_t, y_t
        alphas_x = self.x_noise_sampler(t_x)
        x_t = self.x_loss.masked_process.sample_forward(x_0, alphas_x, maskable_mask, model_config=joint_denoiser.config)
        alphas_y = self.y_noise_sampler(t_y)
        y_t = self.y_loss.masked_process.sample_forward(y_0, alphas_y, maskable_mask_y, model_config=joint_denoiser.config)
        # y_t (B, S)

        # 7. Calculate losses, and some more logging
        loss_dict_x = {'loss_x': torch.tensor(0.0, device=x_0.device), 'elbo_x': torch.tensor(0.0, device=x_0.device)}
        loss_dict_y = {'loss_y': torch.tensor(0.0, device=x_0.device), 'elbo_y': torch.tensor(0.0, device=x_0.device)}
        # create union of loss_dict_x and loss_dict_y
        loss_dict = {**loss_dict_x, **loss_dict_y}

        # whether to drop labels for classifier-free guidance
        if self.p_cfg > 0 and self.training:
            drop_mask = (torch.rand(x_0.shape[0], device=x_0.device) < self.p_cfg) & (x_loss_enabled == 1)
            if drop_mask.any():
                # print_rank_zero(f'Dropping y labels for {drop_mask.sum().item()} samples over {x_0.shape[0]} in the batch')
                y_t[drop_mask] = torch.full_like(y_t[0], self.y_loss.masked_process.mask_token_id)
                y_0[drop_mask] = torch.full_like(y_0[0], self.y_loss.masked_process.mask_token_id)
                y_loss_enabled[drop_mask] = 0

        model_input = None
        if self.encoder_strategy == 'gumbel':
            # now set y_t_onehot thanks to y_t, as y_0 and y_t are severed from the computational graph
            # not really necessary to clone y_0_onehot since gradient flow happens through y_t_onehot only
            y_t_onehot = y_0_onehot.clone()  # (B, S, D)
            y_t_onehot[y_t == self.y_loss.masked_process.mask_token_id] = torch.nn.functional.one_hot(
                y_t[y_t == self.y_loss.masked_process.mask_token_id],
                num_classes=y_0_onehot.shape[-1],
            ).float()
            model_input = y_t_onehot
        elif self.encoder_strategy == 'vqvae':
            # build z_t: replace masked positions by z_mask
            # vq.z_mask is nn.Parameter of shape (D,)
            z_t = z_q_st.clone()  # (B, S, D)
            mask_token_id = self.y_loss.masked_process.mask_token_id
            mask = (y_t == mask_token_id).unsqueeze(-1)  # (B, S, 1), bool
            z_t = torch.where(
                mask,
                vector_quantizer.z_mask.view(1, 1, -1).expand_as(z_t),  # broadcast (D,) -> (B, S, D)
                z_t,
            )  # (B, S, D)
            model_input = z_t
        
        # predict from t > 0
        model_output = self.model_forward(
            model=joint_denoiser,
            x_t=x_t,
            y_t_vec=model_input,
            attention_mask=attention_mask,
            alphas_x=alphas_x,
            alphas_y=alphas_y,
            t_x=t_x,
            t_y=t_y,
        )

        x_0_pred_logits = model_output.logits
        y_0_pred_logits = model_output.y_pred

        if (not torch.isfinite(x_0_pred_logits).all()) or (not torch.isfinite(y_0_pred_logits).all()):
            self._log_prediction_context(
                x_t=x_t,
                y_t=y_t,
                t_x=t_x,
                t_y=t_y,
                attention_mask=attention_mask,
                x_0_pred_logits=x_0_pred_logits,
                y_0_pred_logits=y_0_pred_logits,
            )

        data_loss_weight = self.compute_data_loss_weight()
        latent_loss_weight = self.compute_latent_loss_weight()

        # Loss for x (token sequence)
        if (x_loss_enabled > 0).any():
            # retrieve loss_dict_x in order to know how to format the keys of loss_dict_x_0
            x_mask = x_loss_enabled > 0
            loss_dict_x_t, loss_dict_x = self.get_data_loss_t(
                x_0_pred_logits,
                x_0,
                x_t,
                t_x,
                attention_mask,
                labels,
                example_mask=x_mask,
            )
            loss_dict.update(loss_dict_x_t)
            loss_dict.update(loss_dict_x)

        # Loss for y (latent vector)
        if (y_loss_enabled > 0).any() and latent_loss_weight > 0.0:
            y_mask = y_loss_enabled > 0
            loss_dict_y_t, loss_dict_y = self.get_latent_loss_t(
                y_0_pred_logits.float()[y_mask],
                y_0[y_mask],
                y_t[y_mask],
                t_y[y_mask],
                attention_mask_y[y_mask],
                labels=None, # no labels for latent
            )
            loss_dict.update(loss_dict_y_t)
            # aggregate latent loss used in total loss
            loss_dict['loss_y'] = loss_dict_y['loss']
            loss_dict['elbo_y'] = loss_dict_y['elbo']

        # Conditional independence loss
        if self.conditional_independence_loss and (x_loss_enabled > 0).any():
            # Create fully masked x_t for conditional independence
            # we could also sample t_masked > t_x instead of t_masked = 1
            # which would be closer to current self-distillation, but well.
            t_masked = torch.ones_like(t_x) * (1 - self.x_loss.time_sampler.eps)
            alphas_x_masked = self.x_noise_sampler(t_masked)
            x_t_masked = self.x_loss.masked_process.sample_forward(x_0, alphas_x_masked, maskable_mask, model_config=joint_denoiser.config)
            # x_t_masked = torch.full_like(x_t, self.x_loss.mask_token_id)

            # Prepare model input based on encoder strategy
            if self.encoder_strategy == 'gumbel' or self.encoder_strategy == 'vqvae':
                model_input_cond_ind = model_input
            else:
                raise NotImplementedError(f'Encoder strategy {self.encoder_strategy} not implemented yet')

            # Get predictions with fully masked x
            model_output_cond_ind = self.model_forward(
                model=joint_denoiser,
                x_t=x_t_masked,
                y_t_vec=model_input_cond_ind,
                attention_mask=attention_mask,
                alphas_x=alphas_x_masked,
                alphas_y=alphas_y,
                t_x=t_masked,
                t_y=t_y,
            )

            x_0_pred_logits_cond_ind = model_output_cond_ind.logits

            # Select only samples where x_loss is enabled
            x_mask = x_loss_enabled > 0

            # Compute the KL between x_0_pred_logits_cond_ind and x_0_pred_logits
            # --- numerically sensitive region in fp32 ---
            with torch.autocast(device_type=x_t.device.type, enabled=False):
                # Only enforce KL on tokens originally masked in x_t (for these samples)
                mask_indices = x_t[x_mask] == self.x_loss.mask_token_id  # (B', S) bool

                p_logprobs: torch.Tensor = self.x_loss.masked_process.subs_parameterization(
                    x=x_t[x_mask],
                    x0_logits=x_0_pred_logits[x_mask].float(),
                    mask_indices=mask_indices,
                    return_probs=False,
                )  # (B', S, V) log-probs

                x_t_masked_sel = x_t_masked[x_mask]
                full_mask_indices = x_t_masked_sel == self.x_loss.mask_token_id  # (B', S) bool (likely all True)
                q_logprobs: torch.Tensor = self.x_loss.masked_process.subs_parameterization(
                    x=x_t_masked_sel,
                    x0_logits=x_0_pred_logits_cond_ind[x_mask].float(),
                    mask_indices=full_mask_indices,
                    return_probs=False,
                )  # (B', S, V) log-probs

                # Optional detach of P (teacher)
                p_logprobs_for_kl = p_logprobs.detach() if self.conditional_independence_detach_p else p_logprobs
                q_logprobs_for_kl = q_logprobs

                # Do not optimize on delta Diracs + avoid padding positions
                pos_mask = mask_indices & attention_mask[x_mask].bool()  # (B', S) bool

                # Per-position KL over vocab, then mean over masked positions
                if self.conditional_independence_kl_direction == 'forward':
                    # KL(P || Q): target=P, input=Q (gradient flows into input=Q)
                    kl_per_vocab = F.kl_div(
                        input=q_logprobs_for_kl,
                        target=p_logprobs_for_kl,
                        reduction='none',
                        log_target=True,
                    )  # (B', S, V)
                elif self.conditional_independence_kl_direction == 'backward':
                    # KL(Q || P): target=Q, input=P (gradient flows into input=P)
                    # If you keep detach_p=True, this will (correctly) kill gradients from this KL.
                    kl_per_vocab = F.kl_div(
                        input=p_logprobs_for_kl,
                        target=q_logprobs_for_kl,
                        reduction='none',
                        log_target=True,
                    )  # (B', S, V)
                else:
                    raise ValueError(
                        f'Unknown conditional_independence_kl_direction: {self.conditional_independence_kl_direction}',
                    )

                kl_per_pos = kl_per_vocab.sum(dim=-1)  # (B', S)
                denom = pos_mask.sum().clamp_min(1)
                kl_loss = (kl_per_pos * pos_mask).sum() / denom  # scalar: nats per masked token
                loss_dict['loss_x_cond_ind'] = kl_loss

        # 8. Compute the ELBO
        loss_dict['elbo'] = loss_dict['elbo_x'] + loss_dict['elbo_y']

        # 9. Compute the total loss
        loss_y = loss_dict['elbo_y'] if self.use_latent_loss_elbo else loss_dict['loss_y']
        loss_dict['loss'] = data_loss_weight * loss_dict['loss_x'] + latent_loss_weight * loss_y

        if 'loss_x_cond_ind' in loss_dict:
            loss_dict['loss'] += loss_dict['loss_x_cond_ind']

        # 10. Add the vq loss if using VQ-VAE encoder
        if self.encoder_strategy == 'vqvae':
            loss_dict['vq_loss'] = vq_loss
            loss_dict['loss'] += vq_loss

        # Manage possible nans
        loss_dict['loss'] = torch.nan_to_num(loss_dict['loss'])

        # 11. Add InfoNCE two-view invariance loss
        if self.use_infonce_loss and self.training and self.p_mask_input_enc > 0:
            # Create two corrupted views by masking encoder input
            # View 1: already created above as x_0_view1
            # View 2: independent random masking
            mask_view2 = torch.rand(x_0.shape, device=x_0.device) < self.p_mask_input_enc
            x_0_view2 = x_0.clone()
            x_0_view2[mask_view2 & maskable_mask] = self.x_loss.mask_token_id

            # Encode both views to get latent representations
            # For discrete LDDM, we get logits/embeddings before quantization
            with torch.no_grad() if False else torch.enable_grad():
                if self.encoder_strategy == 'gumbel':
                    # Get logits (continuous) before gumbel-softmax
                    y_logits_view2 = encoder(x_0_view2, attention_mask=attention_mask).y_pred  # (B, S', V)
                    # Use the logits directly as continuous embeddings for InfoNCE
                    z_e_view2 = y_logits_view2  # (B, S', V)

                elif self.encoder_strategy == 'vqvae':
                    # Get pre-quantization embeddings
                    z_e_view2 = encoder(x_0_view2, attention_mask=attention_mask).y_pred  # (B, S', D)
                    # Reshape if needed
                    if vector_quantizer is not None:
                        z_e_view2 = z_e_view2.view(z_e_view2.size(0), -1, vector_quantizer.vqvae_latent_dim)
                else:
                    raise NotImplementedError(f'InfoNCE not implemented for encoder_strategy {self.encoder_strategy}')

            # Compute InfoNCE loss using attention mask for pooling
            infonce_result = self.infonce_loss_fn(
                embeddings_view1=z_e_view1,
                embeddings_view2=z_e_view2,
                attention_mask_view1=None, # no attention mask in latent space
                attention_mask_view2=None, # no attention mask in latent space
            )
            loss_dict['loss_infonce'] = infonce_result['loss']
            loss_dict['loss'] += self.infonce_weight * loss_dict['loss_infonce']

        self.total_iter += 1

        return loss_dict
