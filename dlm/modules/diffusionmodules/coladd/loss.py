from typing import Any
import time

import lightning.pytorch as L
import torch
import torch.nn.functional as F

from dlm.transformers.dit import ContinuousDiTModel, JointDiscreteDiTModel

from ....utils import instantiate_from_config, print_rank_zero
from ..continuous_noise_sampling import ContinuousNoiseSampling
from ..masked_loss import MaskedDiffusionLoss
from ..nonfinite_utils import summarize_nonfinite_tensor
from ..noise_sampling import NoiseSampling
from .infonce_loss import InfoNCELoss
from .latentprocess import LatentProcess


class LatentDDMLoss(MaskedDiffusionLoss):
    """Loss function for Latent Discrete Diffusion Model (LatentDDM).

    This loss combines:
    1. The standard masked diffusion loss for the token sequence `x`.
    2. A reconstruction loss for the latent vector `y` (e.g., MSE).
    """

    def __init__(
        self,
        # Config for the core model to use
        core_model_config: dict,
        ratio_x_to_y_stage_1: float | None = None,
        ratio_x_to_y_stage_2: float | None = None,
        data_loss_weight_stage_1: float = 1.0,
        data_loss_weight_stage_2: float = 1.0,
        latent_loss_weight_stage_1: float = 1.0,
        latent_loss_weight_stage_2: float = 1.0,
        latent_loss_weight_schedule: str | None = None,
        latent_loss_weight_schedule_steps: int | None = None,
        training_stage: int | None = None,
        # New options for using target labels
        use_precomputed_latent: bool = False,
        use_encoder_supervision: bool = False,
        encoder_supervision_weight: float = 1.0,
        p_cfg: float = 0.0,  # p_tau
        use_p_r: bool = False,  # p_r
        use_p_r_eval: bool = False,
        kl_regularization_weight: float = 0.0,
        conditional_independence_loss: bool = False,
        conditional_independence_kl_direction: str = 'forward',  # 'forward' for D_KL(P || Q), 'backward' for D_KL(Q || P)
        conditional_independence_detach_p: bool = True,  # whether to detach P in the KL loss
        independent_timesteps: bool = False,
        use_data_loss_elbo: bool = False,
        use_latent_loss_elbo: bool = False,
        # InfoNCE two-view invariance parameters
        use_infonce_loss: bool = False,
        infonce_weight: float = 1.0,
        infonce_temperature: float = 0.1,
        p_mask_input_enc: float = 0.0,  # probability of masking tokens for two-view corruption
        measure_wall_clock_time: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert self.time_sampler.eps > 0, 'eps must be larger than zero, will be used to determine step size'

        # We override the noise_sampler_config from the parent class
        # by passing the x_noise_sampler_config as the main one.
        self.core_model: LatentProcess = instantiate_from_config(core_model_config)
        self.x_noise_sampler: NoiseSampling = self.core_model.x_noise_sampler
        self.y_noise_sampler: ContinuousNoiseSampling = self.core_model.y_noise_sampler
        self.noise_sampler = self.x_noise_sampler

        self.ratio_x_to_y_stage_1 = ratio_x_to_y_stage_1
        self.ratio_x_to_y_stage_2 = ratio_x_to_y_stage_2
        self.data_loss_weight_stage_1 = data_loss_weight_stage_1
        self.data_loss_weight_stage_2 = data_loss_weight_stage_2
        self.latent_loss_weight_stage_1 = latent_loss_weight_stage_1
        self.latent_loss_weight_stage_2 = latent_loss_weight_stage_2
        self.latent_loss_weight_schedule = latent_loss_weight_schedule
        self.latent_loss_weight_schedule_steps = latent_loss_weight_schedule_steps
        self.latent_loss_weight_training_steps = 0  # to keep track of training steps for latent_loss_weight_schedule

        self.total_x_y_steps_stage_1 = sum(self.ratio_x_to_y_stage_1) if self.ratio_x_to_y_stage_1 is not None else 0
        self.total_x_y_steps_stage_2 = sum(self.ratio_x_to_y_stage_2) if self.ratio_x_to_y_stage_2 is not None else 0
        self.total_x_y_loss_steps = self.total_x_y_steps_stage_1 + self.total_x_y_steps_stage_2
        self.training_stage = training_stage

        assert self.training_stage in [1, 2], f'got training_stage = {self.training_stage}'

        self.use_precomputed_latent = use_precomputed_latent
        self.use_encoder_supervision = use_encoder_supervision
        self.encoder_supervision_weight = encoder_supervision_weight
        self.p_cfg = p_cfg
        self.use_p_r = use_p_r
        self.use_p_r_eval = use_p_r_eval
        self.kl_regularization_weight = kl_regularization_weight
        self.conditional_independence_loss = conditional_independence_loss
        self.conditional_independence_kl_direction = conditional_independence_kl_direction
        self.conditional_independence_detach_p = conditional_independence_detach_p
        self.independent_timesteps = independent_timesteps
        self.use_latent_loss_elbo = use_latent_loss_elbo
        self.use_data_loss_elbo = use_data_loss_elbo
        self.measure_wall_clock_time = measure_wall_clock_time

        # InfoNCE two-view invariance
        self.use_infonce_loss = use_infonce_loss
        self.infonce_weight = infonce_weight
        self.p_mask_input_enc = p_mask_input_enc
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

    def _core_model_forward_timed(self, **kwargs):
        if not self.measure_wall_clock_time:
            return self.core_model.model_forward(**kwargs)
        x = kwargs.get('x', None)
        if x is None or not torch.is_tensor(x):
            return self.core_model.model_forward(**kwargs)
        self._sync_device_for_timing(x.device)
        start_time = time.perf_counter()
        output = self.core_model.model_forward(**kwargs)
        self._sync_device_for_timing(x.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print_rank_zero(f'[LatentDDMLoss] core_model.model_forward wall clock: {elapsed_ms:.3f} ms')
        return output

    def _log_nonfinite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            return
        if torch.isfinite(tensor).all():
            return
        stats = summarize_nonfinite_tensor(tensor)
        print_rank_zero(
            f'[LatentDDMLoss] Non-finite {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} '
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
    ) -> None:
        masked_fraction_x = (x_t == self.mask_token_id).float().mean().item()
        attention_keep_fraction = attention_mask.float().mean().item()
        print_rank_zero(
            '[LatentDDMLoss] Prediction context: '
            f't_x[min={t_x.min().item():.6f}, max={t_x.max().item():.6f}, mean={t_x.mean().item():.6f}] '
            f't_y[min={t_y.min().item():.6f}, max={t_y.max().item():.6f}, mean={t_y.mean().item():.6f}] '
            f'x_masked_frac={masked_fraction_x:.6f} '
            f'attention_keep_frac={attention_keep_fraction:.6f} '
            f'x_vocab={x_0_pred_logits.shape[-1]}',
        )
        self._log_nonfinite('x_t', x_t)
        self._log_nonfinite('y_t', y_t)
        self._log_nonfinite('x_0_pred_logits', x_0_pred_logits)

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

    def get_data_loss_t(self, x_0_pred_logits, x_0, x_t, t, attention_mask, labels, example_mask=None):
        x_loss_weight = self.x_noise_sampler(t).loss_weight
        loss_dict_x = self.get_loss(
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

    def compute_latent_loss_weight(self):
        latent_loss_weight = self.latent_loss_weight_stage_1 if self.training_stage == 1 else self.latent_loss_weight_stage_2
        if self.latent_loss_weight_schedule is None:
            return latent_loss_weight
        if self.latent_loss_weight_schedule == 'warmup':
            assert self.latent_loss_weight_schedule_steps is not None, (
                'latent_loss_weight_schedule_steps must be set for warmup'
            )
            current_step = min(self.latent_loss_weight_training_steps, self.latent_loss_weight_schedule_steps)
            self.latent_loss_weight_training_steps += 1
            latent_loss_weight = latent_loss_weight * current_step / self.latent_loss_weight_schedule_steps
            return latent_loss_weight
        if self.latent_loss_weight_schedule == 'delayed_warmup':
            assert self.latent_loss_weight_schedule_steps is not None, (
                'latent_loss_weight_schedule_steps must be set for delayed_warmup'
            )
            delay_steps = int(0.2 * self.latent_loss_weight_schedule_steps)
            warmup_steps = self.latent_loss_weight_schedule_steps - delay_steps
            current_step = min(self.latent_loss_weight_training_steps, self.latent_loss_weight_schedule_steps)
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
        latent_denoiser: ContinuousDiTModel | None,
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
        t_x = self.time_sampler(x_0.size(0)).to(x_0.device)
        t_y = self.time_sampler(x_0.size(0)).to(x_0.device) if self.independent_timesteps else t_x



        # 4. Encode to y_0, obtain noised outputs from x_0 and y_0
        x_t, y_0, encode_output, y_t, z_t = self.core_model.sample_forward(
            batch,
            encoder,
            x_0,
            t_x,
            t_y,
            maskable_mask,
            model_config=joint_denoiser.config,
            use_p_r=self.use_p_r if self.training else self.use_p_r_eval,
            use_precomputed_latent=self.use_precomputed_latent,
            use_encoder_supervision=self.use_encoder_supervision,
            training=self.training,
        )
        # print_rank_zero(f'Mean difference across batch between y_0 and y_t: {torch.mean(torch.abs(y_0 - y_t), dim=1)}')
        y_0_mean = encode_output['y_0']['mean']
        y_0_std = encode_output['y_0']['std']

        # 7. Calculate losses, and some more logging
        loss_dict_x = {'loss_x': torch.tensor(0.0, device=x_0.device), 'elbo_x': torch.tensor(0.0, device=x_0.device)}
        loss_dict_y = {'loss_y': torch.tensor(0.0, device=x_0.device), 'elbo_y': torch.tensor(0.0, device=x_0.device)}
        # create union of loss_dict_x and loss_dict_y
        loss_dict = {**loss_dict_x, **loss_dict_y}

        # whether to drop labels for classifier-free guidance
        if self.p_cfg > 0 and self.training:
            # drop_everything = torch.rand(1).item() < self.p_cfg  # and (x_loss_enabled == 1)
            # drop_mask = torch.zeros(x_0.shape[0], dtype=torch.bool, device=x_0.device).bool()
            # if drop_everything:
            #     drop_mask = torch.ones(x_0.shape[0], dtype=torch.bool, device=x_0.device).bool()
            drop_mask = (torch.rand(x_0.shape[0], device=x_0.device) < self.p_cfg) & (x_loss_enabled == 1)
            if drop_mask.any():
                # print_rank_zero(f'Dropping y labels for {drop_mask.sum().item()} samples over {x_0.shape[0]} in the batch')
                # set y_t and y_0 to torch.zeros_like(y_0[0]) where drop_mask is True
                y_t[drop_mask] = torch.zeros_like(y_t[0])
                y_0[drop_mask] = torch.zeros_like(y_0[0])
                y_loss_enabled[drop_mask] = 0

        # predict from t > 0
        x_0_pred_logits, model_pred_mean, model_pred_std = self._core_model_forward_timed(
            joint_denoiser=joint_denoiser,
            latent_denoiser=latent_denoiser,
            x=x_t,
            y=y_t,
            t_x=t_x,
            t_y=t_y,
            y_0=y_0,
            attention_mask=attention_mask,
        )

        data_loss_weight = self.compute_data_loss_weight()
        latent_loss_weight = self.compute_latent_loss_weight()

        # Loss for x (token sequence)
        if (x_loss_enabled > 0).any():
            if not torch.isfinite(x_0_pred_logits).all():
                self._log_prediction_context(
                    x_t=x_t,
                    y_t=y_t,
                    t_x=t_x,
                    t_y=t_y,
                    attention_mask=attention_mask,
                    x_0_pred_logits=x_0_pred_logits,
                )
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

        # Loss for y (latent vector) -- v_t is the velocity of the latent vector
        if (y_loss_enabled > 0).any() and latent_loss_weight > 0.0:
            y_mask = y_loss_enabled > 0
            loss_dict_y_t = self.core_model.get_latent_loss_t(
                y_0_mean.float()[y_mask],
                y_0_std.float()[y_mask],
                y_0.float()[y_mask],
                model_pred_mean.float()[y_mask],
                model_pred_std.float()[y_mask],
                t_y.float()[y_mask],
                z_t.float()[y_mask],
                self.time_sampler.eps,
            )
            loss_dict_y['loss_y'] += loss_dict_y_t['loss_y_t']
            loss_dict_y['elbo_y'] += loss_dict_y_t['elbo_y_t']
            loss_dict.update(loss_dict_y_t)
            loss_dict.update(loss_dict_y)

        if self.conditional_independence_loss and (x_loss_enabled > 0).any():
            # Create fully masked x_t for conditional independence
            # we could also sample t_masked > t_x instead of t_masked = 1
            # which would be closer to current self-distillation, but well.
            t_masked = torch.ones_like(t_x) * (1 - self.time_sampler.eps)
            alphas_x_masked = self.x_noise_sampler(t_masked)
            x_t_masked = self.masked_process.sample_forward(x_0, alphas_x_masked, maskable_mask, model_config=joint_denoiser.config)
            # x_t_masked = torch.full_like(x_t, self.mask_token_id)

            x_0_pred_logits_cond_ind, y_0_pred_mean_cond_ind, y_0_pred_std_cond_ind = self._core_model_forward_timed(
                joint_denoiser=joint_denoiser,
                latent_denoiser=latent_denoiser,
                x=x_t_masked,
                y=y_t,
                t_x=t_masked,
                t_y=t_y,
                y_0=y_0,
                attention_mask=attention_mask,
            )

            # retrieve loss_dict_x in order to know how to format the keys of loss_dict_x_0
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

        # Manage possible nans
        if not torch.isfinite(loss_dict['loss']):
            print_rank_zero('[LatentDDMLoss] Non-finite loss detected!')
            self._log_nonfinite('y_0', y_0)
            self._log_nonfinite('y_t', y_t)
            self._log_nonfinite('x_0_pred_logits', x_0_pred_logits)
            self._log_nonfinite('loss', loss_dict['loss'])

        loss_dict['loss'] = torch.nan_to_num(loss_dict['loss'])

        # 10. Compute the encoder supervision loss
        if self.use_encoder_supervision:
            y_0_encoded = encode_output['y_0_encoded']['mean'].float()
            y_0_real = encode_output['y_0_real']['mean'].float()
            loss_dict['loss_encoder'] = F.mse_loss(y_0_encoded, y_0_real)
            loss_dict['loss'] += self.encoder_supervision_weight * loss_dict['loss_encoder']

        # 11. Add KL regularization loss if any
        # 2. bis if self.kl_regularization_weight > 0:
        if self.kl_regularization_weight > 0 and latent_loss_weight > 0.0:
            # compute KL regularization to keep y_0 close to N(0, I)
            # loss_kl_regularization = 0.5 * (y_0_mean**2 + y_0_std**2 - torch.log(y_0_std**2 + 1e-8) - 1)
            loss_kl_regularization = latent_loss_weight * self.kl_regularization_weight * 0.5 * y_0_mean**2
            loss_kl_regularization = loss_kl_regularization.sum(dim=-1).mean()
            loss_dict['loss_kl_regularization'] = loss_kl_regularization
            loss_dict['loss'] += loss_dict['loss_kl_regularization']

        # 12. Add InfoNCE two-view invariance loss
        if self.use_infonce_loss and self.training and self.p_mask_input_enc > 0:
            # Create two corrupted views by masking encoder input
            # View 1: randomly mask tokens with probability p_mask_input_enc
            mask_view1 = torch.rand(x_0.shape, device=x_0.device) < self.p_mask_input_enc
            x_0_view1 = x_0.clone()
            x_0_view1[mask_view1 & maskable_mask] = self.mask_token_id

            # View 2: independent random masking
            mask_view2 = torch.rand(x_0.shape, device=x_0.device) < self.p_mask_input_enc
            x_0_view2 = x_0.clone()
            x_0_view2[mask_view2 & maskable_mask] = self.mask_token_id

            # Encode both views to get continuous latents (z_e, pre-quantization)
            with torch.no_grad() if encode_output.get('stop_gradient', False) else torch.enable_grad():
                # Encode view 1
                encode_output_view1 = encoder(x_0_view1, attention_mask=attention_mask)
                z_e_view1 = encode_output_view1.y_pred  # (B, S', D') or (B, D')

                # Encode view 2
                encode_output_view2 = encoder(x_0_view2, attention_mask=attention_mask)
                z_e_view2 = encode_output_view2.y_pred  # (B, S', D') or (B, D')

            # Ensure we have sequence dimension for pooling
            if z_e_view1.dim() == 2:
                # If already pooled (B, D), add dummy sequence dimension
                z_e_view1 = z_e_view1.unsqueeze(1)  # (B, 1, D)
                z_e_view2 = z_e_view2.unsqueeze(1)  # (B, 1, D)
                # Create dummy mask (all ones)
                infonce_mask = torch.ones(x_0.shape[0], 1, device=x_0.device, dtype=torch.bool)
            else:
                # Use attention mask for pooling
                infonce_mask = attention_mask if attention_mask is not None else None

            # Compute InfoNCE loss
            infonce_result = self.infonce_loss_fn(
                embeddings_view1=z_e_view1,
                embeddings_view2=z_e_view2,
                attention_mask_view1=None,
                attention_mask_view2=None,
            )
            loss_dict['loss_infonce'] = infonce_result['loss']
            loss_dict['loss'] += self.infonce_weight * loss_dict['loss_infonce']

        return loss_dict

    # def get_data_loss_0(self, x_0_pred_logits_eps, x_0, x_t_eps, t_eps, attention_mask, labels, loss_dict_x):
    #     loss_dict_x_0 = {f'{k}_0': torch.zeros_like(v) for k, v in loss_dict_x.items() if '_x' in k}
    #     # check if x_t_eps contains mask
    #     has_mask = (x_t_eps == self.mask_token_id).any()
    #     if has_mask:
    #         loss_dict_x_0 = self.get_loss(
    #             logits=x_0_pred_logits_eps,
    #             x_0=x_0,
    #             x_t=x_t_eps,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             loss_weight=torch.ones_like(t_eps),
    #         )
    #         loss_dict_x_0 = {f'{k}_0': v for k, v in loss_dict_x_0.items() if '_x' in k}
    #     return loss_dict_x_0


# last prediction step from t = eps if self.latent_strategy['variance'] is not None
# if (self.core_model.latent_strategy['variance'] is not None) and (
#     self.core_model.latent_strategy['variance'] != 'fixed'
# ):
#     assert False, 'Must update; not fully tested yet'
#     t_eps = torch.ones_like(t_x) * self.time_sampler.eps
#     x_t_eps, y_t_eps, z_t_eps = self.core_model.sample_forward(x_0, y_0, t_eps, maskable_mask, joint_denoiser.config)
#     x_0_pred_logits_eps, y_0_pred_mean_eps, y_0_pred_std_eps = self.core_model.model_forward(
#         joint_denoiser=joint_denoiser,
#         latent_denoiser=latent_denoiser,
#         x=x_t_eps,
#         y=y_t_eps,
#         t=t_eps,
#         y_0=y_0,
#     )
#     # Loss for x (token sequence)
#     if x_loss_enabled == 1:
#         # retrieve loss_dict_x in order to know how to format the keys of loss_dict_x_0
#         loss_dict_x_0 = self.get_data_loss_0(
#             x_0_pred_logits_eps.float(),
#             x_0,
#             x_t_eps,
#             t_eps,
#             attention_mask,
#             labels,
#             loss_dict_x,
#         )
#         loss_dict_x['loss_x'] += loss_dict_x_0['loss_x_0']
#         loss_dict_x['elbo_x'] += loss_dict_x_0['elbo_x_0']
#         loss_dict.update(loss_dict_x_0)
#         loss_dict.update(loss_dict_x)

#     # Loss for y (latent vector) -- v_t is the velocity of the latent vector
#     if y_loss_enabled == 1:
#         loss_dict_y_0 = self.core_model.get_latent_loss_0(
#             y_0_mean.float(),
#             y_0_std.float(),
#             y_0.float(),
#             y_0_pred_mean_eps.float(),
#             y_0_pred_std_eps.float(),
#             self.time_sampler.eps,
#         )
#         loss_dict_y['loss_y'] += loss_dict_y_0['loss_y_0']
#         loss_dict_y['elbo_y'] += loss_dict_y_0['elbo_y_0']
#         loss_dict.update(loss_dict_y_0)
#         loss_dict.update(loss_dict_y)
