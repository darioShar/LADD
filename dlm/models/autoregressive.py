from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, PreTrainedModel

from .lm import LM
from .output import SampleOutput


class HFLoss:
    """
    Compute the loss for the autoregressive model.
    """

    def __call__(self, model: PreTrainedModel, batch: dict, **kwargs: dict):
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].detach().clone()
        if 'attention_mask' not in batch:
            batch['attention_mask'] = torch.ones_like(
                batch['input_ids'],
                dtype=torch.long,
                device=batch['input_ids'].device,
            )
        input_ids = batch['input_ids'][:, :-1].contiguous()
        attention_mask = batch['attention_mask'][:, :-1].contiguous()
        target_attention_mask = batch['attention_mask'][:, 1:].contiguous()
        labels = batch['labels'][:, 1:].contiguous().clone()
        # Mask supervision using the shifted attention mask, not pad_token_id.
        # In the Qwen LM1B setup, pad/eos/bos share the same token id, so masking by
        # token value would incorrectly remove all EOS targets from training.
        labels[target_attention_mask == 0] = -100

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction='none',
        )
        valid_tokens = labels != -100
        num_valid_tokens = valid_tokens.sum().clamp_min(1)
        total_nll = per_token_loss.sum()
        loss = total_nll / num_valid_tokens
        elbo_x = total_nll / input_ids.size(0)

        ppl = torch.exp(loss)
        return dict(loss=loss, elbo=elbo_x, elbo_x=elbo_x, nll=loss, perplexity=ppl)


@dataclass
class AutoregressiveSampler:
    num_inference_steps: int | None


class Autoregressive(LM):
    def __init__(
        self,
        loss_config: dict | None = None,
        generation_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(loss_config=loss_config, **kwargs)
        if loss_config is None:
            self.loss_fn = HFLoss()
        self.generation_config = GenerationConfig(**generation_config) if generation_config else None
        self.sampler = AutoregressiveSampler(
            num_inference_steps=self._resolve_num_inference_steps(),
        )

    def _resolve_num_inference_steps(self) -> int | None:
        if self.generation_config is not None:
            if self.generation_config.max_length is not None:
                return int(self.generation_config.max_length)
            if self.generation_config.max_new_tokens is not None:
                return int(self.generation_config.max_new_tokens)
        if self.sequence_length is not None:
            return int(self.sequence_length)
        return None

    def sample(
        self,
        batch: dict[str, Any],
        max_batch_size: int | None = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> SampleOutput:
        if 'prompt_ids' in batch:
            prompt_ids = batch['prompt_ids']
        else:
            # use the first token (usually bos token) as the prompt
            prompt_ids = batch['input_ids'][:, :1].clone()
        if max_batch_size is not None:
            prompt_ids = prompt_ids[:max_batch_size]
        if self.generation_config is None:
            generation_config = GenerationConfig(
                # multinomial sampling
                do_sample=True,
                num_beams=1,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_length=self.model.config.max_position_embeddings,
            )
        else:
            generation_config = GenerationConfig.from_dict(self.generation_config.to_dict())

        target_total_length = kwargs.get('num_inference_steps')
        if target_total_length is None:
            if generation_config.max_length is not None:
                target_total_length = generation_config.max_length
            elif generation_config.max_new_tokens is not None:
                target_total_length = prompt_ids.size(1) + generation_config.max_new_tokens
            else:
                target_total_length = self.sequence_length
        target_total_length = max(int(target_total_length), int(prompt_ids.size(1)))
        num_new_tokens = target_total_length - int(prompt_ids.size(1))

        generation_config.max_length = None
        generation_config.max_new_tokens = num_new_tokens
        generation_config.min_new_tokens = None
        # For packed-sequence generation we want EOS to be sampled as a normal token,
        # not used as an early-stop signal. This better matches LM1B sentence packing,
        # where <|endoftext|> appears inside the sequence as a boundary token.
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        if generation_config.temperature is None:
            generation_config.temperature = 1.0

        with self.ema_scope(), torch.no_grad():
            outputs = self.model.generate(
                prompt_ids,
                generation_config=generation_config,
            )
        output_ids = outputs
        completion_ids = outputs[:, prompt_ids.size(1) :]
        return SampleOutput(
            output_ids=output_ids,
            prompts=self.tokenizer.batch_decode(
                prompt_ids,
                skip_special_tokens=skip_special_tokens,
            ),
            completions=self.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=skip_special_tokens,
            ),
        )

    def compute_metrics(self, batch, loss_dict):
        # ce loss for val dataset
        loss = loss_dict['val/loss']
        bpc = loss / torch.log(torch.tensor(2.0, device=loss.device))
        return {'bpc': bpc}
