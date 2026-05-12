import inspect
import re

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import resolve_local_files_only


def load_teacher_model(
    model_name_or_path: str = 'gpt2-large',
) -> tuple[nn.Module, AutoTokenizer]:
    """
    Loads a pretrained model and tokenizer for perplexity computation.

    Args:
        model_name_or_path: HuggingFace model name or path.
        torch_dtype: Data type for model weights (e.g., bfloat16 for performance).

    Returns:
        A tuple of (model, tokenizer).
    """
    local_files_only = resolve_local_files_only()
    # Ensure teacher tensors are created as normal tensors even if caller is inside
    # Lightning evaluation inference_mode.
    with torch.inference_mode(False):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            problem_type=None,  # Suppress loss_type warnings
            local_files_only=local_files_only,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tokenizer



class GenerativePerplexityMetric(Metric):
    """
    Computes generative perplexity using a pretrained autoregressive model.

    This metric processes generated sequences in batches for efficiency, decodes them
    using a source tokenizer, re-encodes them with a perplexity model's tokenizer,
    and calculates perplexity based on the model's cross-entropy loss.
    """

    def __init__(
        self,
        perplexity_model: nn.Module | None = None,
        perplexity_tokenizer: AutoTokenizer | None = None,
        source_tokenizer: AutoTokenizer | None = None,
        max_length: int = 1024,
        yazid: bool = True,
    ):
        """
        Args:
            perplexity_model: Pretrained model for perplexity computation.
            perplexity_tokenizer: Tokenizer for the perplexity model.
            source_tokenizer: Tokenizer used by the source model for decoding.
            max_length: Maximum sequence length for the perplexity model.
            yazid: If True, compute generative perplexity using Yazid's masking
                rule (ignore EOS targets except the first EOS per sequence).
        """
        super().__init__()

        # Register the model as a submodule so Lightning moves it to correct device
        # We'll exclude it from checkpoints using state_dict filtering
        self.perplexity_model = perplexity_model
        self.perplexity_tokenizer = perplexity_tokenizer
        self.source_tokenizer = source_tokenizer
        self.max_length = max_length
        self.yazid = yazid
        self.source_pad_token_id = None  # Will be set in lazy_initialize

        # Metric states for distributed reduction
        self.add_state('total_loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_tokens', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_sequences', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('failed_sequences', default=torch.tensor(0), dist_reduce_fx='sum')

    def lazy_initialize(self, perplexity_model, perplexity_tokenizer, source_tokenizer, max_length) -> None:
        self.perplexity_model = perplexity_model
        self.perplexity_tokenizer = perplexity_tokenizer
        self.source_tokenizer = source_tokenizer
        self.max_length = max_length
        self.source_pad_token_id = source_tokenizer.pad_token_id

        # Ensure the perplexity tokenizer has a pad token for batching
        if self.perplexity_tokenizer.pad_token is None:
            self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token

        # CRITICAL: Verify that source and perplexity tokenizers use the same special token STRINGS.
        # This is required because we decode with source_tokenizer (keeping special tokens as strings)
        # and re-encode with perplexity_tokenizer, which must recognize those special token strings.
        # Note: We compare token strings, not IDs, since different tokenizers assign different IDs.

        # Check EOS token
        assert self.source_tokenizer.eos_token is not None, (
            'source_tokenizer must have an EOS token defined'
        )
        assert self.perplexity_tokenizer.eos_token is not None, (
            'perplexity_tokenizer must have an EOS token defined'
        )
        assert self.source_tokenizer.eos_token == self.perplexity_tokenizer.eos_token, (
            f'source_tokenizer and perplexity_tokenizer must use the same EOS token string. '
            f'Got source={self.source_tokenizer.eos_token}, perplexity={self.perplexity_tokenizer.eos_token}'
        )

        # Check BOS token (can be None for some tokenizers, so only check equality if both are defined)
        if self.source_tokenizer.bos_token is not None or self.perplexity_tokenizer.bos_token is not None:
            assert self.source_tokenizer.bos_token == self.perplexity_tokenizer.bos_token, (
                f'source_tokenizer and perplexity_tokenizer must use the same BOS token string. '
                f'Got source={self.source_tokenizer.bos_token}, perplexity={self.perplexity_tokenizer.bos_token}'
            )

        # Check PAD token
        assert self.source_tokenizer.pad_token is not None, (
            'source_tokenizer must have a PAD token defined'
        )
        assert self.perplexity_tokenizer.pad_token is not None, (
            'perplexity_tokenizer must have a PAD token defined'
        )
        assert self.source_tokenizer.pad_token == self.perplexity_tokenizer.pad_token, (
            f'source_tokenizer and perplexity_tokenizer must use the same PAD token string. '
            f'Got source={self.source_tokenizer.pad_token}, perplexity={self.perplexity_tokenizer.pad_token}. '
            f'This is critical for correct attention masking when decoded padding appears in text.'
        )

    @torch.no_grad()
    def deprecated_update(self, token_ids: torch.Tensor):
        """Update the metric with a batch of generated sequences."""
        if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
            return

        self.perplexity_model.eval()
        self.total_sequences += token_ids.shape[0]

        # 1. Decode the entire batch of sequences using the source tokenizer.
        # KEEPING special tokens to preserve sentence boundaries (see update() for details).
        texts = self.source_tokenizer.batch_decode(token_ids, skip_special_tokens=False)

        # 2. Filter out empty or whitespace-only strings after decoding.
        valid_texts = [text for text in texts if text and not text.isspace()]
        num_failed = token_ids.shape[0] - len(valid_texts)
        if num_failed > 0:
            self.failed_sequences += num_failed
        if not valid_texts:
            return

        # 3. Re-encode the valid texts in a batch with the perplexity tokenizer.
        # The metric's `self.device` attribute is provided by TorchMetrics.
        # Use add_special_tokens=False to avoid duplicating special tokens.
        encoding = self.perplexity_tokenizer(
            valid_texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True,  # Pad to the longest sequence in the batch
            add_special_tokens=False,
        ).to(self.device)

        input_ids = encoding['input_ids']

        # 4. Perform a single forward pass with the batch.
        with torch.no_grad():
            attention_mask = encoding['attention_mask'].to(self.device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # <-- critical

            # Hugging Face models calculate loss internally when `labels` are provided.
            # Using input_ids as labels is standard for perplexity calculation.
            outputs = self.perplexity_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # The returned loss is the *average* cross-entropy over tokens.
            # To get the total loss, we multiply by the number of tokens.
            loss = outputs.loss
            num_tokens = (input_ids != self.perplexity_tokenizer.pad_token_id).sum()

            # Update metric state with tensor operations
            if not torch.isnan(loss) and not torch.isinf(loss):
                self.total_loss += loss.detach() * num_tokens
                self.total_tokens += num_tokens
            else:
                self.failed_sequences += len(valid_texts)

    @torch.no_grad()
    def update(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Update the metric with a batch of generated sequences (token ids from the *source* model).

        IMPORTANT: This method handles padding tokens correctly by using attention_mask to filter them out.
        Works for both sentence padding (left/right padding) and sentence packing (EOS separators).

        Args:
            token_ids: Token IDs from the source model (B, L)
            attention_mask: Optional attention mask (B, L) where 1=real token, 0=padding.
                           If None, will attempt to infer from pad_token_id (less reliable).
        """
        if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
            return

        self.perplexity_model.eval()
        self.total_sequences += token_ids.shape[0]

        # 1) Filter out padding tokens from each sequence BEFORE decoding.
        # This prevents left-padded sequences from having padding tokens decoded as literal text.
        # For sentence packing: attention_mask is all 1s, so this is a no-op.
        # For sentence padding: removes PAD tokens while preserving actual content + EOS/BOS.

        if attention_mask is not None:
            # Use attention mask to identify real tokens (most reliable method)
            # For sentence padding: attention_mask has 0s for padding, 1s for content
            # For sentence packing: attention_mask should be all 1s (but may not be provided)
            filtered_sequences = []
            for i in range(token_ids.shape[0]):
                # Extract tokens where attention_mask is 1
                seq_real_tokens = token_ids[i][attention_mask[i].bool()]
                if seq_real_tokens.numel() > 0:
                    filtered_sequences.append(seq_real_tokens)
                else:
                    self.failed_sequences += 1

            if not filtered_sequences:
                return

            # 2) Decode filtered sequences, KEEPING special tokens (EOS/BOS) to preserve sentence boundaries.
            # This is safe now because we've removed PAD tokens, so only meaningful special tokens remain.
            texts = [self.source_tokenizer.decode(seq, skip_special_tokens=False) for seq in filtered_sequences]

        elif self.source_pad_token_id is not None and self.source_pad_token_id not in [
            self.source_tokenizer.eos_token_id,
            self.source_tokenizer.bos_token_id,
        ]:
            # Fallback: infer padding from pad_token_id, but ONLY if pad_token_id is distinct from eos/bos.
            # This prevents accidentally filtering out sentence boundaries.
            non_pad_mask = token_ids != self.source_pad_token_id

            filtered_sequences = []
            for i in range(token_ids.shape[0]):
                seq_non_pad = token_ids[i][non_pad_mask[i]]
                if seq_non_pad.numel() > 0:
                    filtered_sequences.append(seq_non_pad)
                else:
                    self.failed_sequences += 1

            if not filtered_sequences:
                return

            texts = [self.source_tokenizer.decode(seq, skip_special_tokens=False) for seq in filtered_sequences]
        else:
            # No padding information, or pad_token_id == eos/bos (can't distinguish padding from boundaries).
            # Decode as-is. This is correct for sentence packing (no padding) but will include padding
            # for sentence padding when pad/eos/bos share the same ID.
            texts = self.source_tokenizer.batch_decode(token_ids, skip_special_tokens=False)

        # 3) Filter empties
        valid_texts = [t for t in texts if t and not t.isspace()]
        num_failed = len(texts) - len(valid_texts)
        if num_failed > 0:
            self.failed_sequences += num_failed
        if not valid_texts:
            return

        # 4) Re-encode with perplexity tokenizer.
        # Use add_special_tokens=False to avoid adding duplicate BOS/EOS tokens,
        # since the text already contains special token strings from the source tokenizer.
        enc = self.perplexity_tokenizer(
            valid_texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=False,
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        # 5) Compute per-token NLL.
        # Default behavior follows Yazid's implementation for packed sequences.
        if input_ids.size(1) < 2:
            self.failed_sequences += len(valid_texts)
            return

        with torch.autocast(device_type='cuda', enabled=False):
            outputs = self.perplexity_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

            if self.yazid:
                # Yazid masking rule:
                # keep tokens after position 0 except EOS targets after the first EOS.
                nlls = F.cross_entropy(
                    logits[:, :-1].transpose(-1, -2),
                    input_ids[:, 1:],
                    reduction='none',
                )
                eos_token_id = self.perplexity_tokenizer.eos_token_id
                first_eos = (input_ids == eos_token_id).cumsum(-1) == 1
                token_mask = input_ids != eos_token_id
                valid_tokens = first_eos[:, 1:] + token_mask[:, 1:]
                loss_sum = (nlls * valid_tokens).sum()
                num_tokens = valid_tokens.sum()
            else:
                labels = input_ids.clone()
                labels[attention_mask.eq(0)] = -100
                outputs = self.perplexity_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss.float()
                num_tokens = attention_mask.sum()
                loss_sum = loss * num_tokens

        # 6) Accumulate sum of token-level losses and count
        if num_tokens.item() <= 0:
            self.failed_sequences += len(valid_texts)
        elif torch.isfinite(loss_sum):
            self.total_loss += loss_sum
            self.total_tokens += num_tokens
        else:
            self.failed_sequences += len(valid_texts)

    def compute(self) -> torch.Tensor:
        if self.total_tokens == 0 or self.total_sequences == 0:
            return self._get_empty_result()
        avg_loss = self.total_loss / self.total_tokens.clamp(min=1)
        return torch.exp(avg_loss)  # perplexity

    def _get_empty_result(self) -> torch.Tensor:
        """Return default value for an empty state."""
        return torch.tensor(float('inf'), device=self.device)




class GradientMomentMetric(Metric):
    """
    Computes the centered Gradient Moment metric of a reference causal LM.

    This implements the estimator from Eq. (14) of the attached paper:
        (g1 - q1)^T (g2 - q2)
    where each term is a gradient of the reference model log-likelihood
    (equivalently, up to sign, the gradient of mean negative log-likelihood).

    Important:
        - update() expects TWO independent minibatch pairs:
            (generated_a, data_a) and (generated_b, data_b)
        - lower is better
        - the true metric is nonnegative, but the unbiased finite-sample estimator
          can be slightly negative
        - this is much more expensive than perplexity because it backprops through
          the teacher model

    The metric decodes tokens with the source tokenizer, then re-tokenizes with the
    reference model tokenizer, similarly to the generative perplexity metric.
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    def __init__(
        self,
        reference_model: nn.Module | None = None,
        reference_tokenizer: AutoTokenizer | None = None,
        source_tokenizer: AutoTokenizer | None = None,
        max_length: int = 1024,
        include_parameter_regex: str | None = None,
    ):
        super().__init__()

        self.reference_model = reference_model
        self.reference_tokenizer = reference_tokenizer
        self.source_tokenizer = source_tokenizer
        self.max_length = max_length
        self.include_parameter_regex = include_parameter_regex

        self.source_pad_token_id = None
        self._keep_special_tokens = True
        self._selected_params: list[nn.Parameter] = []
        self._forward_accepts_use_cache = False

        self.add_state('total_estimate', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_pairs', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_sequences', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('failed_sequences', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('failed_pairs', default=torch.tensor(0), dist_reduce_fx='sum')

    def lazy_initialize(
        self,
        reference_model: nn.Module,
        reference_tokenizer: AutoTokenizer,
        source_tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.reference_model = reference_model
        self.reference_tokenizer = reference_tokenizer
        self.source_tokenizer = source_tokenizer
        self.max_length = max_length

        self.source_pad_token_id = source_tokenizer.pad_token_id

        if self.reference_tokenizer.pad_token is None:
            self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token

        self._keep_special_tokens = self._can_preserve_special_tokens()
        self._selected_params = self._select_reference_parameters()

        if len(self._selected_params) == 0:
            raise ValueError('No reference-model parameters were selected for GradientMomentMetric.')

        try:
            sig = inspect.signature(self.reference_model.forward)
            self._forward_accepts_use_cache = 'use_cache' in sig.parameters
        except (TypeError, ValueError):
            self._forward_accepts_use_cache = False

    def _can_preserve_special_tokens(self) -> bool:
        """
        Keep source special tokens only when the token STRINGS agree between
        source and reference tokenizers. Otherwise decode without special tokens.
        """
        if self.source_tokenizer is None or self.reference_tokenizer is None:
            return False

        for attr in ('eos_token', 'bos_token', 'pad_token'):
            source_tok = getattr(self.source_tokenizer, attr, None)
            ref_tok = getattr(self.reference_tokenizer, attr, None)

            if source_tok is None and ref_tok is None:
                continue
            if source_tok != ref_tok:
                return False

        return True

    def _select_reference_parameters(self) -> list[nn.Parameter]:
        pattern = re.compile(self.include_parameter_regex) if self.include_parameter_regex else None

        params: list[nn.Parameter] = []

        for name, param in self.reference_model.named_parameters():
            if not torch.is_floating_point(param):
                continue
            if pattern is not None and pattern.search(name) is None:
                continue
            params.append(param)

        return params

    def _filter_source_sequences(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        filtered_sequences: list[torch.Tensor] = []

        if attention_mask is not None:
            for i in range(token_ids.shape[0]):
                seq = token_ids[i][attention_mask[i].bool()]
                if seq.numel() > 0:
                    filtered_sequences.append(seq)
                else:
                    self.failed_sequences += 1
            return filtered_sequences

        if self.source_pad_token_id is not None and self.source_pad_token_id not in [
            self.source_tokenizer.eos_token_id,
            self.source_tokenizer.bos_token_id,
        ]:
            non_pad_mask = token_ids != self.source_pad_token_id
            for i in range(token_ids.shape[0]):
                seq = token_ids[i][non_pad_mask[i]]
                if seq.numel() > 0:
                    filtered_sequences.append(seq)
                else:
                    self.failed_sequences += 1
            return filtered_sequences

        return [token_ids[i] for i in range(token_ids.shape[0])]

    def _decode_source_batch(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[str]:
        if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
            return []

        filtered_sequences = self._filter_source_sequences(token_ids, attention_mask)

        texts: list[str] = []
        for seq in filtered_sequences:
            if seq.numel() == 0:
                self.failed_sequences += 1
                continue

            text = self.source_tokenizer.decode(
                seq,
                skip_special_tokens=not self._keep_special_tokens,
            )

            if text and not text.isspace():
                texts.append(text)
            else:
                self.failed_sequences += 1

        return texts

    def _retokenize_for_reference_model(
        self,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not texts:
            return None

        enc = self.reference_tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=False,
        )

        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        if input_ids.size(0) == 0 or input_ids.size(1) < 2:
            self.failed_sequences += len(texts)
            return None

        return input_ids, attention_mask

    def _prepare_reference_batch(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        texts = self._decode_source_batch(token_ids, attention_mask)
        return self._retokenize_for_reference_model(texts)

    def _compute_batch_grads(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...] | None:
        """
        Returns gradients of mean NLL wrt selected reference-model parameters.

        This differs from grad(log p) by an overall sign, which cancels in the
        centered inner-product estimator.
        """
        if input_ids.size(1) < 2:
            return None

        original_requires_grad = [p.requires_grad for p in self._selected_params]
        for p in self._selected_params:
            p.requires_grad_(True)

        self.reference_model.eval()
        self.reference_model.zero_grad(set_to_none=True)

        try:
            device_type = self.device.type if self.device.type in ('cuda', 'cpu') else 'cpu'
            # Lightning validation/prediction commonly runs under `inference_mode`.
            # `torch.enable_grad()` does not override inference_mode, so we must
            # explicitly disable it. Also, if tensors were created under inference_mode,
            # rematerialize them here as normal tensors before building autograd graph.
            with torch.inference_mode(False):
                if hasattr(input_ids, 'is_inference') and input_ids.is_inference():
                    input_ids = input_ids.clone()
                if hasattr(attention_mask, 'is_inference') and attention_mask.is_inference():
                    attention_mask = attention_mask.clone()

                labels = input_ids.clone()
                labels[attention_mask.eq(0)] = -100

                forward_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }
                if self._forward_accepts_use_cache:
                    forward_kwargs['use_cache'] = False

                with torch.enable_grad():
                    with torch.autocast(device_type=device_type, enabled=False):
                        outputs = self.reference_model(**forward_kwargs)
                        loss = outputs.loss.float()
                if not torch.isfinite(loss):
                    return None

                grads = torch.autograd.grad(
                    loss,
                    self._selected_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            detached_grads: list[torch.Tensor | None] = []
            for grad in grads:
                if grad is None:
                    detached_grads.append(None)
                else:
                    detached_grads.append(grad.detach())

            return tuple(detached_grads)

        finally:
            self.reference_model.zero_grad(set_to_none=True)
            for p, req in zip(self._selected_params, original_requires_grad, strict=False):
                p.requires_grad_(req)

    def _clone_grads_to_storage(
        self,
        grads: tuple[torch.Tensor | None, ...],
    ) -> list[torch.Tensor | None]:
        stored: list[torch.Tensor | None] = []
        for grad in grads:
            if grad is None:
                stored.append(None)
            else:
                stored.append(grad.float().clone())
        return stored

    def _subtract_grads_in_place(
        self,
        lhs: list[torch.Tensor | None],
        rhs: tuple[torch.Tensor | None, ...],
    ) -> None:
        for i, grad_rhs in enumerate(rhs):
            if grad_rhs is None:
                continue

            grad_rhs = grad_rhs.float().to(lhs[i].device if lhs[i] is not None else grad_rhs.device)

            if lhs[i] is None:
                lhs[i] = -grad_rhs
            else:
                lhs[i].sub_(grad_rhs)

    def _dot_grad_lists(
        self,
        grads_a: list[torch.Tensor | None],
        grads_b: list[torch.Tensor | None],
    ) -> torch.Tensor:
        dot: torch.Tensor | None = None
        for ga, gb in zip(grads_a, grads_b, strict=False):
            if ga is None or gb is None:
                continue
            term = torch.sum(ga * gb)
            dot = term if dot is None else dot + term

        if dot is None:
            return torch.tensor(0.0, device=self.total_estimate.device, dtype=self.total_estimate.dtype)
        return dot

    def _compute_centered_gradient(
        self,
        generated_token_ids: torch.Tensor,
        reference_token_ids: torch.Tensor,
        generated_attention_mask: torch.Tensor | None = None,
        reference_attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor | None] | None:
        gen_batch = self._prepare_reference_batch(generated_token_ids, generated_attention_mask)
        ref_batch = self._prepare_reference_batch(reference_token_ids, reference_attention_mask)

        if gen_batch is None or ref_batch is None:
            return None

        gen_input_ids, gen_attn = gen_batch
        ref_input_ids, ref_attn = ref_batch

        gen_grads = self._compute_batch_grads(gen_input_ids, gen_attn)
        if gen_grads is None:
            return None

        centered = self._clone_grads_to_storage(gen_grads)
        del gen_grads

        ref_grads = self._compute_batch_grads(ref_input_ids, ref_attn)
        if ref_grads is None:
            return None

        self._subtract_grads_in_place(centered, ref_grads)
        del ref_grads

        return centered

    def update(
        self,
        generated_token_ids_a: torch.Tensor,
        reference_token_ids_a: torch.Tensor,
        generated_token_ids_b: torch.Tensor,
        reference_token_ids_b: torch.Tensor,
        generated_attention_mask_a: torch.Tensor | None = None,
        reference_attention_mask_a: torch.Tensor | None = None,
        generated_attention_mask_b: torch.Tensor | None = None,
        reference_attention_mask_b: torch.Tensor | None = None,
    ):
        """
        Update with two independent minibatch pairs.

        Args:
            generated_token_ids_a: Generated samples, pair A, in source-tokenizer ids.
            reference_token_ids_a: Real data samples, pair A, in source-tokenizer ids.
            generated_token_ids_b: Generated samples, pair B, in source-tokenizer ids.
            reference_token_ids_b: Real data samples, pair B, in source-tokenizer ids.
            *_attention_mask_*: Optional source-side attention masks.

        Returns:
            None. Accumulates one unbiased estimator sample.
        """
        if any(
            not isinstance(x, torch.Tensor) or x.numel() == 0
            for x in [
                generated_token_ids_a,
                reference_token_ids_a,
                generated_token_ids_b,
                reference_token_ids_b,
            ]
        ):
            self.failed_pairs += 1
            return

        self.total_sequences += (
            generated_token_ids_a.shape[0]
            + reference_token_ids_a.shape[0]
            + generated_token_ids_b.shape[0]
            + reference_token_ids_b.shape[0]
        )

        centered_a = self._compute_centered_gradient(
            generated_token_ids=generated_token_ids_a,
            reference_token_ids=reference_token_ids_a,
            generated_attention_mask=generated_attention_mask_a,
            reference_attention_mask=reference_attention_mask_a,
        )
        if centered_a is None:
            self.failed_pairs += 1
            return

        centered_b = self._compute_centered_gradient(
            generated_token_ids=generated_token_ids_b,
            reference_token_ids=reference_token_ids_b,
            generated_attention_mask=generated_attention_mask_b,
            reference_attention_mask=reference_attention_mask_b,
        )
        if centered_b is None:
            self.failed_pairs += 1
            return

        estimate = self._dot_grad_lists(centered_a, centered_b)

        if torch.isfinite(estimate):
            self.total_estimate += estimate.to(self.total_estimate.device, dtype=self.total_estimate.dtype)
            self.total_pairs += 1
        else:
            self.failed_pairs += 1

    def compute(self) -> torch.Tensor:
        if self.total_pairs == 0:
            return self._get_empty_result()
        return self.total_estimate / self.total_pairs.clamp(min=1)

    def _get_empty_result(self) -> torch.Tensor:
        return torch.tensor(float('nan'), device=self.device)
