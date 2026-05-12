"""These processors are based on HuggingFace LogitsWarper but modified to
- Input only `scores` instead of `input_ids` and `scores`
- Handle sequence-wise logits, whose shape is (bsz, seq_len, vocab_size). The original processor assumes (bsz, vocab_size)
"""

import inspect

import torch
from transformers import (
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

from ..utils import instantiate_from_config


class LogitsProcessorList(list):
    def __call__(self, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f'Make sure that all the required parameters: {list(function_args.keys())} for '
                        f'{processor.__class__} are passed to the logits processor.',
                    )
                scores = processor(scores, **kwargs)
            else:
                scores = processor(scores)
        return scores


class Temperature(TemperatureLogitsWarper):
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        return scores_processed


class TopP(TopPLogitsWarper):
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1,
            sorted_indices,
            sorted_indices_to_remove,
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores


class TopK(TopKLogitsWarper):
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class Typical(TypicalLogitsWarper):
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=-1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
            -1,
            last_ind.unsqueeze(-1),
        )
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1,
            sorted_indices,
            sorted_indices_to_remove,
        )
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


def get_logit_processors(
    logit_processors: dict | list[dict],
) -> LogitsProcessorList:
    if isinstance(logit_processors, dict):
        logit_processors = [logit_processors]
    warpers = LogitsProcessorList()
    for proc_config in logit_processors:
        processor = instantiate_from_config(proc_config)
        assert isinstance(processor, LogitsProcessor), 'logit_processor must be `LogitsProcessor`!'
        warpers.append(processor)
    return warpers
