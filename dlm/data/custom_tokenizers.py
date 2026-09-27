import re

import torch


class SimpleTokenizer:
    """Minimal tokenizer that just handles vocab mapping for our simple case."""

    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Simple decode - just convert token IDs to string representation"""
        # token_ids = [t for t in token_ids if t not in [self.mask_token_id]]
        return ' '.join([str(t.item() if hasattr(t, 'item') else t) for t in token_ids])

    def batch_decode(self, token_ids_batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch]


class AdditionTokenizer:
    """Tokenizer for addition dataset compatible with CustomDataModule."""

    def __init__(
        self,
        vocab_size: int = 15,
        mask_token_id: int | None = None,
        pad_token: str = '[PAD]',
        eos_token: str = '[EOS]',
        mask_token: str = '[MASK]',
    ):
        # Define vocabulary: 0-9, +, =, [PAD], [EOS], [MASK]
        self.vocab = [str(i) for i in range(10)] + ['+', '=', pad_token, eos_token, mask_token]
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size

        self.pad_token = pad_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        # Ensure vocab_size matches actual vocabulary
        if len(self.vocab) != vocab_size:
            print(f"Warning: vocab_size ({vocab_size}) doesn't match actual vocab length ({len(self.vocab)})")

    def encode(self, text: str) -> list[int]:
        """Encode text string to token IDs."""
        if isinstance(text, torch.Tensor):
            # Already encoded
            return text

        token_ids = []
        for char in text:
            if char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                raise ValueError(f'Unknown character: {char}')
        return token_ids

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.token_to_id[self.pad_token], self.mask_token_id]:
                continue
            if token_id in self.id_to_token:
                chars.append(self.id_to_token[token_id])

        return ''.join(chars)

    def batch_decode(
        self,
        token_ids_batch: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Batch decode token IDs back to text strings."""
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()

        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch]


class naive_tokenizer:
    """character-level (legacy - kept for compatibility)"""

    def __init__(
        self,
        number_bits,
        pad_token='[PAD]',
        eos_token='[EOS]',
        mask_token='[MASK]',
    ):
        self.vocab = [str(x) for x in range(10)] + ['=', '+'] + [pad_token, eos_token, mask_token]
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)
        self.pattern = f'[^{re.escape("".join(self.vocab))}]'
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.masking_index = 2 * number_bits + 2

    def clean(self, text):
        """Removes all characters not in the vocabulary"""
        out = re.sub(self.pattern, '', text)
        return out

    def pre_tokenization(self, text):
        """character-level"""
        return [c for c in text]

    def encode(self, text):
        text_list = self.pre_tokenization(self.clean(text))
        return [self.token_to_id[c] for c in text_list]

    def decode(self, token_list):
        return ''.join([self.id_to_token[x] for x in token_list])
