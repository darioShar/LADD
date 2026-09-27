import random

import torch


class BinarySawtoothLatent:
    """Generate a random shift for each sequence.

    The shift is a random number in [0, 1] for each sequence.
    """

    def __init__(self, seed: None | int):
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

    def __call__(self, num_samples: int) -> torch.Tensor:
        return torch.rand(num_samples, 1)


def sawtooth_wave(t: torch.Tensor, num_saws: int = 2, scale: float = 0.05) -> torch.Tensor:
    """Create a sawtooth wave."""
    x = (t * num_saws) % 1.0
    triangle_wave = 1 - torch.abs(2 * x - 1)
    p = scale + (1 - 2 * scale) * triangle_wave
    return p


def binary_sawtooth_probs(num_samples, sequence_length, shift, num_saws, scale):
    t_base = torch.arange(0, sequence_length, dtype=torch.float32, device=shift.device) / sequence_length
    t_base = t_base.unsqueeze(0).repeat(num_samples, 1)
    t = (t_base + shift) % 1.0
    x = (t * num_saws) % 1.0
    triangle_wave = 1 - torch.abs(2 * x - 1)
    p = scale + (1 - 2 * scale) * triangle_wave
    return p

def binary_sawtooth(
    num_samples: int = 1000,
    sequence_length: int = 64,
    vocab_size: int = 2,
    num_saws: int = 2,
    scale: float = 0.01,
    uniform_shift: bool = True,
    seed: None | int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic binary sequences based on a sawtooth/triangle wave.

    Each sequence is sampled from a Bernoulli distribution where the probability
    is determined by a triangle wave.

    The triangle wave `f(t)` for `t` in `[0,1]` has `num_saws` periods,
    a minimum value of `scale` and a maximum of `1-scale`.

    For each sequence, a random phase shift can be applied to the wave.

    Args:
        num_samples: Number of sequences to generate
        sequence_length: Length of each sequence
        vocab_size: Ignored. The output is always binary (0 or 1).
        num_saws: Number of sawtooth periods in the sequence.
        scale: The minimum value of the sawtooth wave probability. The maximum is 1-scale. Must be in [0, 0.5].
        uniform_shift: If True, apply a random phase shift to each sequence.
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sequences, latents) where:
        - sequences: Tensor of shape [num_samples, sequence_length] with binary token indices (0 or 1)
        - latents: Tensor of shape [num_samples, 1] with the shift values used for each sequence (the 'y' parameter)

    """
    if seed is not None:
        torch.manual_seed(seed)

    # assert vocab_size == 2
    if vocab_size != 2:
        msg = f'vocab_size must be 2, but got {vocab_size}'
        raise ValueError(msg)

    if not (0 <= scale <= 0.5):
        msg = f'scale must be in [0, 0.5], but got {scale}'
        raise ValueError(msg)

    # Time steps for a single sequence, from 0 up to (1 - 1/sequence_length)
    latent_generator = BinarySawtoothLatent(seed=seed)
    shift = latent_generator(num_samples) if uniform_shift else torch.zeros(num_samples, 1)
    p = binary_sawtooth_probs(num_samples, sequence_length, shift, num_saws, scale)
    all_sequences = torch.bernoulli(p).long()  # Convert to long integers for token IDs

    # Return sequences and latents (keep shift as [num_samples, 1] for latent dimension)
    return all_sequences, shift


def addition_dataset(
    num_samples: int = 1000,
    sequence_length: int = 64,
    vocab_size: int = 15,  # 0-9, +, =, [PAD], [EOS], [MASK]
    number_bits: int = 3,
    seed: int | None = 42,
) -> torch.Tensor:
    """Generate addition dataset compatible with SimpleDiscreteDataset.

    Creates sequences in the format: "123+456=579[EOS][PAD][PAD]..."
    Each sequence is padded to sequence_length.

    Args:
        num_samples: Number of addition problems to generate
        sequence_length: Fixed length for all sequences
        vocab_size: Size of vocabulary (should be at least 15 for digits, +, =, special tokens)
        number_bits: Number of digits in each operand
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape [num_samples, sequence_length] with token indices

    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Define vocabulary mapping
    # 0-9: digits, 10: '+', 11: '=', 12: '[PAD]', 13: '[EOS]', 14: '[MASK]'
    digit_to_id = {str(i): i for i in range(10)}
    special_tokens = {'+': 10, '=': 11, '[PAD]': 12, '[EOS]': 13, '[MASK]': 14}
    token_to_id = {**digit_to_id, **special_tokens}

    # Validate vocab_size
    required_vocab_size = len(token_to_id)
    if vocab_size < required_vocab_size:
        raise ValueError(f'vocab_size ({vocab_size}) must be at least {required_vocab_size}')

    sequences = []

    for _ in range(num_samples):
        # Generate two random numbers with specified number of bits
        a = random.randint(10 ** (number_bits - 1), 10**number_bits - 1)
        b = random.randint(10 ** (number_bits - 1), 10**number_bits - 1)
        sum_result = a + b

        # Create the equation string: "a+b=sum"
        equation = f'{a}+{b}={sum_result}'

        # Convert to token indices
        token_ids = []
        for char in equation:
            if char in token_to_id:
                token_ids.append(token_to_id[char])
            else:
                raise ValueError(f'Unknown character: {char}')

        # Add EOS token
        token_ids.append(token_to_id['[EOS]'])

        # Pad to sequence_length
        if len(token_ids) > sequence_length:
            raise ValueError(f"Equation '{equation}' is too long for sequence_length {sequence_length}")

        # Pad with PAD tokens
        while len(token_ids) < sequence_length:
            token_ids.append(token_to_id['[PAD]'])

        sequences.append(token_ids)

    return torch.tensor(sequences, dtype=torch.long)
