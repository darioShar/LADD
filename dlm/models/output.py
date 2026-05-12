from dataclasses import dataclass

import torch


@dataclass
class SampleOutput:
    """Output of the model sampler.

    This includes the prompts, the completions, and the output ids.
    """

    def __init__(self, prompts: list[str] | None, completions: list[str] | None, output_ids: torch.Tensor | list[int]):
        self.prompts = prompts
        self.completions = completions
        self.output_ids = output_ids


@dataclass
class LatentDDMSampleOutput(SampleOutput):
    """Output of the LatentDDM sampler.

    This includes the output token ids, the prompts, the completions, and the
    denoised latent vectors.
    """

    def __init__(
        self,
        output_ids: torch.Tensor,
        prompts: list[str] | None,
        completions: list[str] | None,
        y_outputs: torch.Tensor,
    ) -> None:
        """Initialize the LatentDDMSampleOutput.

        Args:
            output_ids: The output token ids.
            prompts: The prompts.
            completions: The completions.
            y_outputs: The denoised latent vectors.

        """
        super().__init__(prompts, completions, output_ids)
        # Store the denoised latent vectors
        self.y_outputs = y_outputs
