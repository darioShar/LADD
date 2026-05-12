from pathlib import Path

import lightning as L
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only


class SavePredictionCallback(L.Callback):
    """Callback to collect and save prediction samples from all GPUs."""

    def __init__(self, generated_samples_dir: str):
        """
        Args:
            generated_samples_dir: Path to the directory where samples will be saved.
        """
        super().__init__()
        self.generated_samples_dir = Path(generated_samples_dir)

    @rank_zero_only
    def _save_samples(self, samples, trainer: L.Trainer):
        """Save the generated samples to the specified directory."""
        # Extract run name from trainer logdir
        if hasattr(trainer, 'logdir') and trainer.logdir:
            run_name = '/'.join(trainer.logdir.split('/')[1:])
        else:
            run_name = 'unknown_run'

        # Create save path
        save_dir = self.generated_samples_dir / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'generated_samples_epoch_{trainer.current_epoch}_step_{trainer.global_step}.pt'
        save_path = save_dir / filename

        # Filter out empty lists and move tensors to CPU
        filtered_samples = {k: v for k, v in samples.items() if v}
        if not filtered_samples:
            return

        samples_on_cpu = {k: [s.cpu() if hasattr(s, 'cpu') else s for s in v] for k, v in filtered_samples.items()}

        # Try to concatenate batches into single tensors
        try:
            all_samples = {k: torch.cat(v, dim=0) for k, v in samples_on_cpu.items()}
        except (TypeError, RuntimeError):
            # If concatenation fails, try converting lists to tensors first
            converted_samples = {}
            for k, v in samples_on_cpu.items():
                converted_batch = [torch.tensor(batch) if isinstance(batch, list) else batch for batch in v]
                converted_samples[k] = converted_batch

            try:
                all_samples = {k: torch.cat(v, dim=0) for k, v in converted_samples.items()}
            except (TypeError, RuntimeError):
                # If still failing, save as original format
                all_samples = samples_on_cpu

        # Save the samples
        torch.save(all_samples, save_path)
        print(f'Samples saved to: {save_path}')

    def on_predict_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of the prediction epoch."""
        # Only run on rank 0 in distributed training
        if trainer.global_rank != 0:
            return

        # Get predictions from trainer
        if not hasattr(trainer, 'predict_loop') or not hasattr(trainer.predict_loop, 'predictions'):
            return

        predictions = trainer.predict_loop.predictions
        if not predictions or not predictions[0]:
            return

        # Extract samples from prediction outputs
        all_batch_outputs = predictions[0]
        keys_to_log = ['output_ids', 'y_outputs']
        extracted_samples = {}

        # Handle both single output and list of outputs
        is_iterable = hasattr(all_batch_outputs, '__iter__') and not isinstance(
            all_batch_outputs,
            (torch.Tensor, str),
        )
        batch_outputs = all_batch_outputs if is_iterable else [all_batch_outputs]

        for batch_output in batch_outputs:
            for key in keys_to_log:
                if hasattr(batch_output, key):
                    if key not in extracted_samples:
                        extracted_samples[key] = []
                    extracted_samples[key].append(getattr(batch_output, key))

        self._save_samples(extracted_samples, trainer)
