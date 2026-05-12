# dlm/callbacks/wandb_callbacks.py
import lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class SaveWandbIDCallback(Callback):
    """Saves the wandb run ID to a file in the logdir."""

    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Find the WandbLogger
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break

        if wandb_logger is None:
            print('Warning: WandbLogger not found. Could not save run ID.')
            return

        # Get the ID and save it
        run_id = wandb_logger.experiment.id
        if trainer.logdir and run_id:
            id_file_path = f'{trainer.logdir}/wandb_id.txt'
            with open(id_file_path, 'w') as f:
                f.write(run_id)
            print(f"Saved W&B run ID '{run_id}' to {id_file_path}")
