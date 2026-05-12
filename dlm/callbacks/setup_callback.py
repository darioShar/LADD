import os
import traceback

import lightning as L
from omegaconf import OmegaConf

from ..utils import print_rank_zero

ENABLE_MULTINODE_STARTUP_DELAY = True


# copied from https://github.com/Stability-AI/generative-models/blob/main/main.py#L230
class SetupCallback(L.Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: L.Trainer, pl_module, exception):
        # get trainer state, either fit, validate or predict
        if hasattr(pl_module, '_ema_activation_nested_level'):
            pl_module._ema_activation_nested_level = 0
        print_rank_zero(f'Unhandled exception during {trainer.state.stage}: {exception}')
        tb_text = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        print_rank_zero(tb_text)
        if (not self.debug) and (trainer.global_rank == 0) and (trainer.state.stage in ['train', 'validate']):
            print_rank_zero('Summoning checkpoint.')
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, 'last.ckpt')
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            print_rank_zero('Saving checkpoint to:', ckpt_path)
            trainer.save_checkpoint(ckpt_path)
            print_rank_zero('Done')

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if ('callbacks' in self.lightning_config) and (
                'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']
            ):
                os.makedirs(
                    os.path.join(self.ckptdir, 'trainstep_checkpoints'),
                    exist_ok=True,
                )
            if ENABLE_MULTINODE_STARTUP_DELAY:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, f'{self.now}-project.yaml'),
            )

            OmegaConf.save(
                OmegaConf.create({'lightning': self.lightning_config}),
                os.path.join(self.cfgdir, f'{self.now}-lightning.yaml'),
            )

        # ModelCheckpoint callback created log directory --- remove it
        elif not ENABLE_MULTINODE_STARTUP_DELAY and not self.resume and os.path.exists(self.logdir):
            dst, name = os.path.split(self.logdir)
            dst = os.path.join(dst, 'child_runs', name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            try:
                os.rename(self.logdir, dst)
            except FileNotFoundError:
                pass
