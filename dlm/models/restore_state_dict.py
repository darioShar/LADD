import lightning as L
import torch
import torchmetrics
from torch import nn
from ..utils import print_rank_zero





def load_weights_agnostic(model: L.LightningModule, checkpoint: dict) -> None:
    """
    Loads weights from a checkpoint dictionary into a LightningModule in an agnostic way.

    Args:
        model: The instantiated LightningModule into which weights will be loaded.
        checkpoint: The loaded checkpoint dictionary.
    """
    print_rank_zero('⚡ Attempting to agnostically load weights from checkpoint dictionary...')

    full_state_dict: dict[str, torch.Tensor] = checkpoint.get('state_dict')
    if not full_state_dict:
        print_rank_zero("⚠️ Checkpoint does not contain a 'state_dict'. No weights will be loaded.")
        return

    # 2. Identify all nn.Module components (children) of the LightningModule
    # model.named_children() is the key to making this agnostic
    # explicitly IGNORE torchmetrics
    components = {name: module for name, module in model.named_children() if not isinstance(module, torchmetrics.Metric)}
    if not components:
        print_rank_zero('⚠️ No `nn.Module` children found in the LightningModule. Cannot load weights.')
        return

    print_rank_zero(f'Found model components: {list(components.keys())}')

    loaded_components = []
    # 3. Iterate over each component and attempt to load its weights
    for name, module in components.items():
        # The prefix in the checkpoint will be the attribute name, e.g., "encoder."
        prefix = f'{name}.'

        # Create a state_dict for the current component by filtering and stripping the prefix
        component_state_dict = {k.removeprefix(prefix): v for k, v in full_state_dict.items() if k.startswith(prefix)}

        if not component_state_dict:
            print_rank_zero(f"  - No weights found for component '{name}' in checkpoint. Skipping.")
            continue

        # Load the weights into the component module
        missing_keys, unexpected_keys = module.load_state_dict(component_state_dict, strict=False)

        # Report the status for this specific component
        if not missing_keys and not unexpected_keys:
            print_rank_zero(f"  ✅ Successfully loaded all weights for component '{name}'.")
        else:
            print_rank_zero(f"  - Partially loaded weights for component '{name}':")
            if missing_keys:
                print_rank_zero(f'    - Missing keys: {missing_keys[:20]}...')
            if unexpected_keys:
                print_rank_zero(f'    - Unexpected keys: {unexpected_keys[:20]}...')
        loaded_components.append(name)

    print_rank_zero('-' * 40)
    if loaded_components:
        print_rank_zero(f'✅ Finished loading weights for: {loaded_components}')
    else:
        print_rank_zero('⚠️ No weights were loaded for any component.')


def _make_current_param_groups_layout(opt: torch.optim.Optimizer):
    """Create a param_groups list compatible with opt.load_state_dict mapping
    using sequential param indices across all current groups."""
    new_param_groups = []
    base = 0
    for g in opt.param_groups:
        # copy hyperparams except 'params'
        gcopy = {k: v for k, v in g.items() if k != 'params'}
        n = len(g['params'])
        gcopy['params'] = list(range(base, base + n))  # global sequential indices
        new_param_groups.append(gcopy)
        base += n
    return new_param_groups


def load_optimizer_group_state_only(
    opt: torch.optim.Optimizer,
    ckpt_opt_state: dict,
    src_group_idx: int = 0,
    dst_group_idx: int = 0,
    strict_len: bool = True,
) -> None:
    """
    Load AdamW moments for a single destination param group (dst_group_idx)
    from a single source group in the checkpoint (src_group_idx).

    Assumes the *order* of params inside the destination group matches how
    they were ordered in the checkpoint’s source group.
    """
    # Current layout
    cur_groups = opt.param_groups
    assert 0 <= dst_group_idx < len(cur_groups), 'dst_group_idx out of range'

    # Build compatible param_groups structure for the *current* optimizer
    new_param_groups = _make_current_param_groups_layout(opt)

    # How many params are in the current dst group?
    dst_len = len(cur_groups[dst_group_idx]['params'])

    # Get source ids from the checkpoint for the chosen group
    try:
        src_pg = ckpt_opt_state['param_groups'][src_group_idx]
    except Exception as e:
        raise ValueError(f'Bad checkpoint format for optimizer: {e}')

    src_ids = list(src_pg['params'])
    if strict_len and len(src_ids) != dst_len:
        raise ValueError(
            f'Param count mismatch for group load: src_len={len(src_ids)} vs dst_len={dst_len} '
            f'(disable strict_len to allow min-overlap).',
        )

    # Map states: keys are *global param indices* in the state dict you pass to load_state_dict.
    # For dst_group_idx, its global index range starts at the sum of sizes of all previous groups.
    offset = sum(len(g['params']) for g in cur_groups[:dst_group_idx])
    overlap = min(len(src_ids), dst_len)

    new_state = {}
    for i in range(overlap):
        src_id = src_ids[i]
        st = ckpt_opt_state['state'].get(src_id)
        if st is None:
            continue
        new_state[offset + i] = st  # map onto current optimizer’s global index

    # Other groups will have empty state (AdamW will initialize on first step)
    sliced = {'state': new_state, 'param_groups': new_param_groups}
    opt.load_state_dict(sliced)
    print_rank_zero(
        f'✅ Partially restored optimizer state: {overlap}/{dst_len} params for group {dst_group_idx} '
        f'from checkpoint group {src_group_idx}.',
    )


def restore_optim_from_checkpoint(trainer: L.LightningModule, checkpoint: dict[str, nn.Module]) -> bool:
    # 1) OPTIMIZERS
    all_passed = True
    if hasattr(trainer, 'optimizers') and trainer.optimizers and 'optimizer_states' in checkpoint:
        opt_states = checkpoint['optimizer_states']
        for i, (opt, state) in enumerate(zip(trainer.optimizers, opt_states, strict=False)):
            try:
                opt.load_state_dict(state)  # works only if param groups match
                print_rank_zero(f'✅ Restored optimizer[{i}] state.')
            except Exception as e:
                print_rank_zero(f'⚠️ Could not restore optimizer[{i}] state: {e}')
                # Fallback: try to partially restore dst_group 0 from src_group 0
                try:
                    load_optimizer_group_state_only(opt, state, src_group_idx=0, dst_group_idx=0, strict_len=False)
                    print_rank_zero(f'🟡 Partially restored optimizer[{i}] group 0 moments from ckpt.')
                except Exception as e2:
                    print_rank_zero(f'❌ Partial restore failed for optimizer[{i}]: {e2}')
                    all_passed = False
    return all_passed


def restore_lr_from_checkpoint(trainer: L.LightningModule, checkpoint: dict[str, nn.Module]) -> bool:
    # 2) LR SCHEDULERS
    # PL>=2 stores configs wrapping the scheduler; handle both dict & config objects.
    sched_confs = getattr(trainer, 'lr_scheduler_configs', None) or getattr(trainer, 'lr_schedulers', None)
    all_passed = True
    if sched_confs:
        # Lightning’s checkpoint formats vary by version; try several keys.
        ckpt_sched_states = (
            checkpoint.get('lr_scheduler_states') or checkpoint.get('lr_schedulers') or checkpoint.get('lr_schedulers_state')
        )
        if ckpt_sched_states:
            for i, (conf, st) in enumerate(zip(sched_confs, ckpt_sched_states, strict=False)):
                sch = getattr(conf, 'scheduler', None) or (conf.get('scheduler') if isinstance(conf, dict) else None) or conf
                try:
                    # Some PL versions store {"state_dict": ...}
                    sch.load_state_dict(st.get('state_dict', st))
                    print_rank_zero(f'✅ Restored lr_scheduler[{i}] state.')
                    print_rank_zero(f'Type of loaded scheduler: {type(sch)}')
                except Exception as e:
                    print_rank_zero(f'⚠️ Could not restore lr_scheduler[{i}] state: {e}')
                    all_passed = False
        # else:
        #     # Fallback: fast-forward to the original global_step
        #     gs = int(checkpoint.get('global_step', 0))
        #     for i, conf in enumerate(sched_confs):
        #         sch = getattr(conf, 'scheduler', None) or (conf.get('scheduler') if isinstance(conf, dict) else None) or conf
        #         try:
        #             if hasattr(sch, 'last_epoch'):
        #                 sch.last_epoch = gs - 1
        #                 sch._step_count = gs
        #                 sch.step()
        #                 print_rank_zero(f'ℹ️ Set lr_scheduler[{i}].last_epoch to {gs}.')
        #         except Exception as e:
        #             print_rank_zero(f'⚠️ Failed to fast-forward lr_scheduler[{i}]: {e}')
    return all_passed


class _ReentryLRController:
    def __init__(self, zero_steps: int = 2000, warm_steps: int = 2000, target_scale: float = 1.0):
        self.zero_steps = zero_steps
        self.warm_steps = warm_steps
        self.total = zero_steps + warm_steps
        self.target_scale = target_scale
        self.base_lrs = None  # filled at start

    def lr_at(self, step: int, base_lr: float) -> float:
        if step < self.zero_steps:
            return 0.0
        if step < self.total:
            t = (step - self.zero_steps) / max(1, self.warm_steps)
            return base_lr * self.target_scale * t
        return base_lr * self.target_scale  # hand back control to normal scheduler after total


def _get_all_param_groups(trainer):
    pgs = []
    for opt in getattr(trainer, 'optimizers', []):
        pgs.extend(opt.param_groups)
    return pgs
