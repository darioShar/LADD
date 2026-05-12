import os

from accelerate import init_empty_weights
from transformers import PreTrainedModel

from lightning.pytorch.strategies import FSDPStrategy

from ..models.lm import LM
from ..utils import get_obj_from_str, instantiate_model_from_config


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    if len(modules_children) == 0:
        return None
    for child_module in modules_children:
        module_class = get_module_class_from_name(child_module, name)
        if module_class is not None:
            return module_class


def get_split_modules(model):
    return ','.join(model._no_split_modules) if getattr(model, '_no_split_modules', None) is not None else ''


@init_empty_weights()
def load_empty_model(config):
    return instantiate_model_from_config(config)


def fsdp_huggingface(model, transformer_cls_to_wrap=None, **kwargs):
    if transformer_cls_to_wrap is None:
        if isinstance(model, LM):
            models = [
                load_empty_model(model.hparams['model_config']),
            ]
        elif isinstance(model, PreTrainedModel):
            models = [model]
        else:
            raise NotImplementedError
        default_transformer_cls_names_to_wrap = ','.join(
            [get_split_modules(model) for model in models],
        )

        transformer_cls_names_to_wrap = os.environ.get(
            'FSDP_TRANSFORMER_CLS_TO_WRAP',
            default_transformer_cls_names_to_wrap,
        ).split(',')
        print(f'FSDP_TRANSFORMER_CLS_TO_WRAP: {transformer_cls_names_to_wrap}')
        transformer_cls_to_wrap = set()
        for layer_class in transformer_cls_names_to_wrap:
            for model in models:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is not None:
                    break
            if transformer_cls is None:
                raise Exception(
                    'Could not find the transformer layer class to wrap in the model.',
                )
            transformer_cls_to_wrap.add(transformer_cls)
        del models
    else:
        if isinstance(transformer_cls_to_wrap, str):
            transformer_cls_to_wrap = [transformer_cls_to_wrap]
        transformer_cls_to_wrap = set(
            [get_obj_from_str(s) for s in transformer_cls_to_wrap],
        )
    auto_wrap_policy = transformer_cls_to_wrap
    return FSDPStrategy(auto_wrap_policy=auto_wrap_policy, **kwargs)
