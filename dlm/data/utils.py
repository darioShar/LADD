import re

import torch


def pad_sequence(
    inputs,
    pad_id: int,
    max_length: int,
    padding_side: str = "right",
) -> torch.Tensor:
    assert padding_side in ["right", "left"]
    if padding_side == "right":
        data = [d[:max_length] + [pad_id] * (max_length - len(d)) for d in inputs]
    else:
        data = [[pad_id] * (max_length - len(d)) + d[:max_length] for d in inputs]
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)
    return data


# Text Regularization
# copied from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/data.py#L70
def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x
