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


def wt_detokenizer(text: str) -> str:
    # contractions
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)

    # number separators
    text = text.replace(' @-@ ', '-')
    text = text.replace(' @,@ ', ',')
    text = text.replace(' @.@ ', '.')

    # punctuation
    text = text.replace(' : ', ': ')
    text = text.replace(' ; ', '; ')
    text = text.replace(' . ', '. ')
    text = text.replace(' ! ', '! ')
    text = text.replace(' ? ', '? ')
    text = text.replace(' , ', ', ')

    # double brackets
    text = re.sub(r'\(\s*([^\)]*?)\s*\)', r'(\1)', text)
    text = re.sub(r'\[\s*([^\]]*?)\s*\]', r'[\1]', text)
    text = re.sub(r'{\s*([^}]*?)\s*}', r'{\1}', text)
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)

    # miscellaneous
    text = text.replace('= = = =', '====')
    text = text.replace('= = =', '===')
    text = text.replace('= =', '==')
    text = text.replace(f' {chr(176)} ', chr(176))
    text = text.replace(' \n', '\n')
    text = text.replace('\n ', '\n')
    text = text.replace(' N ', ' 1 ')
    return text.replace(" 's", "'s")


def ptb_detokenizer(text: str) -> str:
    text = text.replace(" 's", "'s")
    text = text.replace("s ' ", "s' ")
    text = text.replace(" n't", "n't")
    text = text.replace(' \n ', '\n')
    text = text.replace('\\/', '/')
    for _ in range(10):
        text = text.replace(' N ', ' 1 ')
    text = text.replace('$ 1', '$1')
    text = text.replace('# 1', '#1')
    return text.replace('<unk>', '?')


def lambada_detokenizer(text: str) -> str:
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return f'\n{text.strip()}'


_DETOKENIZERS = {
    'lm1b': lm1b_detokenizer,
    'one_billion_word': lm1b_detokenizer,
    'one_billion_words': lm1b_detokenizer,
    'wikitext': wt_detokenizer,
    'wt': wt_detokenizer,
    'ptb': ptb_detokenizer,
    'penn_treebank': ptb_detokenizer,
    'lambada': lambada_detokenizer,
}


def get_detokenizer(name: str):
    key = name.strip().lower()
    if key not in _DETOKENIZERS:
        available = ', '.join(sorted(_DETOKENIZERS))
        raise ValueError(f'Unknown detokenizer "{name}". Available detokenizers: {available}')
    return _DETOKENIZERS[key]
