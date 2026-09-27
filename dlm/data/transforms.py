from typing import Any
import copy
import logging

from itertools import chain

import torch
import transformers

transformers.utils.logging.get_logger("transformers.tokenization_utils_base").setLevel(
    logging.ERROR
)


class BaseTransform:
    """
    tokenize text
    """

    IGNORE_INDEX = -100

    def __init__(self, tokenizer, tokenize_kwargs: dict | None = None):
        self.tokenizer = tokenizer
        # default padding side is right, but respect existing setting if already configured
        if not hasattr(self.tokenizer, 'padding_side') or self.tokenizer.padding_side is None:
            self.tokenizer.padding_side = "right"
        if tokenize_kwargs is None:
            tokenize_kwargs = {"padding": True}
        self.tokenize_kwargs = tokenize_kwargs

    def __call__(self, example: dict[str, Any]):
        if self.tokenizer is None:
            return example
        encodings = self.tokenizer(
            example["text"],
            return_tensors="pt",
            **self.tokenize_kwargs,
        )
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.IGNORE_INDEX
        encodings["labels"] = labels
        return encodings


ROLE_KEY_MAPPING = {"human": "user", "gpt": "assistant", "system": "system"}


class TransformForSFT(BaseTransform):
    def __init__(
        self,
        messages_field: str | None = None,
        system_prompt: str | None = None,
        verbose: bool = False,
        supervised_setting: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.messages_field = messages_field
        self.system_prompt = system_prompt
        self.printed_sample = not verbose
        self.supervised_setting = supervised_setting

    def __call__(self, example: dict[str, Any], **kwargs):
        if self.messages_field is None:
            columns = [
                col for col in example.keys() if col in ["conversations", "messages"]
            ]
            assert len(columns) > 0, (
                f"Didn't find messages field in the columns {list(example.keys())}! Please specify with `messages_field`!"
            )
            self.messages_field = columns[0]
        if "content" in example[self.messages_field][0][0]:
            batch_messages: list[list[dict]] = example[self.messages_field]
        else:
            # formatting
            batch_messages = []
            for conversation in example[self.messages_field]:
                messages = []
                for message in conversation:
                    messages.append(
                        {
                            "role": ROLE_KEY_MAPPING[message["from"]],
                            "content": message["value"],
                        }
                    )
                batch_messages.append(messages)
        texts = []
        for conversation in batch_messages:
            if self.system_prompt is not None:
                if conversation[0]["role"] == "system":
                    raise ValueError(
                        "System prompt is already provided in the conversation!"
                    )
                conversation = [
                    {"role": "system", "content": self.system_prompt}
                ] + conversation
            texts.append(
                self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )

        example["text"] = texts
        if not self.printed_sample:
            print("=" * 10, "Sample", "=" * 10)
            print(example["text"][0])
            self.printed_sample = True
        encodings = super().__call__(example, **kwargs)
        bsz = encodings["input_ids"].size(0)
        if self.supervised_setting:
            assert self.tokenize_kwargs.get("padding_side", "right") == "right", (
                "padding_side must be right"
            )

            target = encodings["input_ids"].clone()
            for i in range(bsz):
                conv = batch_messages[i]
                temp_conv = []
                start_idx = 0
                for j in range(len(conv)):
                    temp_conv.append(conv[j])
                    temp_text = self.tokenizer.apply_chat_template(
                        temp_conv, tokenize=False, add_generation_prompt=False
                    )
                    prompt_len = self.tokenizer(
                        temp_text, return_tensors="pt"
                    ).input_ids.size(1)
                    if conv[j]["role"] == "user":
                        end_idx = prompt_len
                        target[i, start_idx:end_idx] = self.IGNORE_INDEX
                    else:
                        start_idx = prompt_len
            encodings["labels"] = target
        # append prompt_ids
        prompts = []
        for i in range(bsz):
            conv = batch_messages[i]
            # assume the last message is the assistant message
            assert conv[-1]["role"] == "assistant", (
                "The last message must be the assistant message"
            )
            if self.system_prompt is not None:
                conv = [{"role": "system", "content": self.system_prompt}] + conv
            prompts.append(
                self.tokenizer.apply_chat_template(
                    conv[:-1], tokenize=False, add_generation_prompt=False
                )
            )
        encodings["prompt_ids"] = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).input_ids
        return encodings


class TransformForPT(BaseTransform):
    """
    Append bos and eos tokens to the input text.
    """

    def __init__(
        self,
        text_field: str = "text",
        drop_last: bool = True,
        **kwargs,
    ):
        """
        Args:
            text_field (str, optional): The field in the example that contains the text to tokenize. Defaults to "text".
            drop_last (bool, optional): Whether to drop the last incomplete chunk after grouping. Defaults to True.
        """
        super().__init__(**kwargs)
        self.text_field = text_field
        self.drop_last = drop_last

    def __call__(self, example: dict[str, Any], **kwargs):
        # tokenize the input text
        max_length = self.tokenize_kwargs.get(
            "max_length", self.tokenizer.model_max_length
        )
        input_ids: list[list[int]] = self.tokenizer(
            example[self.text_field],
            add_special_tokens=False,
        ).input_ids
        input_ids = [i + [self.tokenizer.eos_token_id] for i in input_ids]

        # group texts
        flat_input_ids = list(chain(*input_ids))
        total_length = len(flat_input_ids)
        block_size = max_length - 2
        all_pack_size = (total_length // block_size) * block_size
        input_ids = [
            [self.tokenizer.bos_token_id]
            + flat_input_ids[i : i + block_size]
            + [self.tokenizer.eos_token_id]
            for i in range(0, all_pack_size, block_size)
        ]
        labels = copy.deepcopy(input_ids)
        # if the last part remains, pad it
        remaining_size = total_length - all_pack_size
        if remaining_size > 0 and not self.drop_last:
            input_ids.append(
                [self.tokenizer.bos_token_id]
                + flat_input_ids[-remaining_size:]
                + [self.tokenizer.eos_token_id]
                + [self.tokenizer.pad_token_id] * (block_size - remaining_size)
            )
            labels.append(
                [self.tokenizer.bos_token_id]
                + flat_input_ids[-remaining_size:]
                + [self.tokenizer.eos_token_id]
                + [self.IGNORE_INDEX] * (block_size - remaining_size)
            )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(labels, dtype=torch.long)
        encodings = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return encodings
