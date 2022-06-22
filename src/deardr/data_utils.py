#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import itertools
import json
import pickle
from logging import getLogger
from typing import Callable, Dict, Iterable, List

import numpy as np
from torch import nn
from torch.utils.data import Sampler
from transformers import BartTokenizer



def recursive_clean(metadata_dict):
    if isinstance(metadata_dict, dict):
        return {
            k: recursive_clean(v) for k, v in metadata_dict.items() if v is not None
        }
    elif isinstance(metadata_dict, list) or isinstance(metadata_dict, tuple):
        return [recursive_clean(k) for k in metadata_dict if k is not None]
    else:
        return metadata_dict


def encode_line(
    tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
):
    extra_kw = (
        {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    )
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate(
            [sorted(s, key=self.key, reverse=True) for s in ck_idx]
        )
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax(
            [self.key(ck[0]) for ck in ck_idx]
        )  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = (
            ck_idx[max_ck],
            ck_idx[0],
        )  # then make sure it goes first.
        sort_idx = (
            np.concatenate(np.random.permutation(ck_idx[1:]))
            if len(ck_idx) > 1
            else np.array([], dtype=np.int)
        )
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)

def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(
        model_grads
    ), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def is_truthy(arg):
    return arg.strip().lower() in ["1", "y", "yes", "t", "true"]


