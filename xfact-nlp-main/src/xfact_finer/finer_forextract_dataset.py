import os

import re
import logging
from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader

import operator
from abc import ABC
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from xfact.nlp.data_utils import encode_line, trim_batch, SortishSampler
from xfact.registry.registrable import Registrable
from overrides import overrides
logger = logging.getLogger(__name__)


hierarchical_dict = {"monetary values":{"ab":"df"},"percentage":{"df"},"ratio":"df"}

@XFactDataset.register("extracting-finer-139")
class NERQADataset(XFactSeq2SeqDataset):
    def __init__(self, tokenizer, instance_generator, max_source_length, generation_max_length=64, prefix_decode=None, eval_examples=None, post_process_function=None, train_beam=10,output_prompt="", **kwargs):
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.predict_with_generate = True
        self.generation_max_length = generation_max_length
        self.generation_num_beams = train_beam
        self.prefix_decode = prefix_decode
        self.generation_config = None
        self.output_prompt = output_prompt
        self.prompt_tokens = tokenizer(self.output_prompt)['input_ids'][:-1]
        # self.prompt_mapper = prompt_mapper
        self.max_target_length = self.generation_max_length
        self.output_prompt = output_prompt
        # self.prompt_tokens = tokenizer(self.output_prompt)['input_ids'][:-1]
        self.hierarchical_dict = hierarchical_dict
        self.__dict__.update(kwargs)

        super(NERQADataset, self).__init__(tokenizer=tokenizer,
                                                  instance_generator=instance_generator,
                                                  max_source_length=max_source_length,
                                                  output_prompt=output_prompt,
                                                  **kwargs)

    def prepare_src(self, instance):
        context = instance["context"]
        options = list(hierarchical_dict.keys())
        # f"Given sentence: \"{context}\" The known entity types are: \"{options}\". Please answer: Extract all the text spans that belong to any entity types. Please ensure that the identified text spans do not overlap between different entity types."
        return f"Given sentence: \"{context}\" The known entity types are: \"{options}\". Please answer: Extract all the text spans that refers to any entity types. Please ensure that the identified text spans do not overlap."

    def prepare_tgt(self, instance):
        if instance["span"] == None:
            return "No answer"
        else:
            return ", ".join(instance["span"])#따지고 보면 span list 이다.

    @staticmethod
    def _split_string(str):
        return " ".join(re.findall(r"[A-Z][^A-Z]*", str))
