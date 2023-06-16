import re
import logging
import datasets
from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader
import os

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


############## 정의해야함
# hierarchical_dict =  json 형식으로 들고 올것


"""
prompt_mapper = {
    "LOC": "location",
    "PER": "person",
    "ORG": "organization",
    "MISC": "miscellaneous entity"
}
"""
hierarchical_dict = {
    "A":{"a-1":{"a-1-2","a-1-2"},"a-2":"a-2-1"},
    "B":{"b-1":{"b-1-2","b-1-1"},"b-2":{"b-1"}},
}

# 이게 맞음.. ! 다른 거 쓰지마

def build_question(hierarchical_dict, list_chain, depth, context, qa):

    if qa == "end_wq":

        starting_question = """The following is a conversation between a human and an AI. The AI only answers based on the requested from the user and doesn't provide details. \If the answer doesn't exist, AI says "**NONE**"\n###HUMAN: I’m going to give you (a) sentence(s) with one span. I will let you to identify the category that a word belongs to based on the context. There will only be {0} types of entities: {1}."{2}" Give an Answer without explanation\n###AI:The answer is """

        starting_question = starting_question.format(len(hierarchical_dict), list(hierarchical_dict.keys()), context)

        continuing_answer = "{0}"
        continuing_question = """\n###HUMAN: As it belongs to \'{0}\', I will let you to identify the sub category of \'{0}\' that a word belongs to based on the context. There will only be {1} types of entities: {2}.\"{3}\" Give an Answer without explanation.\n###AI:The answer is """
        # d
        if depth >= len(list_chain):
            raise ValueError(
                f"Depth ({depth}) cannot be greater than or equal to the length of listchain ({len(list_chain)}).")

        for i in range(depth):
            # import pdb;pdb.set_trace()
            if i >= len(list_chain):
                break  # We've exhausted all keys in listchain.

            key = list_chain[i]  # Key to access current level of hierarchical_dict.
            if isinstance(hierarchical_dict, dict) and key in hierarchical_dict:
                # Append to question string.
                starting_question += continuing_answer.format(key)

                # Update hierarchical_dict and choices for next level.
                hierarchical_dict = hierarchical_dict[key]
                choices = list(hierarchical_dict.keys()) if isinstance(hierarchical_dict, dict) else hierarchical_dict
                length = len(choices)

                # Append next part of question.
                starting_question += continuing_question.format(continuing_answer.format(key), length, choices, context,
                                                                choices)
        return starting_question

def build_answer(list_chain, depth, qa):
    if qa == "end_wq":
        answer = list_chain[depth]

    return answer

"""
# Example usage:
context = "japanese is"
qa = "end_wq"
depth = 2
list_chain = ['price', 'priceA', 'PriceA-2']

result = build_question(hierarchical_dict, list_chain, depth, context, qa)
print(result)
"""

# # Example usage:
# context = "japanese is"
# qa = "end_wq"
# i_depth = 2
# result = build_question(hierarchical_dict, listchain, i_depth, context, qa)
# print(result)


@XFactDataset.register("finer-139-askanswer-dataset")
class NERQADataset(XFactSeq2SeqDataset):
    def __init__(
            self,
            tokenizer,
            instance_generator,
            max_source_length,
            max_target_length=32,
            name="",
            n_obs=None,
            test_mode=False,
            output_prompt="",
            streaming=False,
            # prompt_mapper = prompt_mapper
    ):
        # self.prompt_mapper = prompt_mapper
        self.max_target_length = max_target_length
        self.output_prompt = output_prompt
        self.prompt_tokens = tokenizer(self.output_prompt)['input_ids'][:-1]
        self.hierarchical_dict = hierarchical_dict

        super(XFactSeq2SeqDataset, self).__init__(tokenizer=tokenizer,
                                                  instance_generator=instance_generator,
                                                  max_source_length=max_source_length,
                                                  name=name,
                                                  n_obs=n_obs,
                                                  test_mode=test_mode,
                                                  output_prompt=output_prompt,
                                                  streaming=streaming)


        logger.info(f"Output prompt tokens are {self.prompt_tokens}")



    def prepare_src(self, instance, qa = "end_wq"):
        list_chain = instance["list_chain"]
        i_depth = instance["i_depth"]
        context = instance["context"]

        return build_question(self.hierarchical_dict, list_chain, i_depth, context, qa)
        # return f"What is the {self.prompt_mapper[instance['label']]}? {instance['context']}"

    def prepare_tgt(self, instance):
        list_chain = instance["list_chain"]
        return instance['span']

    @staticmethod
    def _split_string(str):
        return " ".join(re.findall(r"[A-Z][^A-Z]*", str))


