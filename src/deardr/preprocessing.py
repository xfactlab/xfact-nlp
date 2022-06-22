import itertools
from collections import OrderedDict
from typing import List

from drqa.retriever.utils import normalize


def recover(text):
    return normalize(text.replace("-COLON-",":").replace("_"," ").replace("-LRB-","(").replace("-RRB-",")"))


def flatten_list(list_of_lists: List[List]):
    return [x for x in itertools.chain.from_iterable(list_of_lists)]


def flatten_and_deduplicate(list_of_lists: List[List]):
    ret_list = OrderedDict()
    for l in list_of_lists:
        ret_list.update({val: 1 for val in l})
    return list(ret_list.keys())


def deduplicate(items: List[List]):
    ret_list = OrderedDict()
    for l in items:
        ret_list.update({l: 1})
    return list(ret_list.keys())


def deduplicate_list_of_lists(list_of_lists):
    return list(set(tuple(deduplicate(a)) for a in list_of_lists))
