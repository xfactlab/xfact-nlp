import logging
from datasets import tqdm

logger = logging.getLogger(__name__)


def get_redirects(redirs_path):
    rd = dict()
    for line in tqdm(open(redirs_path, encoding='utf-8'),desc="Loading redirects"):
        bits = line.strip().split("\t")
        if len(bits) == 2:
            frm, to = bits
            rd[frm] = to
    return rd


def recursive_redirect_lookup(redirects_list, word):
    if word in redirects_list:
        try:
            found = redirects_list[word]
            return recursive_redirect_lookup(redirects_list, found) if found != word else word
        except RecursionError:
            return word
    else:
        return word