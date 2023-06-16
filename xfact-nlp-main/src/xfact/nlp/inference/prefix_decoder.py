import pickle
from typing import List, Iterable
import logging

logger = logging.getLogger(__name__)


class PrefixTree():
    def __init__(self):
        self.root = dict()

    def add_string(self, other:Iterable[int]):
        current = self.root
        for tok in other:
            previous = current
            assert previous is not None

            current = current.get(tok, None)

            if current is None:
                current = dict()
                previous[tok] = current

    def get_next(self, test_string: Iterable[int]):
        current = self.root

        for tok in test_string:
            current = current.get(tok, None)

            if current is None:
                return None

        return current


def single_document_prefix(tokenizer, path):
    with open(path, "rb") as f:
        print("loading prefix decoder")
        prefix_decoder = pickle.load(f)
        print("dne")

    def do_prefix_decode(batch_id, hypothesis) -> List[int]:
        decode_string = hypothesis.cpu().tolist()

        if len(decode_string) == 1 and (decode_string[0] == tokenizer.pad_token_id):
            return list(prefix_decoder.root.keys())

        elif len(decode_string) > 1 and decode_string[-1] == tokenizer.eos_token_id:
            return []

        if decode_string[0] == 0:
            decode_string = decode_string[1:]

        next_tokens = prefix_decoder.get_next(decode_string)

        if next_tokens is None:
            return []

        return list(next_tokens.keys())

    return do_prefix_decode


def multi_document_prefix(tokenizer, path):
    with open(path, "rb") as f:
        logger.info("Loading prefix decoder")
        prefix_decoder = pickle.load(f)
        logger.info("Done")

    def do_prefix_decode_line_number(batch_id, hypothesis) -> List[int]:
        decode_string = hypothesis.cpu().tolist()

        if decode_string[0] == 0:
            decode_string = decode_string[1:]

        if tokenizer.sep_token_id in decode_string:
            decode_string = decode_string[len(decode_string) - decode_string[::-1].index(tokenizer.sep_token_id):]

        if len(decode_string) == 0:
            return list(prefix_decoder.root.keys())

        elif len(decode_string) > 1 and decode_string[-1] == tokenizer.eos_token_id:
            return []

        next_tokens = prefix_decoder.get_next(decode_string)
        if next_tokens is None:
            return []

        if tokenizer.eos_token_id in next_tokens:
            return [tokenizer.sep_token_id] + list(next_tokens.keys())

        return list(next_tokens.keys())

    return do_prefix_decode_line_number