import json
import logging
import csv
from collections import OrderedDict
from datetime import datetime

import spacy
from spacy.tokens import Span
from tqdm import tqdm

logger = logging.getLogger()


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Span) or isinstance(obj, datetime):
            return str(obj)
        elif isinstance(obj, OrderedDict):
            return obj.__dict__
        return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    def read(path):
        logger.info("reading instances from {}".format(path))

        with open(path) as f:
            reader = csv.DictReader(f)
            for idx, line in enumerate(tqdm(reader)):
                yield from generate_instances(idx, line)

    def generate_instances(idx, instance):
        abstract = instance["abstract"]
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(abstract)
        del instance["abstract"]
        for sent_idx, sent in enumerate(doc.sents):
            yield {
                "source": sent,
                "instance": instance,
                "entities": [],
                "doc_idx": idx,
                "sent_idx": sent_idx
            }

    with open("data/yahoo_split.jsonl","w+") as f:
        for inst in read("data/yahoo.csv"):
            f.write(json.dumps(inst, cls=MyEncoder)+"\n")


