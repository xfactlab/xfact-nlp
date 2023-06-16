import re
import logging
import datasets
from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader
import os
import sys
import pickle
logger = logging.getLogger(__name__)


@Reader.register("extractor_reader")
class FinerReader(Reader):

    def __init__(self, filter_instances=None, test_mode=False):
        self.filter_instances = filter_instances
        self.test_mode = test_mode

    def read(self, path):

        logger.info("reading instances from {}".format(path))
        path, split = path.rsplit(":", maxsplit=1)

        if "pkl" in path:
            with open(path, 'rb') as f:
                self.dataset = pickle.load(f)

                dataset2 = datasets.load_dataset("nlpaueb/finer-139")
                self.tmp_label = dataset2['test'].features["ner_tags"].feature.names

            yield from self.enumerate(self.dataset)
        else:
            self.dataset = datasets.load_dataset(path)
            # self.un_list_chain = wrap_generate_paths(hierarchical_dict)
            yield from self.enumerate(self.dataset[split])

    def enumerate(self, file):
        for idx, instance in enumerate(file):
            if self.filter_instances and not self.test_mode and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 100:
                break

    def generate_instances(self, instance):

        """ in here span and label means list of span and labels."""
        if not any(instance["ner_tags"]):  # no answers. -> make questions with no answers
            return {
                "context": " ".join(instance["tokens"]),
                "span": None,
                "label": None,
                "instance": instance
            }

        buffer = []
        buffer_tag = None

        if hasattr(self, 'tmp_label'):

            labels = [self.tmp_label[idx] for idx in instance["ner_tags"]]
        else:
            labels = [self.dataset['train'].features["ner_tags"].feature.names[idx] for idx in instance["ner_tags"]]
        # tokens :[ "App","ple","O","HE"]
        # labels :[ "B-fruit","I-fruit","O","B-human"]

        span_list = []
        label_list = []

        for token, tag in zip(instance["tokens"], labels):

            # If we have a buffer, then we should return the tag if we encounter O or B-
            if (tag == "O" or tag.startswith("B-")) and buffer:
                span_list.append(" ".join(buffer))
                # label_list.append(" ".join(buffer))

                # yield {
                #     "context": " ".join(instance["tokens"]),
                #     "span": " ".join(buffer),
                #     "label": buffer_tag,
                #     "instance": instance
                # }

                buffer = []
                buffer_tag = None

            # If it starts with a B- we add to buffer
            if tag.startswith("B-"):
                buffer_tag = tag.replace("B-", "")
                buffer.append(token)

            # If it starts with a I- we continue adding to buffer
            elif tag.startswith("I-"):
                buffer.append(token)

        if buffer:
            span_list.append(" ".join(buffer))

            # yield {
            #     "context": " ".join(instance["tokens"]),
            #     "span": " ".join(buffer),
            #     # "label": buffer_tag,
            #     # "instance": instance
            # }

        yield {
            "context": " ".join(instance["tokens"]),
            "span": span_list,#ex,["apple","banana"]
            "label": span_list,
            "instance": instance
        }


if __name__ == "__main__":
    reader = FinerReader()

    for a in reader.read("nlpaueb/finer-139:test"):
        print(a)
