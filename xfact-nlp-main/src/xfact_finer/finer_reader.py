import re
import logging
import datasets
from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader
import os

logger = logging.getLogger(__name__)
@Reader.register("finer-139-1")
class FinerReader(Reader):

    def __init__(self, filter_instances=None, test_mode=False):
        self.filter_instances = filter_instances
        self.test_mode = test_mode

        super(FinerReader, self).__init__()


    def read(self, path):
        logger.info("reading instances from {}".format(path))
        path, split = path.rsplit(":", maxsplit=1)
        self.dataset = datasets.load_dataset(path)

        yield from self.enumerate(self.dataset[split])

    def enumerate(self, file):
        for idx, instance in enumerate(file):
            if self.filter_instances and not self.test_mode and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 10:
                break

    def generate_instances(self, instance):

        if not any(instance["ner_tags"]):  # no answers. -> make questions with no answers
            return
        buffer = []
        buffer_tag = None

        labels = [self.dataset['train'].features["ner_tags"].feature.names[idx] for idx in instance["ner_tags"]]
        for token, tag in zip(instance["tokens"], labels):

            # If we have a buffer, then we should return the tag if we encounter O or B-
            if (tag == "O" or tag.startswith("B-")) and buffer:
                yield {
                    "context": " ".join(instance["tokens"]),
                    "span": " ".join(buffer),
                    "label": buffer_tag,
                    "instance": instance
                }
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
            yield {
                "context": " ".join(instance["tokens"]),
                "span": " ".join(buffer),
                "label": buffer_tag,
                "instance": instance
            }


if __name__ == "__main__":
    reader = FinerReader()

    for j, a in enumerate(reader.read("nlpaueb/finer-139:test")):
        print(a)


