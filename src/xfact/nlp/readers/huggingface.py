from xfact.nlp import Reader
from xfact.logs.logs import setup_logging
import logging

import os


logger = logging.getLogger(__name__)


@Reader.register("huggingface")
class HuggingFaceReader(Reader):

    def __init__(self, filter_instances =None, test_mode = False):
        self.filter_instances = filter_instances
        self.test_mode = test_mode

    def read(self, path):
        try:
            import datasets
        except ImportError:
            raise ImportError("Datasets library must be installed")

        logger.info("reading instances from {}".format(path))
        path, split = path.rsplit(":",maxsplit=1)
        self.dataset = datasets.load_dataset(path)

        yield from self.enumerate(self.dataset[split])

    def enumerate(self, file):
        for idx, instance in enumerate(file):
            if self.filter_instances and not self.test_mode and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 1000:
                break

    def generate_instances(self, instance):
        a = {"instance": instance}
        a.update(**instance)
        yield a
