import json
import logging
import os
from abc import ABC

from xfact.registry.registrable import Registrable

logger = logging.getLogger(__name__)


class Reader(Registrable, ABC):

    def __init__(self, filter_instances =None, test_mode = False):
        self.filter_instances = filter_instances
        self.test_mode = test_mode

    def read(self, path):
        logger.info("reading instances from {}".format(path))

        with open(path) as f:
            yield from self.enumerate(f)

    def enumerate(self, file):
        for idx, line in enumerate(file):
            instance = json.loads(line)

            if self.filter_instances and not self.test_mode and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 10:
                break

    def generate_instances(self, instance):
        raise NotImplementedError()

    def filter(self, instance):
        return False

