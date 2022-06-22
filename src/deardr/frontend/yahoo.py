import csv
import json
import logging
import os

from deardr.frontend.base_reader import Reader
from deardr.preprocessing import recover

logger = logging.getLogger()

def get_pages(e):
    ev = set()
    for page,line in e:
        ev.add(recover(page))

    return list(ev)

class YahooAbstractReader(Reader):

    def read(self, path):
        logger.info("reading instances from {}".format(path))

        with open(path) as f:
            reader = csv.DictReader(f)
            yield from self.enumerate(reader)

    def enumerate(self, file):
        for idx, instance in enumerate(file):
            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 10:
                break

    def generate_instances(self, instance):

        a = {
            "source": instance["abstract"],
            "instance": instance,
            "entities": [],

        }

        yield a


