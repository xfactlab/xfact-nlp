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

class YahooAbstractSentSplitReader(Reader):


    def generate_instances(self, instance):
        yield {
            "source": instance["source"],
            "entities": [],
            "instance": instance
        }


