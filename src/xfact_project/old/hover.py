import json
import os

from deardr.frontend.base_reader import Reader
from deardr.preprocessing import recover


def get_pages(e):
    ev = set()
    for page,line in e:
        ev.add(recover(page))

    return list(ev)

class HOVERPageLevelReader(Reader):
    def enumerate(self, file):

        json_loaded = json.load(file)

        for idx, instance in enumerate(json_loaded):
            if not self.test_mode and self.filter is not None and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if (os.getenv("DEBUG") is not None and idx > 10):
                break


    def generate_instances(self, instance):
        ents = get_pages(instance["supporting_facts"])
        a = {
            "source": instance["claim"],
            "entities": ents,
            "nested_entities": [ents],
            "instance": instance,
        }

        yield a


