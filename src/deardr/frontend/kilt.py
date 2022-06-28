from collections import OrderedDict

from deardr.frontend.base_reader import Reader
from deardr.preprocessing import flatten_and_deduplicate, recover


def get_titles(list_of_dicts):
    return list(OrderedDict({recover(dic["title"]):1 for dic in list_of_dicts}).keys())


class KILTPageLevelReader(Reader):
    def generate_instances(self, instance):
        entities = [get_titles(output["provenance"]) for output in instance["output"] if "provenance" in output]
        a = {
            "source": instance["input"].replace("[START_ENT] ","").replace(" [END_ENT]",""),
            "entities": flatten_and_deduplicate(entities),
            "nested_entities": entities,
            "instance": instance,
        }

        yield a