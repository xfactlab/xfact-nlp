import itertools
import unicodedata

from deardr.frontend.base_reader import Reader
from deardr.preprocessing import recover


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)



def get_pages(e):
    ev = set()
    for _,_, page,line in e:
        if page is not None:
            ev.add(recover(page))

    return list(ev)


class FEVERTestReader(Reader):

    @staticmethod
    def get_pages(evidence_sets):
        ev = set()
        for e in evidence_sets:
            for _, _, page, line in e:
                ev.add(normalize(recover(page)))

        return list(ev)

    def generate_instances(self, instance):

        if instance["label"] == "NOT ENOUGH INFO":
            return []

        evidence = itertools.chain(*instance["evidence"])
        a = {
            "source": instance["claim"],
            "entities": get_pages(evidence),
            "instance": instance
        }

        if len(a["entities"]):
            yield a
