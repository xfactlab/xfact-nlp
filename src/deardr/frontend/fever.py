from deardr.frontend.base_reader import Reader
from deardr.preprocessing import recover, flatten_and_deduplicate, deduplicate_list_of_lists


class FEVERPageLevelReader(Reader):
    def get_evidence(self, evidence):
        return_evidence = []
        for e in evidence:
            found_evidence = set(recover(page) for _, _, page, line in e if page is not None)
            if found_evidence:
                return_evidence.append(found_evidence)

        return flatten_and_deduplicate(sorted(return_evidence,key= lambda a: len(a)))

    def get_pages_fever(self, evidence_sets):
        ev_sets = list()
        for e in evidence_sets:
            if any([es[2] is not None for es in e]):

                ev_sets.append([])
                for _, _, page, line in e:
                    if page is None:
                        continue

                    ev_sets[-1].append(recover(page))

        return deduplicate_list_of_lists(ev_sets)

    def generate_instances(self, instance):
        a = {
            "source": instance["claim"],
            "entities": self.get_evidence(instance["evidence"]),
            "nested_entities": self.get_pages_fever(instance["evidence"]),
            "instance": instance,
        }

        yield a


class FEVERPageLevelReaderSkipNEI(FEVERPageLevelReader):
    def filter(self, instance):
        # Skip if returns NEI
        return instance["label"] == "NOT ENOUGH INFO"
