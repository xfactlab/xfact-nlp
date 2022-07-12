from deardr.frontend.base_reader import Reader


class PretrainPT(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": [instance["page"]],
            "instance": instance

        }
        a["nested_entities"] = [a["entities"]]

        yield a


class PretrainHL(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": instance.get("entities",[]),
            "instance": instance
        }

        a["nested_entities"] = [a["entities"]]
        yield a


class PretrainHLFiltered(PretrainHL):
    def filter(self, instance):
        # Filter out if there's no entities
        return not len(instance["entities"])


class PretrainPTHL(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": [instance["page"]] + instance.get("entities",[]),
            "instance": instance
        }

        a["nested_entities"] = [a["entities"]]

        yield a


class PretrainPTHLFiltered(PretrainPTHL):
    def filter(self, instance):
        # Filter out if there's no entities
        return not len(instance["entities"])