from xfact.nlp.reader import Reader


@Reader.register("deardr_pt")
class PretrainPT(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": [instance["page"]],
            "instance": instance

        }
        a["nested_entities"] = [a["entities"]]

        yield a


@Reader.register("deardr_hl")
class PretrainHL(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": instance.get("entities",[]),
            "instance": instance
        }

        a["nested_entities"] = [a["entities"]]
        yield a


@Reader.register("deardr_hlfilter")
class PretrainHLFiltered(PretrainHL):
    def filter(self, instance):
        # Filter out if there's no entities
        return not len(instance["entities"])


@Reader.register("deardr_pthl")
class PretrainPTHL(Reader):
    def generate_instances(self, instance):
        a = {
            "source": instance["line"],
            "entities": [instance["page"]] + instance.get("entities",[]),
            "instance": instance
        }

        a["nested_entities"] = [a["entities"]]

        yield a


@Reader.register("deardr_pthlfilter")
class PretrainPTHLFiltered(PretrainPTHL):
    def filter(self, instance):
        # Filter out if there's no entities
        return not len(instance["entities"])