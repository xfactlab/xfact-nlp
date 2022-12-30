from xfact.nlp.dataset import XFactClassificationDataset, XFactDataset
from xfact.nlp.reader import Reader


@XFactDataset.register("fever-claim-only")
class FEVERDataset(XFactClassificationDataset):
    def prepare_src(self, instance):
        return instance["claim"]



@Reader.register("fever")
class FEVERReader(Reader):
    def generate_instances(self, instance):
        yield {
            "claim": instance["claim"],
            "evidence": instance["evidence"],
            "label": instance["label"],
            "instance": instance
        }