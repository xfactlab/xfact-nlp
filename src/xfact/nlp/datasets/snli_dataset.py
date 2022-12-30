from xfact.nlp.dataset import XFactClassificationDataset, XFactDataset
from xfact.nlp.reader import Reader


@XFactDataset.register("snli")
class SNLIDataset(XFactClassificationDataset):
    def prepare_src(self, instance):
        return instance["sentence1"] + \
               " " + self.sep_token + " " + \
               instance["sentence2"]


@Reader.register("snli")
class SNLIReader(Reader):
    def generate_instances(self, instance):
        if instance["gold_label"] != "-":
            yield {
                "sentence1": instance["sentence1"],
                "sentence2": instance["sentence2"],
                "label": instance["gold_label"],
                "instance": instance
            }