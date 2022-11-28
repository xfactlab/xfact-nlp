from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset


@XFactDataset.register("deardr")
class DearDrCommonDataset(XFactSeq2SeqDataset):
    def prepare_src(self, instance):
        return "predict document titles: " + instance["source"]

    def prepare_tgt(self, instance):
        if 'entities' in instance:
            return f"{self.sep_token}".join(instance["entities"])

        raise Exception("Missing entities arg")
