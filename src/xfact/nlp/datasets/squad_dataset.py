import re
import logging

from xfact.nlp.dataset import XFactDataset, XFactSeq2SeqDataset

logger = logging.getLogger(__name__)


@XFactDataset.register("squad-open")
class SQuADOpenDataset(XFactSeq2SeqDataset):
    def prepare_src(self, instance):
        return instance["question"] + " ::: " + instance["context"]

    def prepare_tgt(self, instance):

        answer = instance["answers"]["text"]
        if not answer:
            return "No Answer"
        else:
            return answer[0]



@XFactDataset.register("squad-closed")
class SQuADClosedDataset(XFactSeq2SeqDataset):
    def prepare_src(self, instance):
        return instance["question"]

    def prepare_tgt(self, instance):
        return instance["answers"]["text"][0]