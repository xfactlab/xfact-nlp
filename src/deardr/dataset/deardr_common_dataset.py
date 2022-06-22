import pickle
from typing import List

from deardr.dataset.page_title_prediction_dataset import PageTitlePredictionDataset


class DearDrCommonDataset(PageTitlePredictionDataset):
    def prepare_src(self, source, instance):
        return "predict document titles: " + source

    def prepare_tgt(self, target, instance):
        if 'entities' in instance:
            return f"{self.sep_token}".join(instance["entities"])

        raise Exception("Missing entities arg")
