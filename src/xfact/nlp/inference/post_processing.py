import re
from abc import ABC
from collections import OrderedDict
from xfact.registry.registrable import Registrable


class PostProcessor(Registrable, ABC):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process_text(self, examples, features, predictions, trainer):
        raise NotImplementedError()

    @staticmethod
    def clean(tokenizer, text):
        return text.replace(tokenizer.pad_token, "")\
            .replace(tokenizer.unk_token, "")\
            .replace(tokenizer.eos_token, "")


@PostProcessor.register("default")
class DefaultPostProcessor(PostProcessor):
    def process_text(self, examples, features, predictions, trainer):
        predicted = [
            [self.clean(features.tokenizer, p) for p in features.tokenizer.batch_decode(pred)]
            for pred in predictions.predictions]

        results = []
        for inst in predicted:
            resultset = OrderedDict()

            for beam in inst:
                resultset.update({k: 1 for k in beam})
            results.append(list(resultset.keys()))

        assert len(results) == len(examples) == len(predictions.predictions)
        yield from zip(results, examples, predictions.predictions)

