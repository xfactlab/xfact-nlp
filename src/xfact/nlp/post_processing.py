from abc import ABC
from collections import OrderedDict
from xfact.registry.registrable import Registrable


class PostProcessor(Registrable, ABC):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process_text(self, examples, features, predictions, trainer):
        raise NotImplementedError()


    def clean(self, text):
        # Check the internal _bos_token to prevent tokenizer from spamming errors
        return text.replace(self.tokenizer.pad_token, "")\
            .replace(self.tokenizer.unk_token, "<UNK>")\
            .replace(self.tokenizer.eos_token if self.tokenizer._eos_token else "", "")\
            .replace(self.tokenizer.sep_token if self.tokenizer._sep_token else "", "")\
            .replace(self.tokenizer.bos_token if self.tokenizer._bos_token else "", "")


@PostProcessor.register("default")
class DefaultPostProcessor(PostProcessor):
    def process_text(self, examples, features, predictions, trainer):
        predicted = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.predictions)]
        actual = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.label_ids)]

        assert len(actual) == len(examples) == len(predicted)

        return {
            "predicted": predicted,
            "actual": actual
        }


@PostProcessor.register("classification")
class LabelOnly(PostProcessor):
    def process_text(self, examples, features, predictions, trainer):
        predicted = [trainer.model.config.id2label[p] for p in predictions.predictions.argmax(axis=-1)]
        actual = [f['label'] for f in features.instances]

        assert len(actual) == len(predicted)
        assert not examples or len(examples) == len(actual)

        return {
            "predicted": predicted,
            "actual": actual
        }

@PostProcessor.register("nested")
class NestedPostProcessor(PostProcessor):
    def process_text(self, examples, features, predictions, trainer):
        predicted = [
            [self.clean(r).strip() for r in p.split(self.tokenizer.sep_token)] for p in
            features.tokenizer.batch_decode(predictions.predictions)
        ]

        actual = [
            [self.clean(r).strip() for r in p.split(self.tokenizer.sep_token)] for p in
            features.tokenizer.batch_decode(predictions.label_ids)
        ]

        assert len(actual) == len(examples) == len(predicted)

        return {
            "predicted": predicted,
            "actual_flat": actual,
            "actual": [f['nested_entities'] for f in features.instances]
        }

