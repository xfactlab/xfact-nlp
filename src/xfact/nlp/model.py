from abc import ABC
import logging

from overrides import overrides
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from xfact.nlp.dataset import XFactSeq2SeqDataset
from xfact.registry.registrable import Registrable

logger = logging.getLogger(__name__)


class ModelFactory(Registrable, ABC):
    def get_model(self,
                  model_name_or_path,
                  tokenizer,
                  loaded_datasets,
                  model_args,
                  data_args):
        raise NotImplementedError()


@ModelFactory.register("huggingface")
class HuggingFaceModelFactory(ModelFactory):
    @overrides
    def get_model(self,
                  model_name_or_path,
                  tokenizer,
                  loaded_datasets,
                  model_args,
                  data_args):

        extra_model_kwargs = {}

        is_seq2seq = isinstance(loaded_datasets["train"], XFactSeq2SeqDataset)
        logger.info(f"Is seq2seq? {is_seq2seq}")

        if is_seq2seq:
            model_cls = AutoModelForSeq2SeqLM

        else:
            model_cls = AutoModelForSequenceClassification
            extra_model_kwargs["num_labels"] = len(loaded_datasets["train"].label_dict)
            extra_model_kwargs["label2id"] = loaded_datasets["train"].label_dict
            extra_model_kwargs["id2label"] = {v: k for k, v in loaded_datasets["train"].label_dict.items()}

            if data_args.weighted_loss:
                extra_model_kwargs["class_weights"] = loaded_datasets["train"].class_weights

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            **extra_model_kwargs
        )
        if model_args.dropout_rate is not None:
            config.dropout_rate = model_args.dropout_rate

        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,

        )

        tok_length = len(tokenizer.vocab)
        if len(tokenizer.vocab) > tok_length:
            print("resizing vocab")
            model.resize_token_embeddings(len(tokenizer))

        return model
