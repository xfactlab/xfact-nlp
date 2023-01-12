import logging
import operator
from abc import ABC
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from xfact.nlp.data_utils import encode_line, trim_batch, SortishSampler
from xfact.registry.registrable import Registrable
from overrides import overrides
logger = logging.getLogger(__name__)


class XFactDataset(TorchDataset, Registrable, ABC):
    def __init__(
            self,
            tokenizer,
            instance_generator,
            max_source_length,
            name="",
            n_obs=None,
            test_mode=False,
            output_prompt="",
            streaming=False,
            num_instances_to_preview=5,
            sep_token="<sep />"
    ):
        super().__init__()


        # Default sep token. If tokenizer has a sep token, this will be ignored.
        self.sep_token = tokenizer._sep_token or sep_token

        # Set class specific things
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer

        # We want to preview a number of instances
        self.num_instances_to_preview = num_instances_to_preview
        self.blind_test_mode = test_mode
        self.has_preview = num_instances_to_preview

        # Set pad token
        self.pad_token_id = self.tokenizer.pad_token_id

        # Set sep token
        if not self.sep_token in self.tokenizer.vocab:
            logger.info("Tokenizer doesn't have a sep token. Create it")
            self.tokenizer.add_tokens([self.sep_token], special_tokens=True)
            self.tokenizer._sep_token = self.sep_token

        self.sep_token_id = self.tokenizer.vocab.get(self.sep_token)
        self.sep_token = self.tokenizer._sep_token
        logger.info(f"Sep token id is {self.sep_token_id}")

        # If we're streaming instances from disk, then we should continuously generate instances
        # Otherwise, load instances from an instance generator
        self.streaming = streaming
        if not self.streaming:
            self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator), desc=name))
            self.generated = list(tqdm(map(self.generate, self.instances), desc="Generating instances"))
        else:
            self.instances = filter(lambda i: i is not None, instance_generator)

        # Needed for Sortish sampler
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.generated)

    def prepare_src(self, instance):
        raise NotImplementedError()

    def prepare_tgt(self, instance):
        raise NotImplementedError()

    def generate(self, instance) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.generated[index] if not self.streaming else \
            self.generate(next(self.instances))

        return instance

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y

    @staticmethod
    def collate_fn(model, batch, pad_token_id, ignore_pad_token_for_loss=True) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])

        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        ret_batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }

        if "decoder_input_ids" in batch[0]:
            target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
            y = trim_batch(target_ids, -100 if ignore_pad_token_for_loss else pad_token_id)
            ret_batch["labels"] = y

            # prepare decoder_input_ids
            if (
                    ret_batch['labels'] is not None
                    and model is not None
                    and hasattr(model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=ret_batch["labels"])
                ret_batch["decoder_input_ids"] = decoder_input_ids

            # ret_batch["decoder_input_ids"] = y

        return ret_batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)


class XFactSeq2SeqDataset(XFactDataset, ABC):
    def __init__(
            self,
            tokenizer,
            instance_generator,
            max_source_length,
            max_target_length=32,
            name="",
            n_obs=None,
            test_mode=False,
            output_prompt="",
            streaming=False
    ):
        self.max_target_length = max_target_length
        self.output_prompt = output_prompt
        self.prompt_tokens = tokenizer(self.output_prompt)['input_ids'][:-1]

        super(XFactSeq2SeqDataset, self).__init__(tokenizer=tokenizer,
                                                  instance_generator=instance_generator,
                                                  max_source_length=max_source_length,
                                                  name=name,
                                                  n_obs=n_obs,
                                                  test_mode=test_mode,
                                                  output_prompt=output_prompt,
                                                  streaming=streaming)

        logger.info(f"Output prompt tokens are {self.prompt_tokens}")

    @staticmethod
    def collate_fn(model, batch, pad_token_id, ignore_pad_token_for_loss=True) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])

        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        ret_batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }

        if "decoder_input_ids" in batch[0]:
            target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
            y = trim_batch(target_ids, -100 if ignore_pad_token_for_loss else pad_token_id)
            ret_batch["labels"] = y

            # prepare decoder_input_ids
            if (
                    ret_batch['labels'] is not None
                    and model is not None
                    and hasattr(model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=ret_batch["labels"])
                ret_batch["decoder_input_ids"] = decoder_input_ids

            # ret_batch["decoder_input_ids"] = y

        return ret_batch

    def generate(self, instance) -> Dict[str, torch.Tensor]:
        source_input = self.prepare_src(instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        if not self.blind_test_mode:
            target_input = self.prepare_tgt(instance)
            target_inputs = encode_line(
                self.tokenizer, target_input, self.max_target_length
            )
            target_ids = target_inputs["input_ids"].squeeze()

        if self.has_preview >= 0:
            self.has_preview -= 1
            print(source_input)
            print(source_ids)

            if not self.blind_test_mode:
                print(target_input)
                print(target_ids)

            print("*" * 100)

        ret = {
            "input_ids": source_ids,
            "attention_mask": src_mask
        }

        if not self.blind_test_mode:
            ret["decoder_input_ids"] = target_ids

        return ret


class XFactTaggingDataset(XFactDataset):
    pass


class XFactClassificationDataset(XFactDataset, ABC):
    def __init__(self, tokenizer, instance_generator, max_source_length, label_dict=None, **kwargs):
        self.label_dict = defaultdict(int) if label_dict is None else label_dict
        super().__init__(tokenizer, instance_generator, max_source_length,**kwargs)

        try:
            self.class_weights = self.get_label_distribution()
        except KeyError:
            logger.error("Not possible to determine label distribution. Check that instance contains a label")

    def get_label_distribution(self):
        label_counts = Counter(map(operator.itemgetter("label"), self.instances))

        logger.info(f"Class sizes: {label_counts}")
        summed = sum(label_counts.values())
        return {k:v/summed for k,v in label_counts.items()}


    def prepare_tgt(self, instance):
        return instance["label"]

    @overrides
    def generate(self, instance) -> Dict[str, torch.Tensor]:
        # instance = self.instances[index] if not self.streaming else next(self.instances)

        source_input = self.prepare_src(instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        src_types = source_inputs["token_type_ids"].squeeze() if "token_type_ids" in source_inputs else None

        if not self.blind_test_mode:
            target_input = self.prepare_tgt(instance)

            if target_input not in self.label_dict:
                self.label_dict[target_input] = len(self.label_dict)
            target_ids = torch.LongTensor([self.label_dict[target_input]])


        if self.num_instances_to_preview >= 0:
            self.num_instances_to_preview -= 1
            print(source_input)
            print(source_ids)

            if not self.blind_test_mode:
                print(target_input)
                print(target_ids)

            print("*" * 100)

        ret = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "token_type_ids": src_types
        }

        if not self.blind_test_mode:
            ret["label_ids"] = target_ids

        return ret

    @staticmethod
    def collate_fn(model, batch, pad_token_id, ignore_pad_token_for_loss=True) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])

        # types = torch.stack([x["token_type_ids"] for x in batch]) if "token_type_ids" in batch[0] and batch[0]["token_type_ids"] is not None else None

        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        # source_ids, source_mask = input_ids,masks
        #     # ,types
        ret_batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            # "token_type_ids": source_types
        }

        if "label_ids" in batch[0]:
            target_ids = torch.stack([x["label_ids"] for x in batch])
            ret_batch["labels"] = target_ids


            # ret_batch["decoder_input_ids"] = y

        return ret_batch
