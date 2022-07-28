import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from deardr.data_utils import encode_line, trim_batch, SortishSampler


class PageTitlePredictionDataset(Dataset):
    sep_token = "<sep />"

    def __init__(
        self,
        tokenizer,
        instance_generator,
        max_source_length,
        max_target_length,
        name="",
        n_obs=None,
        test_mode=False,
        output_prompt=""
    ):
        super().__init__()
        self.instances = list(tqdm(filter(lambda i: i is not None, instance_generator),desc=name))
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.has_preview = 0

        self.output_prompt = output_prompt
        self.prompt_tokens = self.tokenizer(self.output_prompt)['input_ids'][:-1]
        print("Output prompt tokens are ", self.prompt_tokens)


        if not self in self.tokenizer.vocab:
            self.tokenizer.add_tokens([self.sep_token], special_tokens=True)

        self.sep_token_id = self.tokenizer.vocab.get(self.sep_token)
        print("Sep token id is ", self.sep_token_id)

        tokenizer._sep_token = self.sep_token

        self.blind_test_mode = test_mode


    def __len__(self):
        return len(self.instances)

    def prepare_src(self, source, instance):
        raise NotImplementedError()

    def prepare_tgt(self, target, instance):
        raise NotImplementedError()

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        instance = self.instances[index]
        source_line = instance["source"]
        tgt_line = instance["entities"]

        source_input = self.prepare_src(source_line, instance)
        source_inputs = encode_line(
            self.tokenizer, source_input, self.max_source_length
        )

        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        if not self.blind_test_mode:
            target_input = self.prepare_tgt(tgt_line, instance)
            target_inputs = encode_line(
                self.tokenizer, target_input, self.max_target_length
            )
            target_ids = target_inputs["input_ids"].squeeze()

        if self.has_preview < 5:
            self.has_preview += 1
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
    def collate_fn(model, batch,pad_token_id, ignore_pad_token_for_loss=True) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])

        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )

        ret_batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            # "metadata": [x["metadata"] for x in batch if x is not None],
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