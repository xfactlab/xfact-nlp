import json
import logging
import os
import sys
from collections import OrderedDict
from operator import itemgetter

import transformers
from datasets import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, T5ForConditionalGeneration,
)

from transformers.utils import check_min_version

from deardr.dataset import dataset_types
from deardr.dataset.page_title_prediction_dataset import PageTitlePredictionDataset
from deardr.frontend import frontend_types, PretrainPT
from deardr.inference.post_processing import post_process_test_multibeam
from deardr.inference.prefix_decoder import multi_document_prefix, single_document_prefix
from deardr.training.args import ModelArguments, DataTrainingArguments
from deardr.training.deardr_trainer import DearDrTestMultiBeamPredictor

check_min_version("4.16.0")
logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(1)

    data_files = {}
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    reader_cls = dataset_types[data_args.dataset_reader]
    frontend_cls = frontend_types[data_args.validation_frontend_reader]

    reader = frontend_cls()
    loaded_datasets = {split: reader_cls(tokenizer,
                                         reader.read(path),
                                         max_seq_length,
                                         max_target_length=data_args.max_target_length,
                                         name=split,
                                         test_mode=True
                                         )
                       for split, path in data_files.items()
                       }

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if "test" not in loaded_datasets:
        raise ValueError("--do_predict requires a test dataset")

    data_collator = lambda batch: PageTitlePredictionDataset.collate_fn(batch, tokenizer.pad_token_id)

    if model_args.predict_single:
        print("Predicting single")

    prefix_decode = single_document_prefix if  isinstance(reader, PretrainPT) or model_args.predict_single else multi_document_prefix

    trainer = DearDrTestMultiBeamPredictor(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_process_test_multibeam,
        eval_dataset=loaded_datasets["test"],
        eval_examples=list(map(itemgetter("instance"), loaded_datasets["test"].instances)),
        prefix_decode=prefix_decode(tokenizer, model_args.prefix_path),
    )

    logger.info("*** Predict ***")
    results = trainer.predict(loaded_datasets["test"],
                              list(map(itemgetter("instance"), loaded_datasets["test"].instances)),
                              max_length=64,
                              num_beams=10,
                              )

    with open(data_args.out_file, "w+") as f:
        for predicted, instance, beam in tqdm(results,desc="Writing results"):
            instance["predicted_documents"] = predicted
            # instance["document_beam_data"] = beam
            f.write(json.dumps(instance) + "\n")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
