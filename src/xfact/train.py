import comet_ml
import logging
import os
import sys
from collections import defaultdict
from operator import itemgetter

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from xfact.config.args import ModelArguments, DataTrainingArguments
from xfact.logs.comet_callback import CometTrainingCallback
from xfact.logs.logs import setup_logging
from xfact.nlp.dataset import XFactDataset, XFactSeq2SeqDataset
from xfact.nlp.deardr_trainer import DearDrTrainer, XFactClsTrainer
from xfact.nlp.model import ModelFactory
from xfact.nlp.post_processing import PostProcessor
from xfact.nlp.reader import Reader
from xfact.nlp.scoring import Scorer
from xfact.registry.module import import_submodules




check_min_version("4.16.0")
logger = logging.getLogger(__name__)


def main():
    remaining_args = None
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)



    setup_logging(training_args.get_process_log_level())

    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if remaining_args:
        logger.error(f"Extra args found: {remaining_args}")
        raise Exception("Extra args")

    if data_args.package:
        logger.info("Trying to load project packages")
        import_submodules(data_args.package)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    if data_args.comet_key and training_args.should_log and (training_args.do_train or training_args.do_eval):
        experiment = comet_ml.Experiment(
            api_key=data_args.comet_key,
            workspace=data_args.workspace,
            project_name=data_args.project_name if data_args.project_name is not None else f"debugging-xfact",
            experiment_key=data_args.experiment_key if data_args.experiment_key is not None else None,

        )
        experiment.log_parameters(training_args.to_dict())
    else:
        experiment = None

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

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


    reader_classes = {
        "train": Reader.resolve(data_args.reader),
        "validation": Reader.resolve(data_args.validation_reader or data_args.reader)
    }

    readers = {
        "train": reader_classes["train"](),
        "validation": reader_classes["validation"](test_mode=True)
    }


    dataset_classes = {
        "train": XFactDataset.resolve(data_args.dataset),
        "validation":  XFactDataset.resolve(data_args.validation_dataset or data_args.dataset)
    }


    is_seq2seq = issubclass(dataset_classes["train"],XFactSeq2SeqDataset)
    logger.info(f"Is seq2seq? {is_seq2seq}")

    extra_kwargs = {}
    if is_seq2seq:
        extra_kwargs["max_target_length"] = data_args.max_target_length
    else: # not is_seq2seq:
        global_labels = defaultdict(int)
        extra_kwargs["label_dict"] = global_labels

    loaded_datasets = {
        split: dataset_classes[split](tokenizer,
                                               readers[split].read(path),
                                               max_seq_length,
                                               name=split,
                                               # test_mode=not split == "train",
                                      **extra_kwargs)
        for split, path in data_files.items()
    }

    if training_args.do_train:
        if "train" not in loaded_datasets:
            raise ValueError("--do_train requires a train dataset")

    if training_args.do_eval:
        if "validation" not in loaded_datasets:
            raise ValueError("--do_eval requires a validation dataset")
    #
    #

    model = ModelFactory.resolve(model_args.model_factory)().get_model(
        model_args.model_name_or_path,
        tokenizer,
        loaded_datasets,
        model_args,
        data_args

    )


    data_collator = lambda batch: dataset_classes["train"].collate_fn(model, batch, tokenizer.pad_token_id, data_args.ignore_pad_token_for_loss)
    post_processor = PostProcessor.init(data_args.post_processor, **{"tokenizer": tokenizer, "model": model})
    scorer = Scorer.init(data_args.scorer)
    logging_callback = CometTrainingCallback(experiment)
    trainer_cls = DearDrTrainer if is_seq2seq else XFactClsTrainer # Maybe do multiple beams with DearDrPredictor but this is SLLOOOOWWWW
    # data_collator = default_data_collator
    # prefix_decode = single_document_prefix if  isinstance(readers["validation"], PretrainPT) else multi_document_prefix

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=loaded_datasets["train"] if training_args.do_train else None,
        eval_dataset=loaded_datasets["validation"] if training_args.do_eval else None,
        eval_examples=list(map(itemgetter("instance"), loaded_datasets["validation"].instances)) if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=scorer,
        post_process_function=post_processor.process_text,
        # train_beam=data_args.train_beam,
        # prefix_decode=prefix_decode(tokenizer, model_args.prefix_path),
        callbacks=[logging_callback],

    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(loaded_datasets['train'])
        )
        metrics["train_samples"] = min(max_train_samples, len(loaded_datasets['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.max_target_length,
            num_beams=data_args.eval_beam # This is fast enough to estimate the R-precision which typically requires less than 2 elements but not perfect
        )

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(loaded_datasets['validation'])
        metrics["eval_samples"] = min(max_eval_samples, len(loaded_datasets['validation']))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero() and experiment:
            experiment.log_metrics(dic={"final/" + key: value for key, value in metrics.items()})

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "dear-dr"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
