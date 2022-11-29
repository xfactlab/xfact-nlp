import logging
import os
import sys
from operator import itemgetter
import transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSeq2SeqLM
)
from transformers.trainer_utils import get_last_checkpoint, EvalLoopOutput
from transformers.utils import check_min_version
from xfact.config.args import ModelArguments, DataTrainingArguments


# from deardr.inference.post_processing import post_process
# from deardr.inference.prefix_decoder import single_document_prefix, multi_document_prefix
# from deardr.inference.scoring import precision, recall, r_precision, macro, f1, max_over_many, average_precision, \
#     recall_corrected, precision_corrected, reciprocal_rank, average_precision_corrected

# from deardr.training.comet_logging_callback import CometTrainingCallback
# from deardr.training.deardr_trainer import DearDrTrainer
from xfact.logs.logs import setup_logging
from xfact.nlp.dataset import XFactDataset
from xfact.nlp.deardr_trainer import DearDrTrainer
from xfact.nlp.post_processing import PostProcessor
from xfact.nlp.reader import Reader
from xfact.nlp.scoring import Scorer
from xfact.registry.module import import_submodules

check_min_version("4.16.0")
logger = logging.getLogger(__name__)





def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    setup_logging(training_args.get_process_log_level())
    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if data_args.package:
        logger.info("Trying to load project packages")
        import_submodules(data_args.package)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    if training_args.should_log and (training_args.do_train or training_args.do_eval):
        # experiment = Experiment(
        #     api_key="J60JdZqL6pTDlG7a81H7o40up",
        #     workspace="j6mes",
        #     project_name=data_args.project_name if data_args.project_name is not None else f"atm_{data_args.dataset_reader}",
        #     experiment_key=data_args.experiment_name if data_args.experiment_name is not None else None
        # )
        experiment = None
        # experiment.log_parameters(training_args.to_dict())
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

    tok_length = len(tokenizer.vocab)

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    dataset_cls = XFactDataset.resolve(data_args.dataset)
    reader = Reader.init(data_args.reader)

    dataset_classes = {
        "train": dataset_cls
    }

    readers = {
        "train": reader
    }

    if data_args.validation_dataset is not None:
        validation_reader = Reader.init(data_args.validation_reader)
        validation_dataset_cls = XFactDataset.resolve(data_args.validation_dataset)

        dataset_classes["validation"] = validation_dataset_cls
        readers["validation"] = validation_reader
    else:
        dataset_classes["validation"] = dataset_classes["train"]
        readers["validation"] = readers["train"]

    loaded_datasets = {
        split: dataset_classes[split](tokenizer,
                                               readers[split].read(path),
                                               max_seq_length,
                                               max_target_length=data_args.max_target_length,
                                               name=split,
                                               test_mode=False
                                               )
        for split, path in data_files.items()
    }

    # Don't use true/false check here as 0 is falsey
    if model_args.dropout_rate is not None:
        config.dropout_rate = model_args.dropout_rate


    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    if len(tokenizer.vocab) > tok_length:
        print("resizing vocab")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        if "train" not in loaded_datasets:
            raise ValueError("--do_train requires a train dataset")

    if training_args.do_eval:
        if "validation" not in loaded_datasets:
            raise ValueError("--do_eval requires a validation dataset")
    #
    #
    data_collator = lambda batch: dataset_cls.collate_fn(model, batch, tokenizer.pad_token_id, data_args.ignore_pad_token_for_loss)

    post_processor = PostProcessor.init("nested", **{"tokenizer": tokenizer, "model": model})
    metrics = Scorer.init("multiset_information_retrieval")



    # logging_callback = CometTrainingCallback(experiment)
    trainer_cls = DearDrTrainer # Maybe do multiple beams with DearDrPredictor but this is SLLOOOOWWWW

    # prefix_decode = single_document_prefix if  isinstance(readers["validation"], PretrainPT) else multi_document_prefix



    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=loaded_datasets["train"] if training_args.do_train else None,
        eval_dataset=loaded_datasets["validation"] if training_args.do_eval else None,
        eval_examples=list(map(itemgetter("instance"), loaded_datasets["validation"].instances)) if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
        post_process_function=post_processor.process_text,
        train_beam=data_args.train_beam,
        # prefix_decode=prefix_decode(tokenizer, model_args.prefix_path),
        # callbacks=[logging_callback]
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

        if trainer.is_world_process_zero():
            experiment.log(payload={"final/" + key: value for key, value in metrics.items()})

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
