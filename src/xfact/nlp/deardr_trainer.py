from typing import Dict, List, Optional, Union, Any, Tuple

import datasets
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import is_torch_tpu_available, Trainer, is_datasets_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class XFactClsTrainer(Trainer):
    def __init__(self, sampler=None, eval_examples=None,post_process_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output, self)

            metrics = self.compute_metrics(**eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            metrics.update(output.metrics)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].
    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.
    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #
    #     train_dataset = self.train_dataset
    #     data_collator = self.data_collator
    #     if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #         train_dataset = self._remove_unused_columns(train_dataset, description="training")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
    #
    #     if isinstance(train_dataset, torch.utils.data.IterableDataset):
    #         if self.args.world_size > 1:
    #             train_dataset = IterableDatasetShard(
    #                 train_dataset,
    #                 batch_size=self._train_batch_size,
    #                 drop_last=self.args.dataloader_drop_last,
    #                 num_processes=self.args.world_size,
    #                 process_index=self.args.process_index,
    #             )
    #
    #         return DataLoader(
    #             train_dataset,
    #             batch_size=self.args.per_device_train_batch_size,
    #             collate_fn=data_collator,
    #             num_workers=self.args.dataloader_num_workers,
    #             pin_memory=self.args.dataloader_pin_memory,
    #         )
    #
    #     train_sampler = self._get_train_sampler()
    #
    #     return DataLoader(
    #         train_dataset,
    #         batch_size=self._train_batch_size,
    #         sampler=train_sampler,
    #         collate_fn=data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #         worker_init_fn=seed_worker,
    #     )




class DearDrTrainer(Seq2SeqTrainer):
    def __init__(self, *args, generation_max_length=64, prefix_decode=None, eval_examples=None, post_process_function=None, train_beam=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.args.predict_with_generate = True
        self.args.generation_max_length = generation_max_length
        self.args.generation_num_beams = train_beam
        self.prefix_decode = prefix_decode

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "prefix_allowed_tokens_fn": self.prefix_decode
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics


        if self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output, self)

            metrics = self.compute_metrics(**eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            metrics.update(output.metrics)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test", sample_temperature=1.0, max_length=30, num_beams=10):
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=False,
                ignore_keys=ignore_keys
            )

        finally:
            self.compute_metrics = compute_metrics

        predictions = self.post_process_function(predict_examples, predict_dataset, output, self)
        return predictions


class DearDrTestMultiBeamPredictor(DearDrTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.per_device_eval_batch_size = 32

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length,
            "num_beams": 10,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "prefix_allowed_tokens_fn": self.prefix_decode
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            # do_sample=True,
            # temperature=1.0,
            attention_mask=inputs.get("attention_mask", None),
            num_return_sequences=10,
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])


        return (None, generated_tokens.view(len(generation_inputs), 10,-1), None)




