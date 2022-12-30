from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)

# Should not import Experiment directly here as this will cause Comet to default a new expt

class CometTrainingCallback(TrainerCallback):
    def __init__(self, expt):
        self.expt = expt

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.expt is not None and args.local_rank <= 0:
            logger.info("Saving logs to comet")
            self.expt.log_metrics(step=state.global_step, dic={"train/" + key: value for key, value in state.log_history[-1].items()})
