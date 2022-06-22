from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class CometTrainingCallback(TrainerCallback):
    def __init__(self, expt):
        self.expt = expt

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.expt is not None:
            self.expt.log_metrics({"train/" + key: value for key, value in state.log_history[-1].items()},
                               step=state.global_step)