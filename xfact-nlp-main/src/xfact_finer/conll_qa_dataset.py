import re
import logging



from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader



logger = logging.getLogger(__name__)

prompt_mapper = {
    "LOC": "location",
    "PER": "person",
    "ORG": "organization",
    "MISC": "miscellaneous entity"
}

@XFactDataset.register("conll-nerqa-ta-natural")
class NERQADataset(XFactSeq2SeqDataset):
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

        super(NERQADataset, self).__init__(tokenizer=tokenizer,
                                                  instance_generator=instance_generator,
                                                  max_source_length=max_source_length,
                                                  name=name,
                                                  n_obs=n_obs,
                                                  test_mode=test_mode,
                                                  output_prompt=output_prompt,
                                                  streaming=streaming)

        logger.info(f"Output prompt tokens are {self.prompt_tokens}")

    def prepare_src(self, instance):
        return f"What is the {self.prompt_mapper[instance['label']]}? {instance['context']}"

    def prepare_tgt(self, instance):
        if instance['span'] == None:
            return "No answer"
        else:
            return instance['span']



    @staticmethod
    def _split_string(str):
        return " ".join(re.findall(r"[A-Z][^A-Z]*", str))
