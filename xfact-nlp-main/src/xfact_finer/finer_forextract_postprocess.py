from xfact.nlp.post_processing import PostProcessor
# import spacy
# from spacy.training import offsets_to_biluo_tags
# nlp = spacy.load("en_core_web_sm")
import re


prompt_mapper = {
    "LOC": "location",
    "PER": "person",
    "ORG": "organization",
    "MISC": "miscellaneous entity"
}


@PostProcessor.register("extracting-finer-139")
class ExtractingProcessor(PostProcessor):
    def process_text(self, examples, features, predictions, trainer):
        #predictions : ndarray [15377,64]
        #predicted : list(15377)
        predicted = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.predictions)]
        actual = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.label_ids)]

        # print(predicted)
        # print(actual)

        for (i,z) in zip(predicted,actual):
            # print(f'predicted:{i}')
            # print(f'actual:{z}')
            if i != z:
                print(f'predicted:{i} & actual:{z} are incorrect')


        return {
            "predicted": predicted,
            "actual": actual
        }

@PostProcessor.register("afterextracting-finer-139")
class ExtractingProcessor(PostProcessor):


    def clean(self, text):
        # Check the internal _bos_token to prevent tokenizer from spamming errors
        med_result = text.replace(self.tokenizer.pad_token, "")\
            .replace(self.tokenizer.unk_token, "<UNK>")\
            .replace(self.tokenizer.eos_token if self.tokenizer._eos_token else "", "")\
            .replace(self.tokenizer.sep_token if self.tokenizer._sep_token else "", "")\
            .replace(self.tokenizer.bos_token if self.tokenizer._bos_token else "", "")

    def split(self,text):
        text.split(", ")


    def process_text(self, examples, features, predictions, trainer):
        #predictions : ndarray [15377,64]
        #predicted : list(15377)
        predicted = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.predictions)]
        actual = [self.clean(p).strip() for p in features.tokenizer.batch_decode(predictions.label_ids)]

        print(predicted)
        print(actual)



        return {
            "predicted": predicted,
            "actual": actual
        }
