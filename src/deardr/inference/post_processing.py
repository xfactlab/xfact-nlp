import re
from collections import OrderedDict

from deardr.dataset.page_title_prediction_dataset import PageTitlePredictionDataset
from deardr.preprocessing import recover


def post_process_test_multibeam(examples, features, predictions, trainer):


    predicted = [
        [

            [r.strip() for r in

             re.split(features.tokenizer.sep_token+ "|"+ ("|".join(features.tokenizer.all_special_tokens_extended[4:])),
             p.replace(features.tokenizer.pad_token, "").
                 replace(features.tokenizer.unk_token, "").
                 replace(features.tokenizer.eos_token, ""))] for p in features.tokenizer.batch_decode(pred)
        ] for pred in predictions.predictions]
    results = []
    for inst in predicted:
        resultset = OrderedDict()

        for beam in inst:
            resultset.update({k: 1 for k in beam})
        results.append(list(resultset.keys()))

        # resultset = Counter()
        # for beam in inst:
        #     resultset.update({b:1 for b in beam})
        # results.append([k[0] for k in resultset.most_common(10)])
    assert len(results) == len(examples) == len(predictions.predictions)
    yield from zip(results, examples, predictions.predictions)



def get_pages_fever(evidence_sets):
    ev_sets = list()
    for e in evidence_sets:
        if any([es[2] is not None for es in e]):

            ev_sets.append([])
            for _,_,page,line in e:
                if page is None:
                    continue

                ev_sets[-1].append(recover(page))

    return list(ev_sets)

def post_process(examples, features, predictions, trainer):
    predicted = [
        [r.strip() for r in
         p.
             replace(features.tokenizer.pad_token, "").
             replace(features.tokenizer.unk_token, "").
             replace(features.tokenizer.eos_token, "").
             split(PageTitlePredictionDataset.sep_token)] for p in
        features.tokenizer.batch_decode(predictions.predictions)
    ]

    actual = [
        [r.strip() for r in p.
            replace(features.tokenizer.pad_token, "").
            replace(features.tokenizer.unk_token, "").
            replace(features.tokenizer.eos_token, "").
            split(PageTitlePredictionDataset.sep_token)] for p in
        features.tokenizer.batch_decode(predictions.label_ids)
    ]

    # with open("predictions.jsonl","w+") as f:
    #     for p, a in zip(predicted,actual):
    #         f.write(json.dumps({"predicted": p, "actual": a})+"\n")

    return {
        "predicted": predicted,
        "actual_flat": actual,
        "actual": [f['nested_entities'] for f in features.instances]
    }

