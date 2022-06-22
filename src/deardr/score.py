import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
import numpy as np
from deardr.inference.scoring import max_over_many, recall, r_precision
from deardr.preprocessing import recover


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

def get_pages_hover(e):
    ev = set()
    for page,line in e:
        ev.add(recover(page))

    return [list(ev)]
#
#


def get_titles(list_of_dicts):
    return list(OrderedDict({dic["title"]:1 for dic in list_of_dicts}).keys())

def get_pages_kilt(loaded):
    return [get_titles(output["provenance"]) for output in loaded if "provenance" in output]


def score_file(in_file):

    recalls = defaultdict(list)
    fsups = defaultdict(list)
    r_prec = list()

    with open(in_file) as g:
        for line in g:
            instance = json.loads(line)

            if "label" in instance and instance["label"] == "NOT ENOUGH INFO":
                continue

            predicted = [recover(a) for a in instance['predicted_documents']]
            if "evidence" in instance:
                pages = get_pages_fever(instance["evidence"])
                pages = list(set(tuple(a) for a in pages))

            elif "output" in instance:
                pages = get_pages_kilt(instance["output"])

            elif "supporting_facts" in instance:
                pages = get_pages_hover(instance["supporting_facts"])

                # if "supporting_facts" in orig:
                #     pages = set(get_pages_hover(orig["supporting_facts"]))
                #     predicted = instance['predicted_pages']
                # elif "output" in orig:
                #     p
                # else:
                #     pages = get_pages_fever(orig["evidence"])
                #     predicted = instance['predicted_pages']

            for k in range(1, 11):
                predicted_pages = set(predicted[:k])
                recalls[k].append(max_over_many(recall)(pages, predicted_pages))

            r_prec.append(max_over_many(r_precision)(pages, predicted))

    return {"recall@k":[np.mean(a) for a in recalls.values()],"r_prec" : np.mean(r_prec)}

if __name__ == "__main__":



    parser = ArgumentParser()
    parser.add_argument("in_file")
    args = parser.parse_args()

    results = score_file(args.in_file)

    if "fever" in args.in_file:
        task = "fever"
    else:
        task = "hover"

    with open(os.path.dirname(args.in_file) + "/" + f"recall_{task}.json","w+") as f:
        json.dump(results,f)

    tags = re.split(r"/|,", args.in_file)

    kvp = [tag.split("=") for tag in tags if "=" in tag]
    kvp.append(["file",tags[-1]])
    kvp.append(["recall@10", results["recall@k"][10]])
    kvp.append(["r_prec", results["r_prec"]])

    with open("results.jsonl", "a+") as f:
        jd = {k:v for k,v in kvp}
        print(jd)
        f.write(json.dumps(jd)+"\n")


