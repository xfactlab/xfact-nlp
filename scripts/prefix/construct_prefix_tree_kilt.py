import json
import pickle
from argparse import ArgumentParser
from datasets import tqdm
from transformers import AutoTokenizer
from deardr.inference.prefix_decoder import PrefixTree
from deardr.preprocessing import recover


def iterate_pages(jsonl):
    with open(jsonl) as f:

        for line in f:
            instance = json.loads(line)
            yield recover(instance["wikipedia_title"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("tokenizer_name")
    parser.add_argument("--kilt", default="data/kilt/kilt_knowledgesource.json")
    args = parser.parse_args()

    pt = PrefixTree()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
        use_fast=True
    )

    for idx,page in enumerate(tqdm(iterate_pages(args.kilt))):
        added = tokenizer(page)["input_ids"]
        pt.add_string(added)

    with open('wikipedia-titles-kilt-structured-pt.pkl',"wb") as f:
        pickle.dump(pt,f)
        