import pickle
from argparse import ArgumentParser

from datasets import tqdm
from transformers import AutoTokenizer

from deardr.fever.document_database import FEVERDocumentDatabaseIterable
from deardr.inference.prefix_decoder import PrefixTree
from deardr.preprocessing import recover


def iterate_pages(db):
    for page, lines in db.iter_all_doc_lines():
        yield recover(page)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("tokenizer_name")
    parser.add_argument("--db", default="data/wikipedia/fever.db")
    args = parser.parse_args()

    pt = PrefixTree()
    db= FEVERDocumentDatabaseIterable(args.db)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
        use_fast=True
    )

    for idx,page in enumerate(tqdm(iterate_pages(db))):
        added = tokenizer(page)["input_ids"]
        pt.add_string(added)

    with open('wikipedia-titles-structured-pt.pkl',"wb") as f:
        pickle.dump(pt,f)