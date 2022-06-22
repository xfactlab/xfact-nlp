import json
from collections import OrderedDict

from datasets import tqdm
from drqa.retriever.utils import normalize

from deardr.fever.redirects import get_redirects, recursive_redirect_lookup
from deardr.fever.document_database import FEVERDocumentDatabaseIterable

import logging

logger = logging.getLogger(__name__)


def recover(text):
    return normalize(text.
                     replace("_"," ").
                     replace("-LRB-","(").
                     replace("-RRB-",")").
                     replace("-LSB-","[").
                     replace("-RSB-","]")
                     ).strip()

def unrecover(text):
    return normalize(text.
                     replace(" ","_").
                     replace("(","-LRB-").
                     replace(")","-RRB-").
                     replace("[","-LSB-").
                     replace("]","-RSB-")
                     ).strip()


def resolve_redirects(chunk):
    if db.get_doc_text(unrecover(chunk)) is not None:
        return chunk

    try:
        result = recursive_redirect_lookup(rd, chunk)
        if result is not None and result != chunk:
            return result
    except IndexError:
        print("Lookup failed for {}".format(chunk))
        logger.warning("Lookup failed for {}".format(chunk))

    return None

def extract_entities(bits):
    if not len(bits):
        return []

    slice = [i for i in range(len(bits)) if i%2 == 1]
    d = OrderedDict()
    d.update({resolve_redirects(bits[i]):1 for i in slice})
    return [a for a in d.keys() if a is not None]

def iterate_lines(db):
    for page, lines in db.iter_all_doc_lines():
        page = recover(page)

        if page.startswith("List of ") or page.startswith("Category:") or page.startswith("Lists of") or page.endswith("(disambiguation)"):
            continue


        for line in lines.split("\n"):

            line_text = line.split("\t")

            if len(line_text) > 2 and len(line_text[1].strip()):
                if line_text[1].startswith("|"):
                    continue

                if len(line_text[1].split(" ")) < 3:
                    continue

                try:
                    position = int(line_text[0])
                    yield page, position, recover(line_text[1]), extract_entities(line_text[2:])

                except ValueError as e:
                    continue

if __name__ == "__main__":
    dbpath = "data/wikipedia/fever.db"
    db= FEVERDocumentDatabaseIterable(dbpath)

    rd = get_redirects("data/wikipedia/redirects.txt")

    with open("data/generated/all.jsonl","w+") as f:
        for idx,(page,position, line, entities) in tqdm(enumerate(iterate_lines(db))):
            f.write(json.dumps({"page": page, "position":position, "line": line, "entities": entities})+ "\n")
            # if idx > 100000:
            #     break