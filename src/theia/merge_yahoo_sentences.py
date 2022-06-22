import json
from collections import defaultdict
from glob import glob


def get_instances(fp):

    for line in fp:
        instance = json.loads(line)
        predicted = set([doc.strip() for doc in instance["predicted_documents"] if doc.strip()])

        yield instance['instance']['ticker'], predicted


def extract_themes_from_file(file):
    with open(f"data/{file}") as f:
        extracted_themes = defaultdict(set)

        for ticker, predicted in get_instances(f):
            extracted_themes[ticker].update(predicted)

        return extracted_themes


if __name__=="__main__":

    in_files = ["dd_pt_hl2_early_yahoo.jsonl", "dd_pt_hl2_late_yahoo.jsonl", "multi_yahoosent_yahoo_split.jsonl.jsonl"]
    out_files = ["yahoo_extracted_deardr_hl_e.jsonl", "yahoo_extracted_deardr_hl_l.jsonl", "yahoo_extracted_deardr_pthl.jsonl"]

    for file_name, output_filename in zip(in_files, out_files):
        print(file_name)
        all_themes = extract_themes_from_file(file_name)

        with open(f"data/{output_filename}","w+") as f:
            for ticker, themes in all_themes.items():
                f.write(json.dumps({"ticker":ticker, "themes": list(themes)})+"\n")
