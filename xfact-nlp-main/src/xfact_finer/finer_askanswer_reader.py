import re
import logging
import datasets
from xfact.nlp.dataset import XFactSeq2SeqDataset, XFactDataset
from xfact.nlp.reader import Reader
import os
import pickle

logger = logging.getLogger(__name__)

hierarchical_dict = {
    "A":{"a-1":{"a-1-2","a-1-2"},"a-2":"a-2-1"},
    "B":{"b-1":{"b-1-2","b-1-1"},"b-2":{"b-1"}},
}

def generate_paths(data, path, paths):
    if isinstance(data, dict):
        for key, value in data.items():
            generate_paths(value, path + [key], paths)
    else:
        paths.append((data, path))

    return paths



def wrap_generate_paths(hierarchical_dict):
    paths = []
    generate_paths(hierarchical_dict, [], paths)
    result_dict = {}
    total_list = []

    # Print the resulting paths
    for value, path in paths:
        #     if len(value) == 1:
        if isinstance(value, str):
            total_list.append(path + [value])

        #         result_dict[result[-1]] = result[:1]
        #         print(result_dict)
        else:
            #         result = path + list(value)
            total_list.append(path + list(value))
    #         result_dict[result[-1]] = result[:1]

    #     print(result_dict)
    # print(total_list)

    abc = {}
    for l in total_list:
        abc[l[-1]] = l[:-1]
    return abc


@Reader.register("finer-139-askanswer")
class FinerAnswerReader(Reader):

    def __init__(self, filter_instances=None, test_mode=False):
        self.filter_instances = filter_instances
        self.test_mode = test_mode


    def read(self, path):

        logger.info("reading instances from {}".format(path))
        path, split = path.rsplit(":", maxsplit=1)

        if "pkl" in path:
            with open(path, 'rb') as f:
                self.dataset = pickle.load(f)
                self.un_list_chain = wrap_generate_paths(hierarchical_dict)

                dataset2 = datasets.load_dataset("nlpaueb/finer-139")
                self.tmp_label = dataset2['test'].features["ner_tags"].feature.names

            yield from self.enumerate(self.dataset)
        else:
            self.dataset = datasets.load_dataset(path)
            self.un_list_chain = wrap_generate_paths(hierarchical_dict)
            yield from self.enumerate(self.dataset[split])

    def enumerate(self, file):
        for idx, instance in enumerate(file):
            if self.filter_instances and not self.test_mode and self.filter(instance):
                continue

            yield from self.generate_instances(instance)

            if os.getenv("DEBUG") is not None and idx > 10:
                break

    def build_question(hierarchical_dict, list_chain, depth, context, qa):

        if qa == "end_wq":  # end with question

            starting_question = """The following is a conversation between a human and an AI. The AI only answers based on the requested from the user and doesn't provide details. \If the answer doesn't exist, AI says "**NONE**"\n###HUMAN: I’m going to give you (a) sentence(s) with one span. I will let you to identify the category that a word belongs to based on the context. There will only be {0} types of entities: {1}."{2}" Give an Answer without explanation\n###AI:The answer is """

            starting_question = starting_question.format(len(hierarchical_dict), list(hierarchical_dict.keys()),
                                                         context)

            continuing_answer = "{0}"
            continuing_question = """\n###HUMAN: As it belongs to \'{0}\', I will let you to identify the sub category of \'{0}\' that a word belongs to based on the context.\
            There will only be {1} types of entities: {2}.\
            \"{3}\" Give an Answer without explanation.\n###AI:The answer is """
            # depth has to be len(listchain)-1

            if depth >= len(list_chain):
                raise ValueError(
                    f"Depth ({depth}) cannot be greater than or equal to the length of listchain ({len(list_chain)}).")

            for i in range(depth):
                if i > len(list_chain):
                    break  # We've exhausted all keys in listchain.

                key = list_chain[i]  # Key to access current level of hierarchical_dict.
                if isinstance(hierarchical_dict, dict) and key in hierarchical_dict:
                    # Append to question string.
                    starting_question += continuing_answer.format(key)

                    # Update hierarchical_dict and choices for next level.
                    hierarchical_dict = hierarchical_dict[key]
                    choices = list(hierarchical_dict.keys()) if isinstance(hierarchical_dict,
                                                                           dict) else hierarchical_dict
                    length = len(choices)

                    # Append next part of question.

                    starting_question += continuing_question.format(continuing_answer.format(key), length, choices,
                                                                    context)

            return starting_question

    # Example usage:
    # context = "japanese is"
    # qa = "end_wq"
    # depth = 2
    # result = build_question(hierarchical_dict, list_chain, depth, context, qa)
    # print(result)





        """
        #build_question 에 들어가기 위한 요소들을 original dict 에 첨가해주고(e.g., depth),
        #depth 만큼 반복해준다.
        each_listchain = listchain[original_dict["label"]]
        depth = len(each_listchain)


        for s, l in enumerate(each_listchain):
            if s == 0:  # A
                original_dict["qa"] = "startanswer"
                original_dict["question"] = problemchain[0]  # options asked
                original_dict["answer"] = instanchain[0]  # answer
                yield original_dict

            if s == depth:
                original_dict["qa"] = "finishanswer"
                original_dict["question"] = problemchain[0]  # options asked
                original_dict["answer"] = instanchain[0]  # answer

                yield original_dict

            else:  # Q / A
                for qa in ["u_gen_question", "u_gen_answer"]:
                    original_dict["qa"] = qa

                    original_dict["question"] = problemchain[l]  # options asked
                    original_dict["answer"] = instanchain[l]  # answer
                    yield original_dict

                    original_dict["qa"] = qa
                    original_dict["answer"] = instanchain[0]  # answer

                    original_dict["question"] = problemchain[0]  # options asked
                    original_dict["chain"] = listchain[0]

                    original_dict["chain"] = depth[l + 1]  # price
                    original_dict["answer"] = depth[:l + 1]
                    yield original_dict
                    
        """


    def chain_yield(self, original_dict, list_chain):

        depth = len(list_chain)

        for i_depth in range(depth):
            original_dict["list_chain"] = list_chain
            original_dict["i_depth"] = i_depth
            yield original_dict

    def generate_instances(self, instance):

        if not any(instance["ner_tags"]):  # no answers. -> make questions with no answers
            return
        buffer = []
        buffer_tag = None
        labels = [self.tmp_label[idx] for idx in instance["ner_tags"]]
        # labels = [self.dataset['train'].features["ner_tags"].feature.names[idx] for idx in instance["ner_tags"]]
        for token, tag in zip(instance["tokens"], labels):

            # If we have a buffer, then we should return the tag if we encounter O or B-
            if (tag == "O" or tag.startswith("B-")) and buffer:

                original_dict = {
                    "context": " ".join(instance["tokens"]),
                    "span": " ".join(buffer),
                    "label": buffer_tag,
                    "instance": instance
                    #buffer_tag으로 self.listchain 에서 찾도록 하자 아니다. chain_yield 에서 시키자
                }
                list_chain = self.un_list_chain[buffer_tag]

                yield from self.chain_yield(original_dict, list_chain)

                buffer = []
                buffer_tag = None

            # If it starts with a B- we add to buffer
            if tag.startswith("B-"):
                buffer_tag = tag.replace("B-", "")
                buffer.append(token)

            # If it starts with a I- we continue adding to buffer
            elif tag.startswith("I-"):
                buffer.append(token)

        if buffer:
            original_dict = {
                "context": " ".join(instance["tokens"]),
                "span": " ".join(buffer),
                "label": buffer_tag,
                "instance": instance
            }

            list_chain = self.un_list_chain[buffer_tag]

            yield from self.chain_yield(original_dict, list_chain)




if __name__ == "__main__":
    reader = FinerAnswerReader()
    hierarchical_dict = {'price': {'priceA': {'PriceA-1', 'PriceA-2'}, 'priceB': 'PriceB-1'},
 'Amount': {'amountA': {'amountA-1': {'amountA-1-a', 'amountA-1-b'}},
  'amountB': 'amountB-1'}}
    for j, a in enumerate(reader.read("nlpaueb/finer-139:test")):
        print(a)


