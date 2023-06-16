from typing import List, Any, Dict, Tuple
import datasets
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle
import pandas as pd
import copy
import itertools
import random
import sys
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import os

random.seed(2024)


class DataGeneratorBase:
    def __init__(self, opt):
        self.opt = opt

    def gen_data(self, raw_data, label_set):
        raise NotImplementedError


class MiniIncludeGenerator(DataGeneratorBase):
    """
    Data generator for the situation that one sample can have multiple label, for example: a sentence may have multiple
    slot or belong to multiple classes.
    """

    def __init__(self, opt):
        super(MiniIncludeGenerator, self).__init__(opt)

    def gen_data(self, raw_data, label_set):

        episodes = []
        domain_data = raw_data
        label_bucket, d_id2label = self.get_label_bucket(domain_data, label_set)
        # all_labels, del_labels = self.get_all_label_set(label_bucket)
        # removed_labels = list(label_id.values())

        # for e in label_set:
        #     removed_labels.remove(e)

        # label_bucket = self.del_samples_in_label_bucket(label_bucket, removed_labels)

        for episode_id in range(10):
            # 이 부분 일단 pass
            support_set = self.sample_support_set(domain_data, label_set, label_bucket, d_id2label)
            # query_set = self.get_query_set(domain_data if self.opt.dup_query else remained_data, label_set)
            episodes.append(support_set)

        return episodes

    def del_samples_in_label_bucket(self, label_bucket: Dict[str, List[int]], del_labels: List[str]):
        """
        some label has not enough samples,
        so these samples with that label should be removed from samples of other remained labels
        :param label_bucket:
        :param del_labels:
        :return:
        """
        del_samples = []
        for label in del_labels:
            del_samples.extend(label_bucket[label])
        del_samples = list(set(del_samples))

        for label in label_bucket.keys():
            for sample_id in del_samples:
                if sample_id in label_bucket[label]:
                    label_bucket[label].remove(sample_id)
        return label_bucket

    def get_label_bucket(self, domain_data: dict, label_set: list) -> Tuple[Dict[str, List[int]], List[List[str]]]:
        label_field = 'ner_tags'
        label_bucket, d_id2label = {}, []
        ner_tags_data = domain_data[label_field]

        for d_id in domain_data['id']:
            d_id = int(d_id)
            labels = list(set(ner_tags_data[d_id]))  # all appeared label within a sample
            for label in labels:  # add data id into buckets of labels
                if label in label_bucket:
                    label_bucket[label].append(d_id)
                else:
                    label_bucket[label] = [d_id]
            d_id2label.append(labels)
        return label_bucket, d_id2label

    def get_all_label_set(self, label_bucket: Dict[str, List[int]]) -> List[str]:
        """ filtering out bad labels & get all label """
        all_labels = []
        del_labels = []
        for label in label_bucket:
            if len(label_bucket[label]) < self.opt.min_label_appear and self.opt.min_label_appear > 1:
                print(f'{label} turned out to be bad label. {len(label_bucket[label])}')
                del_labels.append(label)
                continue
            else:
                all_labels.append(label)
        return all_labels, del_labels

    def sample_support_set(self, data_part, label_set, label_bucket, d_id2label):
        """
        Given data part, sampling k-shot data for n-way with Mini-including Algorithm
        :param data_part: { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        :param label_set: N-way label set for data sampling
        :param label_bucket:  dict, {slot_name:[data_id]}
        :param d_id2label: list, {data id: label set}
        :return: result few shot data part { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                 remained data part { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        """

        shot_num = opt.shot
        tmp_label_bucket = copy.deepcopy(label_bucket)
        shot_counts = {ln: 0 for ln in label_set}
        selected_data_ids = []

        ''' Step0: duplicate the label id if some label samples number is smaller than support shot num'''
        print('step0')

        # import pdb;pdb.set_trace()

        for label in label_set:
            if len(tmp_label_bucket[label]) < shot_num:
                tmp_sample_ids = tmp_label_bucket[label]
                while len(tmp_sample_ids) < shot_num:
                    dup_sample_id = random.choice(tmp_sample_ids)
                    tmp_sample_ids.append(dup_sample_id)
                tmp_label_bucket[label] = tmp_sample_ids
        # print({k: len(v) for k, v in tmp_label_bucket.items()})

        ''' Step1: Sample learning shots, and record the selected data's id '''
        print('step1')
        for label in label_set:
            while shot_counts[label] < shot_num:
                sampled_id = random.choice(
                    tmp_label_bucket[label])  # sample 1 data from all data contains current label.
                self.update_label_bucket(sampled_id, tmp_label_bucket, d_id2label)  # remove selected data ids
                self.update_shot_counts(sampled_id, shot_counts,
                                        d_id2label)  # +1 shot for all labels of sampled data
                selected_data_ids.append(sampled_id)
        num_before = len(selected_data_ids)

        ''' Step2: Remove excess learning shots '''
        print('step2')
        # import pdb;pdb.set_trace()
        for d_id in selected_data_ids:
            to_be_removed_labels = d_id2label[d_id]
            can_remove = True
            for label in to_be_removed_labels:
                if label in shot_counts and shot_counts[label] - 1 < shot_num:
                    can_remove = False
                    break
            if can_remove:
                if random.randint(1,
                                  100) < self.opt.remove_rate:  # Not to remove all removable data to give chances to extreme cases.
                    selected_data_ids.remove(d_id)
                    for label in to_be_removed_labels:
                        if label in shot_counts:
                            shot_counts[label] -= 1
        num_after = len(selected_data_ids)

        ''' Pick data item by selected id '''
        # selected_data = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'slu_labels': []}
        selected_data = []
        # selected_data = datasets.with_format(type="torch", columns=["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"])

        for d_id in range(len(data_part['id'])):

            # s_in, s_out, lb, slu_lb = data_part['seq_ins'][d_id], data_part['seq_outs'][d_id], data_part['labels'][
            #     d_id], data_part['slu_labels'][d_id]
            if d_id in selected_data_ids:  # decide where data go
                repeat_num = selected_data_ids.count(d_id)

                while repeat_num:  # label 개수가 적은 것은 두번 이상 들어갈 수 있다고 가정했어서

                    if len(selected_data) == 0:
                        # selected_data = data_part[d_id]
                        # import pdb;pdb.set_trace()
                        print(data_part[d_id]['id'])

                        selected_data = Dataset.from_dict(
                            {'id': [str(data_part[d_id]['id'])], "tokens": [data_part[d_id]['tokens']],
                             "ner_tags": [data_part[d_id]["ner_tags"]]})
                    else:
                        # add = data_part[d_id]
                        selected_data = selected_data.add_item(data_part[d_id])

                    repeat_num -= 1

        print(selected_data)
        selected_labels = list(itertools.chain.from_iterable(selected_data['ner_tags']))
        label_shots = Counter(selected_labels)
        error_shot = False
        for lb, s in label_shots.items():
            if (s < self.opt.shot) & (lb in label_set):
                error_shot = True
                print("Error: Lack shots of intent:", lb, s)

        # import pdb;pdb.set_trace()

        if error_shot:
            raise RuntimeError('Error in support shot number of intent.')

        return selected_data

    def update_label_bucket(self, sampled_id, tmp_label_bucket, d_id2label):
        """ remove selected data ids """
        labels = d_id2label[sampled_id]
        for label in labels:
            if sampled_id in tmp_label_bucket[label]:
                tmp_label_bucket[label].remove(sampled_id)

    def update_shot_counts(self, sampled_id, shot_counts, d_id2label):
        """ update shots count for all selected number appeared in sampled data """
        labels = d_id2label[sampled_id]
        for label in labels:
            if label in shot_counts:
                shot_counts[label] += 1


if __name__ == '__main__':

    print('Start unit test.')

    import argparse

    parse = argparse.ArgumentParser()
    opt = parse.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', required=False, default=5, type=int)
    parser.add_argument('--data', required=False, default="finer")
    parser.add_argument('--save', required=False, default=True)
    parser.add_argument('--min_label_appear', default=3)
    parser.add_argument('--remove_rate', default=90)
    # parser.add_argument('--min_label_appear', default=90)

    opt = parser.parse_args()

    hugging_dict = {
        "finer": "nlpaueb/finer-139",
        "conll2003": "conll2003",
        "ontonotes": "conll2012_ontonotesv5"
    }
    data_train = datasets.load_dataset(hugging_dict[opt.data], split='train')

    if opt.data == "finer":
        label_id = {}
        label_set = []
        for i, j in enumerate(data_train.features['ner_tags'].feature.names):
            label_id[str(j)] = i
            if j[0] == 'B':
                label_set.append(i)


    elif opt.data == "conll2003":

        label_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                    'I-MISC': 8}
        label_set = [1, 3, 5, 7]

    # new_data = [key:value for key, value in data.features.items()]

    # idxes = np.random.permutation(range(len(data_train)))
    # data_tag_names = label_id.values()
    # data_tag_names = {i: (x[2:] if len(x) > 2 else x) for i, x in enumerate(data_tag_names)}
    # tag_counts = {x: 0 for x in set(data_tag_names.values())}
    # begin = [data_train.features['ner_tags'].feature.names[i] for i in label_set ]
    # tag_counts = dict((x[2:], 0) for x in begin)
    # saved_item = []
    # import pdb;pdb.set_trace()
    gen_loader = MiniIncludeGenerator(opt)
    result_data = gen_loader.gen_data(data_train, label_set)

    dump_path = f'data/{opt.data}/{opt.data}_{opt.shot}shot'

    if os.path.exists(dump_path):
        pass
        # raise ValueError("Output directory () already exists and is not empty.")
    else:
        os.makedirs(dump_path, exist_ok=True)

    for j in range(10):
        concat_path = os.path.join(dump_path, f'{j}.pkl')
        with open(concat_path, 'wb') as f:
            print(result_data)
            pickle.dump(result_data[j], f)

