import numpy as np
from abc import ABC

from transformers import EvalPrediction

from xfact.registry.registrable import Registrable


class Scorer(Registrable, ABC):
    def __call__(self, actual, predicted, **kwargs):
        raise NotImplementedError()


@Scorer.register("multiset_information_retrieval")
class InformationRetrievalScorer(Scorer):

    def __call__(self, actual, predicted, **kwargs):
        return {
            "macro_recall": macro(max_over_many(recall), actual, predicted),
            "macro_precision": macro(max_over_many(precision), actual, predicted),
            "macro_f1": macro(max_over_many(f1), actual, predicted),
            "macro_r_precision": macro(max_over_many(r_precision), actual, predicted),
            "macro_average_precision": macro(max_over_many(average_precision), actual, predicted),
            "macro_recall_corrected": macro(max_over_many(recall_corrected), actual, predicted),
            "macro_precision_corrected": macro(max_over_many(precision_corrected), actual, predicted),
            "macro_reciprocal_rank": macro(max_over_many(reciprocal_rank), actual, predicted),
            "macro_average_precision_corrected": macro(max_over_many(average_precision_corrected), actual, predicted)
        }


@Scorer.register("information_retrieval")
class InformationRetrievalScorer(Scorer):

    def __call__(self, actual, predicted, **kwargs):
        rec = macro(recall, actual, predicted)
        pr = macro(precision, actual, predicted)
        return {
            "macro_recall": rec,
            "macro_precision": pr,
            "macro_f1": f1(pr,rec),
            "macro_r_precision": macro(r_precision, actual, predicted),
        }


@Scorer.register("classification")
class ClassificationScorer(Scorer):
    def __call__(self, actual, predicted, **kwargs):
        return {
            # "macro_recall": macro(recall, actual, predicted),
            # "macro_precision": macro(precision, actual, predicted),
            # "macro_f1": macro(f1, actual, predicted),
            "em": macro(exact_match, actual, predicted)
        }






@Scorer.register("qa")
class ClassificationScorer(Scorer):
    def __call__(self, actual, predicted, **kwargs):
        return {
            "em": macro(exact_match, actual, predicted)
        }


def exact_match(actual, predicted):
    return 1.0 if actual == predicted else 0.0


def lrp(actual, predicted):
    actual = list(actual)
    predicted = list(predicted)
    rank = 1
    denom = 1
    found = False
    if len(predicted) and len(actual):
        for p in reversed(predicted):
            if p in actual and not found:
                found = True
            elif p not in actual and found:
                rank += 1
                denom += 1
            elif p not in actual:
                denom += 1
    return found, rank, denom


def precision(actual, predicted):
    actual = set(actual)
    predicted = set(predicted)
    return (
        sum(1.0 for p in predicted if p in actual) / float(len(predicted))
        if len(predicted)
        else 1.0
    )


def precision_corrected(actual, predicted):
    found, _, denom = lrp(actual, predicted)
    return 0 if not found else 1. / denom


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(actual, predicted):
    """
    https://gist.github.com/bwhite/3726239
    """
    r = []
    for i, p in enumerate(predicted):
        if len(actual) > i and actual[i] == predicted[i]:
            r.append(1)
        else:
            r.append(0)
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def average_precision_corrected(actual, predicted):
    """
    https://gist.github.com/bwhite/3726239
    """
    r = []
    found = False
    predicted_len = len(predicted) - 1
    for j, _ in enumerate(predicted):
        i = predicted_len - j
        if len(actual) > i and actual[i] == predicted[i] and not found:
            found = True
            r.insert(0, 1)
        elif len(actual) > i and actual[i] == predicted[i]:
            continue
        else:
            r.insert(0, 0)
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def reciprocal_rank(actual, predicted):
    """
    https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/
    I assumed that the lists of evidences in the 1 index of evidences matrix were not sorted by relevance score.
    This program is called with one query, so this program takes maximum of
    reciprocal rank of elements in the actual
    """
    found, rank, _ = lrp(actual, predicted)
    return 0 if not found else 1. / rank


def recall(actual, predicted):
    actual = set(actual)
    predicted = set(predicted)
    return (
        sum(1.0 for a in actual if a in predicted) / float(len(actual))
        if len(actual)
        else 1.0
    )


def recall_corrected(actual, predicted):
    found, _, denom = lrp(predicted, actual)
    return 0 if not found else 1. / denom


def r_precision(actual, predicted):
    actual = set(actual)

    R = len(actual)
    r_predicted = predicted[:R]

    return (
        sum(1.0 for p in r_predicted if p in actual) / float(R)
        if R
        else 1.0
    )


def f1(prec, rec):
    return 2*prec*rec/(prec+rec) if prec+rec > 0 else 0


def macro(method, actual, predicted):
    return sum(method(a,p) for a, p in zip(actual,predicted) if a is not None) / len([a for a in actual if a is not None])


def max_over_many(method):
    def apply_multi(actual_many, predicted):
        scores = []
        for actual in actual_many:
            scores.append(method(actual, predicted))
        return max(scores)
    return apply_multi


