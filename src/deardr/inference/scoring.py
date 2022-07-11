import numpy as np


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
    for i, p in enumerate(predicted):
        if len(actual) > i and actual[i] == predicted[i] and not found:
            found = True
            r.append(1)
        elif len(actual) > i and actual[i] == predicted[i]:
            continue
        else:
            r.append(0)
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


def f1(actual, predicted):
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
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
