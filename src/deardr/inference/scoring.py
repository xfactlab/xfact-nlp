from sklearn.metrics import average_precision_score, label_ranking_average_precision_score

def precision(actual, predicted):
    actual = set(actual)
    predicted = set(predicted)
    return (
        sum(1.0 for p in predicted if p in actual) / float(len(predicted)) if len(predicted) else 1.0
    )


def average_precision(actual, predicted):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    AP is averaged over all categories. Traditionally, this is called mAP. mAP score is calculated by taking the mean AP over all classes.
    """
    actual = set(actual)
    predicted = set(predicted)
    return average_precision_score(actual, predicted)


def mean_reciprocal_rank(actual, predicted):
    """
    https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/
    I assumed that the lists of evidences in the 1 index of evidences matrix were not sorted by relevance score.
    This program is called with one query, so this program takes maximum of
    reciprocal rank of elements in the actual
    """
    actual = list(actual)
    predicted = list(predicted)
    maximum = 0
    if len(predicted):
        for p in actual:
            if p in predicted:
                maximum = max(maximum, (1.0/(predicted.index(p)+1)))
    return maximum

def recall(actual, predicted):
    actual = set(actual)
    predicted = set(predicted)
    return (
        sum(1.0 for a in actual if a in predicted) / float(len(actual))
        if len(actual)
        else 1.0
    )


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
