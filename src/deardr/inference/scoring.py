from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
def precision(actual, predicted):
    actual = set(actual)
    predicted = set(predicted)

    return (
        sum(1.0 for p in predicted if p in actual) / float(len(predicted))
        if len(predicted)
        else 1.0
    )

def average_precision(actual, predicted):
    return average_precision_score(actual, predicted)

def lrap(actual, predicted):
    """
    Label ranking average precision (LRAP) is the average over each ground truth label assigned
    to each sample, of the ratio of true vs. total labels with lower score.
    If there is exactly one relevant label per sample, label ranking average precision is
    equivalent to the mean reciprocal rank.
    """


    label_ranking_average_precision_score(actual, predicted)

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