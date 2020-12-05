def true_positive(y_true, y_pred):
    for yt , yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    for yt , yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    for yt , yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def fasle_negative(y_true, y_pred):
    for yt , yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def accuracy(y_pred, y_true):
    tp = true_positive(y_pred, y_true)
    fp = false_positive(y_pred, y_true)
    tn = true_negative(y_pred, y_true)
    fn = fasle_negative(y_pred, y_true)
    return (tp) / (tp + fp + tn + fn)


def precision(y_pred, y_true):
    tp = true_positive(y_pred, y_true)
    fp = false_positive(y_pred, y_true)
    tn = true_negative(y_pred, y_true)
    fn = fasle_negative(y_pred, y_true)
    return tp / (tp + fp)


def recall(y_pred, y_true):
    tp = true_positive(y_pred, y_true)
    fp = false_positive(y_pred, y_true)
    tn = true_negative(y_pred, y_true)
    fn = fasle_negative(y_pred, y_true)
    return tp / (tp + fn)


def f1(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    score = ( 2*p*r ) / (p + r)
    return score


def tpr(y_pred, y_true):
    return recall(y_pred, y_true)


def fpr(y_pred, y_true):
    tp = true_positive(y_pred, y_true)
    fp = false_positive(y_pred, y_true)
    tn = true_negative(y_pred, y_true)
    fn = fasle_negative(y_pred, y_true)
    return fp/(tn + fp)
