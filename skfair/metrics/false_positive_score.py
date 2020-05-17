from sklearn.metrics import confusion_matrix

from skfair.metrics.utils import true_false_positive_negative


def false_positive_score(y_true, y_pred, labels=None):
    """
    Args:
       y_true: 1d array-like, ground truth of target labels
       y_pred: 1d array-like, predictions of target labels
       labels: labels to be included in the calculation of the score
    Returns:
       score: float

    Assuming a confusion matrix with rows the true classes and columns
    the predicted classes then:
    TP: the diagonal element in the matrix where true and predicted class meet
    FP: the sum of the column of predicted class minus TP
    FN: the sum of the row of true class minus TP
    TN: sum of the matrix minus TP+FP+FN

    False positive rate then equals to FP / (TN + FP)
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = true_false_positive_negative(conf_matrix)
    eps = 1e-10
    return fp / (fp + tn + eps)
