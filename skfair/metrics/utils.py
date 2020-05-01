def true_false_positive_negative(conf_matrix):
    """
    Get global true positive, false positive, true negative, false negative
    from confusion matrix

    Args:
       confusion matrix: 2d array-like
    Returns:
       TN, FP, FN, TP
    Assuming a confusion matrix with rows the true classes and columns
    the predicted classes then:
    TP: the diagonal element in the matrix where true and predicted class meet
    FP: the sum of the column of predicted class minus TP
    FN: the sum of the row of true class minus TP
    TN: sum of the matrix minus TP+FP+FN
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    # A second check on the below is appreciated.
    for i in range(conf_matrix.shape[0]):
        tp += conf_matrix[i, i]
        fn += conf_matrix[i, :].sum() - conf_matrix[i, i]
        fp += conf_matrix[:, i].sum() - conf_matrix[i, i]
        tn += (conf_matrix[:i-1, :i-1].sum() + conf_matrix[:i-1, i+1:].sum() +
               conf_matrix[i+1:, :i-1].sum() + conf_matrix[i+1:i+1:].sum())
    return tn, fp, fn, tp
