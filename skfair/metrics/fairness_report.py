from collections import defaultdict

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def true_false_positive_negative(conf_matrix):
    # conf matrix: rows are true, cols are pred
    if conf_matrix.shape == (2, 2):  # binary case
        tn, fp, fn, tp = conf_matrix.ravel()
        return tn, fp, fn, tp
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


def classification_fairness_report(y_true, y_pred, groups, group_names=None,
                                   labels=None, output="text"):
    grouped_data = defaultdict(list)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        group_key = group_names[i] if group_names else groups[i]
        grouped_data[group_key].append((y_t, y_p))

    report_dict = {}
    for group_name, group_data in grouped_data.items():
        y_true_group, y_pred_group = zip(*group_data)

        conf_matrix = confusion_matrix(y_true_group, y_pred_group, labels=labels)
        tn, fp, fn, tp = true_false_positive_negative(conf_matrix)
        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        report_dict[group_name] = {
            'TPR': recall,
            'FPR': fp / (fp + tn + eps),
            'PPVR': precision,
            'FDR': fp / (tp + fp + eps),
            'ACC': (tp + tn) / (tp + fp + tn + fn + eps),
            'F1': 2 * precision * recall / (precision + recall + eps),
            'Support': len(group_data)
        }

    if output == "dict":
        return report_dict
    elif output == "pandas":
        return pd.DataFrame(report_dict)

    headers = [v.keys() for k, v in report_dict.items()][0]

    # almost identical to https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_classification.py
    width = max(len(cn) for cn in grouped_data.keys())
    head_fmt = '{:>{width}s} ' + ' {:>10}' * (len(headers))
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>10.{digits}f}' * (len(headers)-1) + ' {:>10}\n'
    for row_name, row in report_dict.items():
        row_values = [row_name] + list(row.values())
        report += row_fmt.format(*row_values, width=width, digits=3)

    report += '\n'
    return report


if __name__ == '__main__':
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    groups = np.array([1, 0, 1, 0, 1])
    group_names = ["b", "r", "b", "r", "b"]

    report = classification_fairness_report(y_true, y_pred, groups, group_names)
    print(report)

    report = classification_fairness_report(y_true, y_pred, groups, group_names, output="dict")
    print(report)

    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 2, 0])
    groups = np.array([1, 0, 1, 0, 1, 0, 1])
    group_names = ["b", "r", "b", "r", "b", "r", "b"]

    report = classification_fairness_report(y_true, y_pred, groups, group_names, labels=[0, 1, 2])
    print(report)
