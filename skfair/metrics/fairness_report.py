from collections import defaultdict

from sklearn.metrics import confusion_matrix
import numpy as np

def fairness_report(y_true, y_pred, groups, group_names=None):
    unique_groups = set(groups)

    grouped_data = defaultdict(list)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        group_key = group_names[i] if group_names else groups[i]
        grouped_data[group_key].append((y_t, y_p))

    outputs = []
    for group_name, group_data in grouped_data.items():
        y_true_group, y_pred_group = zip(*group_data)

        conf_matrix = confusion_matrix(y_true_group, y_pred_group, labels=[0,1])
        tn, fp, fn, tp = conf_matrix.ravel()
        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        outputs.append({
            'Group': group_name,
            'TPR': recall,
            'FPR': fp / (fp + tn + eps),
            'PPVR': precision,
            'FDR': fp / (tp + fp + eps),
            'ACC': (tp + tn) / (tp + fp + tn + fn + eps),
            'F1': 2 * precision * recall / (precision + recall + eps),
            'Support': len(group_data)
        })

    headers = outputs[0].keys()

    # almost identical to from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_classification.py
    width = max(len(cn) for cn in grouped_data.keys())
    head_fmt = '{:>{width}s} ' + ' {:>10}' * (len(headers) - 1)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>10.{digits}f}' * (len(headers)-2) + ' {:>10}\n'
    for row in outputs:
        row_values = row.values()
        report += row_fmt.format(*row_values, width=width, digits=3)

    report += '\n'
    return report

if __name__ == '__main__':
    y_true = np.array([0,0,1,1,1])
    y_pred = np.array([0,0,0,1,1])
    groups = np.array([1,0,1,0,1])
    group_names = ["b", "r", "b", "r", "b"]

    report = fairness_report(y_true, y_pred, groups, group_names)
    print(report)
