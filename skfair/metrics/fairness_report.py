from functools import partial
from collections import defaultdict


from sklearn.metrics import (f1_score, precision_score,
                             accuracy_score, recall_score)
from terminaltables import AsciiTable
import pandas as pd

from skfair.metrics import false_discovery_score, false_positive_score


DEFAULT_METRICS = {
    "TPR": partial(recall_score, average="micro"),
    "FPR": false_positive_score,
    "PPVR": partial(precision_score, average="micro"),
    "FDR": false_discovery_score,
    "ACC": accuracy_score,
    "F1": partial(f1_score, average="micro")
}


def _yield_metrics(metrics):
    if type(metrics) == list:
        for metric in metrics:
            yield metric.__name__, metric
    elif type(metrics) == dict:
        for metric_name, metric in metrics.items():
            yield metric_name, metric
    else:
        raise ValueError("metrics should be either a list or a dict")


def create_table_report(report_dict):
    headers = [v.keys() for k, v in report_dict.items()][0]
    table_data = [[""] + list(headers)]
    for group_name in report_dict:
        table_data.append([group_name] + [f"{v:.3f}" for v in report_dict[group_name].values()])
    table = AsciiTable(table_data)
    return table.table


def classification_fairness_report(y_true, y_pred, groups, group_names=None,
                                   labels=None, output="text",
                                   metrics=DEFAULT_METRICS):
    grouped_data = defaultdict(list)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        group_key = group_names[i] if group_names else groups[i]
        grouped_data[group_key].append((y_t, y_p))

    report_dict = defaultdict(dict)
    for group_name, group_data in grouped_data.items():
        y_true_group, y_pred_group = zip(*group_data)

        for metric_name, metric in _yield_metrics(metrics):
            report_dict[group_name][metric_name] = metric(y_true_group, y_pred_group, labels)
        report_dict[group_name]["Support"] = len(group_data)

    if output == "dict":
        return report_dict
    if output == "pandas":
        return pd.DataFrame(report_dict)
    return create_table_report(report_dict)
