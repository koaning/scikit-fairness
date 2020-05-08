from skfair.metrics.fairness_report import classification_fairness_report

import numpy as np

from collections import defaultdict


def test_classification_fairness_report():
    y = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    groups = [0, 1, 0, 1, 0]
    group_names = ["b", "r", "b", "r", "b"]
    labels = [0, 1]
    classification_fairness_report(y, y_pred, groups, group_names, labels)


def test_classification_fairness_report_dict():
    y = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    groups = [0, 1, 0, 1, 0]

    report = classification_fairness_report(y, y_pred, groups, output="dict")
    assert type(report) == defaultdict
