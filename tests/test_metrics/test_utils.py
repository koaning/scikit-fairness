from skfair.metrics.utils import true_false_positive_negative

import numpy as np


def test_binary_conf_matrix():
    conf_matrix = np.array([
        [0, 1],
        [0, 1]
    ])
    tn, fp, fn, tp = true_false_positive_negative(conf_matrix)
    assert tp == 1
    assert fp == 1
    assert fn == 0
    assert tn == 0


def test_multiclass_conf_matrix():
    conf_matrix = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    tn, fp, fn, tp = true_false_positive_negative(conf_matrix)
    assert tp == 3
    assert fp == 4
    assert fn == 4
    assert tn == 10
