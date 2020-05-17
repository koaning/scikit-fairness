from skfair.metrics import false_positive_score

import numpy as np


def test_zero_false_positive_score():
    y = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])

    assert false_positive_score(y, y_pred) == 0


def test_nonzero_false_positive_score():
    y = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 0, 2, 0])

    # Gives conf matrix (rows are actual, cols are preds)
    # [2,0,0]
    # [1,0,0]
    # [0,0,1]
    # FP = 0 + 1 + 0
    # TN = 1 + 3 + 3
    # FPR = FP / (FP + TN)
    eps = 1e-3
    expected_false_positive_rate = 1.0 / (1.0 + 7.0)
    assert abs(false_positive_score(y, y_pred) - expected_false_positive_rate) < eps
