from skfair.metrics import false_discovery_score

import numpy as np


def test_zero_false_discovery_score():
    y = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])

    assert false_discovery_score(y, y_pred) == 0


def test_nonzero_false_discovery_score():
    y = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 0, 0])

    # Gives conf matrix (rows actual, cols preds)
    # [2,0,0]
    # [0,1,0]
    # [1,0,0]
    # Thus
    # TP = 2 + 1 + 0 = 3
    # FP = 0 + 0 + 1 = 1
    # FDR = FP / (FP + TP)
    eps = 1e-3
    expected_false_discover_rate = 1.0 / (3.0 + 1.0)
    assert abs(false_discovery_score(y, y_pred) - expected_false_discover_rate) < eps
