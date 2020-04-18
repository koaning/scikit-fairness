import pytest

from skfair.datasets import load_arrests
from skfair.warning import FairnessWarning


def test_shape_arrests():
    assert load_arrests()['data'].shape == (5226, 7)


def test_raise_warning():
    with pytest.warns(FairnessWarning):
        load_arrests()
