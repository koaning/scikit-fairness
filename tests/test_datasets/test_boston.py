import pytest

from skfair.datasets import load_boston
from skfair.warning import FairnessWarning


def test_shape_arrests():
    assert load_boston()['data'].shape == (506, 13)


def test_raise_warning():
    with pytest.warns(FairnessWarning):
        load_boston()
