import pytest

from skfair.datasets import fetch_adult
from skfair.warning import FairnessWarning


def test_shape_arrests():
    assert fetch_adult()['data'].shape == (32561, 14)


def test_raise_warning():
    with pytest.warns(FairnessWarning):
        fetch_adult()
