from skfair.datasets import load_arrests


def test_shape_arrests():
    assert load_arrests()['data'].shape == (5226, 7)
