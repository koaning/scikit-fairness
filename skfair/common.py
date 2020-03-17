import collections
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y


def as_list(val):
    """
    Helper function, always returns a list of the input value.

    :param val: the input value.
    :returns: the input value as a list.

    :Example:

    >>> as_list('test')
    ['test']

    >>> as_list(['test1', 'test2'])
    ['test1', 'test2']
    """
    treat_single_value = str

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return [val]


def flatten(nested_iterable):
    """
    Helper function, returns an iterator of flattened values from an arbitrarily nested iterable

    >>> list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    ['test1', 'test2', 'a', 'b', 'c', 'd']

    >>> list(flatten(['test1', ['test2']]))
    ['test1', 'test2']
    """
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        else:
            yield el


def expanding_list(list_to_extent, return_type=list):
    """
    Make a expanding list of lists by making tuples of the first element, the first 2 elements etc.

    :param list_to_extent:
    :param return_type: type of the elements of the list (tuple or list)

    :Example:

    >>> expanding_list('test')
    [['test']]

    >>> expanding_list(['test1', 'test2', 'test3'])
    [['test1'], ['test1', 'test2'], ['test1', 'test2', 'test3']]

    >>> expanding_list(['test1', 'test2', 'test3'], tuple)
    [('test1',), ('test1', 'test2'), ('test1', 'test2', 'test3')]
    """
    listed = as_list(list_to_extent)
    if len(listed) <= 1:
        return [listed]

    return [return_type(listed[: n + 1]) for n in range(len(listed))]
