import os
import pandas as pd
from pkg_resources import resource_filename

import warnings
from skfair.warning import FairnessWarning


def load_arrests(return_X_y=False, give_pandas=False):
    """
    Loads the arrests dataset which can serve as a benchmark for fairness. It is data on
    the police treatment of individuals arrested in Toronto for simple possession of small
    quantities of marijuana. The goal is to predict whether or not the arrestee was released
    with a summons while maintaining a degree of fairness.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:
    >>> from sklego.datasets import load_arrests
    >>> X, y = load_arrests(return_X_y=True)
    >>> X.shape
    (5226, 7)
    >>> y.shape
    (5226,)
    >>> load_arrests(give_pandas=True).columns
    Index(['released', 'colour', 'year', 'age', 'sex', 'employed', 'citizen',
           'checks'],
          dtype='object')

    The dataset was copied from the carData R package and can originally be found in:

    - Personal communication from Michael Friendly, York University.

    The documentation page of the dataset from the package can be viewed here:
    http://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html
    """
    filepath = resource_filename("skfair", os.path.join("data", "arrests.zip"))
    df = pd.read_csv(filepath)
    warnings.warn(FairnessWarning("You are about to play with a potentially unfair dataset."))

    if give_pandas:
        return df
    colnames = ["colour", "year", "age", "sex", "employed", "citizen", "checks"]
    X, y = (
        df[colnames].values,
        df["released"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y, 'feature_names': colnames}


def load_boston(return_X_y=False, give_pandas=False):
    """
    Loads the boston housing dataset which can serve as a benchmark for fairness. It will be
    removed from scikit-learn because there's big problems with it. In particular there's a
    column (named `b`) that refers to the skin color of inhabitants.

    You can read all about it here:

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:
    >>> from sklego.datasets import load_boston
    >>> X, y = load_boston(return_X_y=True)
    >>> X.shape
    (506, 13)
    >>> y.shape
    (506,)
    >>> load_arrests(give_pandas=True).columns
    Index(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
           'ptratio', 'b', 'lstat', 'price'],
          dtype='object')
    """
    filepath = resource_filename("skfair", os.path.join("data", "boston.zip"))
    df = pd.read_csv(filepath)
    warnings.warn(FairnessWarning("You are about to play with a notorious dataset."))

    if give_pandas:
        return df
    colnames = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                'ptratio', 'b', 'lstat']
    X, y = df[colnames].values, df['price'].values,
    if return_X_y:
        return X, y
    return {"data": X, "target": y, 'feature_names': colnames}
