import os
import numpy as np
import pandas as pd
from pkg_resources import resource_filename


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
    if give_pandas:
        return df
    X, y = (
        df[["colour", "year", "age", "sex", "employed", "citizen", "checks"]].values,
        df["released"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}
