import os
import shutil
import warnings
from pkg_resources import resource_filename

import requests
import pandas as pd
from sklearn.datasets import get_data_home
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
    warnings.warn(FairnessWarning("You are about to play with an unfair dataset."))

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


def download_file(url, local_filename):
    with requests.get(url, stream=True, verify=False) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def fetch_adult(data_home=None, give_pandas=False, download_if_missing=True, return_X_y=False):
    """
    Load the ADULT INCOME dataset.
    Download it if necessary from github.

    ----------

    :param data_home : Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders. optional, default: None
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)
    :param download_if_missing : If False, raise a IOError if the data is not locally available instead
    of trying to download the data from the source site.  True by default
    :param return_X_y : If True, returns `(data, target)` instead of a dictionary. See
    below for more information about the `data` and `target` object.

    :Example:
    >>> from sklego.datasets import fetch_adult
    >>> X, y = fetch_adult(return_X_y=True)
    >>> X.shape
    (32561,, 14)
    >>> y.shape
    (32561,)
    >>> fetch_adult(give_pandas=True).columns
    Index(['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
           'income'], dtype='object')
    """
    data_home = get_data_home(data_home=data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    filepath = os.path.join(data_home, "adult.zip")
    if not os.path.exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        print(f"downloading dataset to {data_home}")
        url = "https://github.com/koaning/scikit-fairness/raw/master/data/adult-census-income.zip"
        download_file(url, filepath)
    df = pd.read_csv(filepath)
    warnings.warn(FairnessWarning("You are about to play with an unfair dataset."))
    if give_pandas:
        return df
    colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
                'marital.status', 'occupation', 'relationship', 'race', 'sex',
                'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
    X, y = df[colnames].values, df['income'].values,
    if return_X_y:
        return X, y
    return {"data": X, "target": y, 'feature_names': colnames}
