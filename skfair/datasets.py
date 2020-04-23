import os
from os.path import dirname, exists, join
from os import makedirs, remove
import pathlib

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

import warnings
from skfair.warning import FairnessWarning
from sklearn.datasets import get_data_home


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


def fetch_olivetti_faces(data_home=None, download_if_missing=True, return_X_y=False):
    """
    Load the ADULT INCOME dataset.
    Download it if necessary from github.
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.
    return_X_y : boolean, default=False.
        If True, returns `(data, target)` instead of a `Bunch` object. See
        below for more information about the `data` and `target` object.
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = str(pathlib.Path(data_home) / "adult.zip")
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        print('downloading Olivetti faces from %s to %s'
              % (FACES.url, data_home))
        mat_path = _fetch_remote(FACES, dirname=data_home)
        mfile = loadmat(file_name=mat_path)
        # delete raw .mat data
        remove(mat_path)

        faces = mfile['faces'].T.copy()
        joblib.dump(faces, filepath, compress=6)
        del mfile
    else:
        faces = _refresh_cache([filepath], 6)
        # TODO: Revert to the following line in v0.23
        # faces = joblib.load(filepath)

    # We want floating point data, but float32 is enough (there is only
    # one byte of precision in the original uint8s anyway)
    faces = np.float32(faces)
    faces = faces - faces.min()
    faces /= faces.max()
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    # 10 images per class, 400 images total, each class is contiguous.
    target = np.array([i // 10 for i in range(400)])
    if shuffle:
        random_state = check_random_state(random_state)
        order = random_state.permutation(len(faces))
        faces = faces[order]
        target = target[order]
    faces_vectorized = faces.reshape(len(faces), -1)

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'olivetti_faces.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return faces_vectorized, target

    return Bunch(data=faces_vectorized,
                 images=faces,
                 target=target,
                 DESCR=fdescr)
