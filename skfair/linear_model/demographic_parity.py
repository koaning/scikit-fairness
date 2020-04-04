import cvxpy as cp
import autograd.numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from skfair.linear_model._fairclassifier import _FairClassifier


class DemographicParityClassifier(BaseEstimator, LinearClassifierMixin):
    r"""
    A logistic regression classifier which can be constrained on demographic parity (p% score).

    Minimizes the Log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier.

    Only works for binary classification problems

    .. math::
        \begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right) d
        \boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}

    Source:
    - M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification

    :param covariance_threshold:
        The maximum allowed covariance between the sensitive attributes and the distance to the
        decision boundary. If set to None, no fairness constraint is enforced
    :param sensitive_cols:
        List of sensitive column names(when X is a dataframe)
        or a list of column indices when X is a numpy array.
    :param C:
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    :param penalty: Used to specify the norm used in the penalization. Expects 'none' or 'l1'
    :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    :param max_iter: Maximum number of iterations taken for the solvers to converge.
    :param train_sensitive_cols: Indicates whether the model should use the sensitive columns in the fit step.
    :param multi_class: The method to use for multiclass predictions
    :param n_jobs: The amount of parallel jobs thata should be used to fit multiclass models

    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):

        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[
            multi_class
        ]
        return multiclass_meta(
            _DemographicParityClassifer(*args, **kwargs), n_jobs=n_jobs
        )


class _DemographicParityClassifer(_FairClassifier):
    def __init__(self, covariance_threshold, **kwargs):
        super().__init__(**kwargs)
        self.covariance_threshold = covariance_threshold

    def constraints(self, y_hat, y_true, sensitive, n_obs):
        if self.covariance_threshold is not None:
            dec_boundary_cov = y_hat @ (sensitive - np.mean(sensitive, axis=0)) / n_obs
            return [cp.abs(dec_boundary_cov) <= self.covariance_threshold]
        else:
            return []

    @classmethod
    def _get_param_names(cls):
        return sorted(
            super(_DemographicParityClassifer, cls)._get_param_names()
            + _FairClassifier._get_param_names()
        )
