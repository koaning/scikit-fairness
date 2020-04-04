import cvxpy as cp
import pandas as pd
import numpy as np
from scipy.special._ufuncs import expit

from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, column_or_1d, check_array


class _FairClassifier(BaseEstimator, LinearClassifierMixin):
    def __init__(
        self,
        sensitive_cols=None,
        C=1.0,
        penalty="l1",
        fit_intercept=True,
        max_iter=100,
        train_sensitive_cols=False,
    ):
        self.sensitive_cols = sensitive_cols
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.max_iter = max_iter
        self.train_sensitive_cols = train_sensitive_cols
        self.C = C

    def fit(self, X, y):
        if self.penalty not in ["l1", "none"]:
            raise ValueError(
                f"penalty should be either 'l1' or 'none', got {self.penalty}"
            )

        self.sensitive_col_idx_ = self.sensitive_cols
        if isinstance(X, pd.DataFrame):
            self.sensitive_col_idx_ = [
                i for i, name in enumerate(X.columns) if name in self.sensitive_cols
            ]
        X, y = check_X_y(X, y, accept_large_sparse=False)

        sensitive = X[:, self.sensitive_col_idx_]
        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)
        X = self._add_intercept(X)

        column_or_1d(y)
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        self.classes_ = label_encoder.classes_

        if len(self.classes_) > 2:
            raise ValueError(
                f"This solver needs samples of exactly 2 classes"
                f" in the data, but the data contains {len(self.classes_)}: {self.classes_}"
            )

        self._solve(sensitive, X, y)
        return self

    def constraints(self, y_hat, y_true, sensitive, n_obs):
        raise NotImplementedError(
            "subclasses of _FairClassifier should implement constraints"
        )

    def _solve(self, sensitive, X, y):
        n_obs, n_features = X.shape
        theta = cp.Variable(n_features)
        y_hat = X @ theta

        log_likelihood = cp.sum(
            cp.multiply(y, y_hat)
            - cp.log_sum_exp(
                cp.hstack([np.zeros((n_obs, 1)), cp.reshape(y_hat, (n_obs, 1))]), axis=1
            )
        )
        if self.penalty == "l1":
            log_likelihood -= cp.sum((1 / self.C) * cp.norm(theta[1:]))

        constraints = self.constraints(y_hat, y, sensitive, n_obs)

        problem = cp.Problem(cp.Maximize(log_likelihood), constraints)
        problem.solve(max_iters=self.max_iter)

        if problem.status in ["infeasible", "unbounded"]:
            raise ValueError(f"problem was found to be {problem.status}")

        self.n_iter_ = problem.solver_stats.num_iters

        if self.fit_intercept:
            self.coef_ = theta.value[np.newaxis, 1:]
            self.intercept_ = theta.value[0:1]
        else:
            self.coef_ = theta.value[np.newaxis, :]
            self.intercept_ = np.array([0.0])

    def predict_proba(self, X):
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        return expit(decision_2d)

    def decision_function(self, X):
        X = check_array(X)

        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)
        return super().decision_function(X)

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.c_[np.ones(len(X)), X]

