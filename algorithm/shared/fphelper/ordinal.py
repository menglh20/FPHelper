"""Cumulative ordinal logistic regression with strictly ordered thresholds."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def cuts_from_parameters(parameters, dimension, count):
    cuts = np.full(count, parameters[dimension])
    if count > 1:
        cuts[1:] += np.cumsum(np.logaddexp(0, parameters[dimension + 1:]) + 1e-4)
    return cuts


def objective(parameters, x, thresholds, regularization):
    n, dimension = x.shape
    count = thresholds.shape[1]
    beta = parameters[:dimension]
    cuts = cuts_from_parameters(parameters, dimension, count)
    logits = x @ beta[:, None] - cuts[None, :]
    loss = np.mean(np.logaddexp(0, logits) - thresholds * logits) + regularization * (beta @ beta) / 2
    residual = (expit(logits) - thresholds) / (n * count)
    beta_gradient = x.T @ residual.sum(axis=1) + regularization * beta
    cut_gradient = -residual.sum(axis=0)
    gradient = np.empty_like(parameters)
    gradient[:dimension] = beta_gradient
    gradient[dimension] = cut_gradient.sum()
    if count > 1:
        gradient[dimension + 1:] = expit(parameters[dimension + 1:]) * np.cumsum(cut_gradient[::-1])[::-1][1:]
    return float(loss), gradient


class OrderedLogit:
    def __init__(self, low, high, regularization=0.1):
        self.low = int(low)
        self.high = int(high)
        self.regularization = regularization

    def fit(self, x, y):
        self.supported = np.unique(y).astype(int)
        self.constant = int(y[0]) if len(self.supported) == 1 else None
        self.converged = True
        self.iterations = 0
        if self.constant is not None:
            return self
        count = self.high - self.low
        z = (y[:, None] > np.arange(self.low, self.high)[None, :]).astype(float)
        # Jeffreys smoothing produces finite, separated initial cutpoints even with missing classes.
        cdf = (len(y) - z.sum(axis=0) + 0.5) / (len(y) + 1)
        cuts = np.log(cdf / (1 - cdf))
        for j in range(1, count):
            cuts[j] = max(cuts[j], cuts[j - 1] + 0.1)
        delta = np.diff(cuts) - 1e-4
        initial = np.r_[np.zeros(x.shape[1]), cuts[0], np.log(np.expm1(delta))]
        result = minimize(objective, initial, args=(x, z, self.regularization),
                          jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Non-finite ordinal training result")
        self.parameters = result.x
        self.converged = bool(result.success)
        self.iterations = int(result.nit)
        return self

    def predict_proba(self, x):
        if self.constant is not None:
            probabilities = np.zeros((len(x), self.high - self.low + 1))
            probabilities[:, self.constant - self.low] = 1
            return probabilities
        cuts = cuts_from_parameters(self.parameters, x.shape[1], self.high - self.low)
        q = expit(x @ self.parameters[:x.shape[1], None] - cuts[None, :])
        return np.c_[1 - q[:, 0], q[:, :-1] - q[:, 1:], q[:, -1]]

    def predict(self, x):
        # Conditional median is appropriate for absolute error on an ordered scale.
        return self.low + (np.cumsum(self.predict_proba(x), axis=1) < 0.5).sum(axis=1)
