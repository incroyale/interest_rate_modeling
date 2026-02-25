import numpy as np
from scipy.optimize import minimize


class NelsonSiegel:

    def __init__(self):
        self.params = None


    def _curve(self, tau, b0, b1, b2, lamb):
        tau = np.asarray(tau)
        term1 = (1 - np.exp(-lamb * tau)) / (lamb * tau)
        term2 = term1 - np.exp(-lamb * tau)
        return b0 + b1 * term1 + b2 * term2


    def _objective(self, params, tau, y):
        return np.sum((y - self._curve(tau, *params)) ** 2)


    def fit(self, tau, y, init_lambda=0.5):
        tau = np.asarray(tau, dtype=float)
        y = np.asarray(y, dtype=float)
        init_params = [y.mean(), -1.0, 1.0, init_lambda]
        bounds = [(0, 15), (-20, 20), (-20, 20), (0.01, 5)]
        res = minimize(self._objective, init_params, args=(tau, y), method="L-BFGS-B",bounds=bounds)
        self.params = res.x
        return self.params


    def predict(self, tau):
        if self.params is None:
            raise ValueError("Model not fitted yet.")
        return self._curve(tau, *self.params)