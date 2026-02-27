# drt = a(b - rt) dt + σ * sqrt(rt) dWt
# Feller condition: 2*a*b > sigma**2

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')
from scipy.optimize import minimize
from scipy.stats import ncx2

class CIR:
    def __init__(self, df):
        self.df = df

    def cir_log_likelihood(self, params, r, dt):
        a, b, sigma = params
        if a <= 0 or b <= 0 or sigma <= 0:
            return 1e10

        # Feller condition
        if 2 * a * b <= sigma ** 2:
            return 1e10

        ll = 0
        for t in range(len(r) - 1):
            rt = max(r[t], 1e-8)  # floor to avoid zero
            rt1 = r[t + 1]
            exp_adt = np.exp(-a * dt)
            c = sigma ** 2 * (1 - exp_adt) / (4 * a)
            d = 4 * a * b / sigma ** 2
            lam = (4 * a * exp_adt / (sigma ** 2 * (1 - exp_adt))) * rt

            if c <= 0 or lam < 0 or rt1 <= 0:
                return 1e10

            # Logpdf to avoid underflow
            ll += ncx2.logpdf(rt1 / c, d, lam) - np.log(c)

        return -ll


    def calibrate(self, sampling_period="D", verbose=True):
        r = self.df[self.df.columns[0]].values
        r = r / 100
        r_cont = 2 * np.log(1 + (r / 2))
        if sampling_period == "M":
            step = 21
        elif sampling_period == "Q":
            step = 63
        else:
            step = 1
        r_cont = r_cont[::step]
        dt = step / 252

        # Inital Guesses
        a0 = 0.5
        b0 = np.mean(r_cont)
        sigma0 = 0.02  # more realistic for daily data in decimal form
        res = minimize(self.cir_log_likelihood,x0=[a0, b0, sigma0],args=(r_cont, dt),method="L-BFGS-B",bounds=[(1e-6, 20), (1e-6, 0.5), (1e-6, 1.0)])

        if not res.success: # Soft Raise
            print(f"Warning: {res.message}")
            print(f"Best params found: a={res.x[0]:.5f}, b={res.x[1]:.5f}, sigma={res.x[2]:.5f}")

        a, b, sigma = res.x
        self.a, self.b, self.sigma = a, b, sigma
        if verbose:
            print(f"a (mean reversion) = {a:.6f}")
            print(f"b (long-term mean) = {b:.6f}")
            print(f"sigma (volatility) = {sigma:.6f}")
            print(f"Feller condition satisfied: {2 * a * b > sigma ** 2}")
        return a, b, sigma, r_cont

    def simulate_paths(self, n_paths=1000, T=252, sampling_period="D"):
        """Simulate paths using calibrated Vasicek model.
        :param n_paths: number of paths to simulate
        :param T: number of days to simulate
        """
        dt = 1 / 252
        a, b, sigma, r_cont = self.calibrate(sampling_period=sampling_period)
        r0 = r_cont[-1]

        # Constants that don't depend on r
        exp_adt = np.exp(-a * dt)
        c = sigma ** 2 * (1 - exp_adt) / (4 * a)
        d = 4 * a * b / sigma ** 2

        simulated = np.zeros((n_paths, T))
        for i in range(n_paths):
            r = np.zeros(T)
            r[0] = r0
            for t in range(1, T):
                lam = (4 * a * exp_adt / (sigma ** 2 * (1 - exp_adt))) * r[t - 1]
                r[t] = c * ncx2.rvs(d, lam)
            simulated[i] = r

        expected_value = np.sum(simulated[:, -1]) / n_paths
        plt.figure(figsize=(10, 5))
        for i in range(n_paths):
            plt.plot(simulated[i] * 100, lw=0.8)
        plt.axhline(expected_value * 100, color='white', lw=1.5, linestyle='--', label=f'Expected value (T={T}d): {expected_value*100:.4f}%')
        plt.legend()
        plt.title(f"Simulated CIR Short-Rate Paths — {T} days ({n_paths} paths)")
        plt.xlabel("Days")
        plt.ylabel("Short Rate (%)")
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        plt.tight_layout()
        plt.show()



