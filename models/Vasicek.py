# drt = a(b − rt) dt + σ dWt

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')

class Vasicek:

    def __init__(self, df):
        self.df = df

    def calibrate(self, sampling_period="D", verbose=True):
        # Historical calibration for a and sigma
        r = self.df[self.df.columns[0]]
        r = r / 100
        r_cont = 2 * np.log(1 + (r / 2))

        if sampling_period == "M":
            step = 21
        elif sampling_period == "Q":
            step = 63
        else:
            step = 1

        r_cont = r_cont.iloc[::step].values
        dt = step / 252
        n = len(r_cont) - 1
        r_t  = r_cont[:-1]
        r_t1 = r_cont[1:]

        # Sufficient statistics
        Sx  = np.sum(r_t)
        Sy  = np.sum(r_t1)
        Sxx = np.sum(r_t ** 2)
        Sxy = np.sum(r_t * r_t1)
        Syy = np.sum(r_t1 ** 2)

        # Closed-form OLS
        e = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
        a = -np.log(e) / dt
        b = (Sy - e * Sx) / (n * (1 - e))
        resid_var = (Syy - 2 * e * Sxy + e ** 2 * Sxx - 2 * b * (1 - e) * (Sy - e * Sx) + n * (b * (1 - e)) ** 2) / n
        sigma = np.sqrt(resid_var * 2 * a / (1 - np.exp(-2 * a * dt)))

        if verbose:
            print(f"a (mean reversion) = {a:.6f}")
            print(f"b (long-term mean) = {b:.6f}")
            print(f"sigma (volatility) = {sigma:.6f}")
        return a, b, sigma, r_cont

    def simulate_paths(self, n_paths=1000, T=252, sampling_period="D"):
        """Simulate paths using calibrated Vasicek model.
        :param n_paths: number of paths to simulate
        :param T: number of days to simulate
        """
        dt = 1 / 252
        a, b, sigma, r_cont = self.calibrate(sampling_period=sampling_period)
        # Bounds
        a = max(a, 1e-4)
        sigma = abs(sigma)
        r0 = r_cont[-1]

        simulated = np.zeros((n_paths, T))
        for i in range(n_paths):
            r = np.zeros(T)
            r[0] = r0
            for t in range(1, T):
                e = np.exp(-a * dt)
                std = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
                r[t] = r[t - 1] * e + b * (1 - e) + std * np.random.randn()
            simulated[i] = r

        expected_value = np.sum(simulated[:, -1]) / n_paths
        plt.figure(figsize=(10, 5))
        for i in range(n_paths):
            plt.plot(simulated[i] * 100, lw=0.8)
        plt.axhline(expected_value * 100, color='white', lw=1.5, linestyle='--', label=f'Expected value (T={T}d): {expected_value * 100:.4f}%')
        plt.legend()
        plt.title(f"Simulated Vasicek Short-Rate Paths — {T} days ({n_paths} paths)")
        plt.xlabel("Days")
        plt.ylabel("Short Rate (%)")
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        plt.tight_layout()
        plt.show();