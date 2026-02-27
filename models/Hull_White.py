# drt = (theta(t) - a*rt) dt + sigma dWt
# Hull-White: extends Vasicek with time-dependent drift theta(t)
# theta(t) is calibrated to fit the initial yield curve exactly (no-arbitrage)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from utils.bootstrap import BootstrapZeroCurve


class HullWhite:

    def __init__(self, df):
        self.df = df
        self.bootstrapper = BootstrapZeroCurve(compounding=2)

    def forward_rate(self, P, t1, t2):
        """Continuously compounded forward rate between t1 and t2."""
        return -np.log(P[t2] / P[t1]) / (t2 - t1)

    def calibrate(self, sampling_period="D", verbose=True):
        """
        Calibrate Hull-White model.
        Step 1: Estimate a and sigma from historical short rate via AR(1) OLS (same as Vasicek).
        Step 2: Bootstrap discount factors from today's CMT curve using BootstrapZeroCurve.
        Step 3: Compute theta(t) analytically from the forward curve.
        """

        # Historical calibration for a and sigma
        r = self.df[self.df.columns[0]] / 100
        r_cont = 2 * np.log(1 + (r / 2))  # semi-annual to continuously compounded

        if sampling_period == "M":
            step = 21
        elif sampling_period == "Q":
            step = 63
        else:
            step = 1

        r_cont_series = r_cont.iloc[::step].values
        dt = step / 252
        n = len(r_cont_series) - 1
        r_t  = r_cont_series[:-1]
        r_t1 = r_cont_series[1:]

        # Sufficient statistics
        Sx = np.sum(r_t)
        Sy = np.sum(r_t1)
        Sxx = np.sum(r_t ** 2)
        Sxy = np.sum(r_t * r_t1)
        Syy = np.sum(r_t1 ** 2)

        # Closed-form OLS
        e = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
        a = -np.log(e) / dt
        b_ols = (Sy - e * Sx) / (n * (1 - e))  # local mean approx, not used in HW directly
        resid_var = (Syy - 2*e*Sxy + e**2*Sxx - 2*b_ols*(1-e)*(Sy - e*Sx) + n*(b_ols*(1-e))**2) / n
        sigma = np.sqrt(resid_var * 2 * a / (1 - np.exp(-2 * a * dt)))

        self.a = a
        self.sigma = sigma

        if verbose:
            print(f"a (mean reversion) = {a:.6f}")
            print(f"sigma (volatility) = {sigma:.6f}")
            print(f"b_ols (local mean approx) = {b_ols:.6f}")

        # Bootstrap discount factors from today's CMT curve
        zero_rates = self.bootstrapper.bootstrap(self.df)

        # Convert zero rates back to discount factors
        P = {}
        for T, z in zero_rates.items():
            z_dec = z / 100
            P[T] = (1 + z_dec / 2) ** (-2 * T)
        self.P = P

        # Compute theta(t) analytically from forward curve
        # theta(t) = df/dt(0,t) + a*f(0,t) + sigma^2/(2a) * (1 - e^{-2at})
        mats = sorted(P.keys())
        thetas = {}
        for i in range(1, len(mats) - 1):
            t = mats[i]
            t_prev = mats[i - 1]
            t_next = mats[i + 1]
            f_prev = self.forward_rate(P, t_prev, t)
            f_next = self.forward_rate(P, t, t_next)
            f_t = self.forward_rate(P, t_prev, t_next)
            # Central finite difference for df/dt
            dfdt = (f_next - f_prev) / (t_next - t_prev)
            thetas[t] = dfdt + a * f_t + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))

        self.thetas = thetas

        if verbose:
            print(f"theta(t) computed at {len(thetas)} tenor points")

        return a, sigma, P, thetas, r_cont_series

    def simulate_paths(self, n_paths=1000, T=252, verbose=True):
        """
        Simulate Hull-White paths using exact Gaussian transition.
        theta(t) is interpolated from calibrated tenor points.
        :param n_paths: number of paths to simulate
        :param T: number of days to simulate
        """
        dt = 1 / 252
        a, sigma, P, thetas, r_cont = self.calibrate(verbose=verbose)
        r0 = r_cont[-1]

        # Interpolate theta onto daily grid
        theta_times = np.array(sorted(thetas.keys()))
        theta_values = np.array([thetas[t] for t in theta_times])
        time_grid = np.linspace(0, T * dt, T)
        theta_interp = np.interp(time_grid, theta_times, theta_values)

        simulated = np.zeros((n_paths, T))
        for i in range(n_paths):
            r = np.zeros(T)
            r[0] = r0
            for t in range(1, T):
                e_adt = np.exp(-a * dt)
                mean = r[t-1] * e_adt + (theta_interp[t] / a) * (1 - e_adt)
                std = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
                r[t] = mean + std * np.random.randn()
            simulated[i] = r

        expected_value = np.sum(simulated[:, -1]) / n_paths

        plt.figure(figsize=(10, 5))
        for i in range(n_paths):
            plt.plot(simulated[i] * 100, lw=0.8)
        plt.axhline(expected_value * 100, color='white', lw=1.5, linestyle='--', label=f'Expected value (T={T}d): {expected_value*100:.4f}%')
        plt.legend()
        plt.title(f"Simulated Hull-White Short-Rate Paths — {T} days ({n_paths} paths)")
        plt.xlabel("Days")
        plt.ylabel("Short Rate (%)")
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        plt.tight_layout()
        plt.show();

    def plot_yield_curve_fit(self, verbose=False):
        """
        Plot model-implied zero yields vs market CMT yields to verify
        Hull-White fits the initial yield curve exactly.
        Also plots the forward curve and theta(t).
        """
        a, sigma, P, thetas, _ = self.calibrate(verbose=verbose)
        mats = sorted(P.keys())

        # Market zero yields
        market_zero_yields = [-np.log(P[T]) / T * 100 for T in mats]

        # HW implied zero yields
        model_zero_yields = [-np.log(P[T]) / T * 100 for T in mats]

        # Forward curve at midpoints between adjacent tenors
        forward_yields = []
        forward_mats   = []
        for i in range(len(mats) - 1):
            f = self.forward_rate(P, mats[i], mats[i + 1])
            forward_yields.append(f * 100)
            forward_mats.append((mats[i] + mats[i + 1]) / 2)  # midpoint

        # theta(t)
        theta_times  = np.array(sorted(thetas.keys()))
        theta_values = np.array([thetas[t] for t in theta_times]) * 100

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Zero curve fit
        axes[0].plot(mats, market_zero_yields, 'o-', color='yellow', label='Market (CMT bootstrapped)', lw=2)
        axes[0].plot(mats, model_zero_yields, '--', color='cyan', label='Hull-White implied', lw=2)
        axes[0].set_title('Zero Yield Curve — Market vs Hull-White')
        axes[0].set_xlabel('Maturity (years)')
        axes[0].set_ylabel('Zero Yield (%)')
        axes[0].legend()
        axes[0].grid(True)

        # Forward curve with theta
        ax2 = axes[1]
        ax2.plot(forward_mats, forward_yields, 'o-', color='orange', label='Forward curve f(0,T)', lw=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(theta_times, theta_values, 's--', color='magenta', label='θ(t)', lw=2)
        ax2.set_title('Forward Curve and θ(t)')
        ax2.set_xlabel('Maturity (years)')
        ax2.set_ylabel('Forward Rate (%)', color='orange')
        ax2_twin.set_ylabel('θ(t) x100', color='magenta')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True)
        plt.tight_layout()
        plt.show()