import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from utils.bootstrap import BootstrapZeroCurve
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

class G2:

    def __init__(self, df):
        self.df           = df
        self.bootstrapper = BootstrapZeroCurve(compounding=2)

    def forward_rate(self, P, t1, t2):
        return -np.log(P[t2] / P[t1]) / (t2 - t1)

    def _compute_phi(self, t_grid):
        a, b, sigma, eta, rho = self.theta

        # Build interpolator over P once — avoids KeyError on arbitrary t values
        p_times = np.array(sorted(self.P.keys()))
        p_values = np.array([self.P[t] for t in p_times])
        from scipy.interpolate import PchipInterpolator
        P_interp = PchipInterpolator(p_times, p_values)

        def f_market(t):
            """Instantaneous forward rate via finite difference on interpolated P."""
            eps = 1e-5
            t1 = max(t - eps, 1e-6)
            t2 = t + eps
            return -np.log(P_interp(t2) / P_interp(t1)) / (t2 - t1)

        phi = []
        for t in t_grid:
            t_safe = min(max(t, 1e-6), p_times[-1])
            phi_t = (f_market(t_safe) + sigma ** 2 / (2 * a ** 2) * (1 - np.exp(-a * t_safe)) ** 2
                    + eta ** 2 / (2 * b ** 2) * (1 - np.exp(-b * t_safe)) ** 2
                    + rho * sigma * eta / (a * b) * (1 - np.exp(-a * t_safe)) * (1 - np.exp(-b * t_safe)))
            phi.append(phi_t)
        return np.array(phi)

    def calibrate(self, verbose=True):
        dt = 1 / 252
        # Step 1: Bootstrap P from today's curve (last row via bootstrapper)
        zero_rates = self.bootstrapper.bootstrap(self.df)
        P = {T: (1 + z/100 / 2)**(-2*T) for T, z in zero_rates.items()}
        self.P = P
        maturities = sorted(P.keys())

        cmt_cols    = np.array(self.df.columns.astype(float))
        cmt_matrix  = self.df.values.astype(float) / 100      # (T_days, N_cmt_tenors)

        # Interpolate each daily row onto the bootstrapped semiannual grid
        aligned = np.array([PchipInterpolator(cmt_cols, row)(maturities) for row in cmt_matrix])                                                      # (T_days, N_maturities)

        delta_y = np.diff(aligned, axis=0)                   # (T_days-1, N_maturities)
        Sigma_emp = np.cov(delta_y.T)                          # (N_maturities, N_maturities)

        # Step 3: Model-implied covariance matrix
        def Sigma_model(theta):
            a, b, sigma, eta, rho = theta
            Bx = np.array([(1 - np.exp(-a*T)) / a for T in maturities])
            By = np.array([(1 - np.exp(-b*T)) / b for T in maturities])
            Var_x = sigma**2 / (2*a) * (1 - np.exp(-2*a*dt))
            Var_y = eta**2   / (2*b) * (1 - np.exp(-2*b*dt))
            Cov_xy = rho * sigma * eta / (a+b) * (1 - np.exp(-(a+b)*dt))
            return (np.outer(Bx, Bx) * Var_x + np.outer(By, By) * Var_y + (np.outer(Bx, By) + np.outer(By, Bx)) * Cov_xy)

        # Step 4: Objective — Frobenius norm with degeneracy penalty
        def objective(theta):
            a, b = theta[0], theta[1]
            if abs(a - b) < 1e-4:
                return 1e10
            return np.sum((Sigma_emp - Sigma_model(theta))**2)

        # Step 5: Optimise — L-BFGS-B
        x0     = [0.1, 0.5, 0.01, 0.01, -0.7]
        bounds = [(1e-4, 2.0), (1e-4, 2.0), (1e-5, 0.1), (1e-5, 0.1), (-0.999, 0.999)]
        res    = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
        self.theta = res.x
        a, b, sigma, eta, rho = res.x

        if verbose:
            print(f"a (x mean reversion) = {a:.6f}")
            print(f"b (y mean reversion) = {b:.6f}")
            print(f"sigma (x volatility) = {sigma:.6f}")
            print(f"eta (y volatility) = {eta:.6f}")
            print(f"rho (correlation) = {rho:.6f}")
            print(f"Optimizer converged: {res.success}")
            print(f"Objective value: {res.fun:.6e}")
        return res.x

    def simulate_paths(self, n_paths=100, T_horizon=1.0, plot=True):
        """
        Simulate short rate paths under calibrated G2++ model.
        Exact OU transition — not Euler discretization error.
        r(t) = x(t) + y(t) + phi(t)
        """
        a, b, sigma, eta, rho = self.theta
        dt = 1 / 252
        N = int(T_horizon / dt) + 1
        t_grid = np.linspace(0, T_horizon, N)

        # Exact OU transition
        e_adt = np.exp(-a * dt)
        e_bdt = np.exp(-b * dt)
        std_x = sigma * np.sqrt((1 - np.exp(-2*a*dt)) / (2*a))
        std_y = eta * np.sqrt((1 - np.exp(-2*b*dt)) / (2*b))

        # Cholesky for correlated Brownian increments
        L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
        x = np.zeros((n_paths, N))
        y = np.zeros((n_paths, N))

        for t in range(1, N):
            Z = np.random.randn(n_paths, 2) @ L.T
            x[:, t] = x[:, t-1] * e_adt + std_x * Z[:, 0]
            y[:, t] = y[:, t-1] * e_bdt + std_y * Z[:, 1]

        # phi(t) with full G2++ correction
        phi = self._compute_phi(t_grid)
        r_paths = x + y + phi[np.newaxis, :]

        if plot:
            plt.figure(figsize=(12, 6))
            for i in range(min(n_paths, 50)):
                plt.plot(t_grid, r_paths[i] * 100, lw=0.6, alpha=0.6)
            plt.axhline(r_paths[:, -1].mean() * 100, color='white', lw=1.5, linestyle='--', label=f'Mean at T={T_horizon}y: {r_paths[:,-1].mean()*100:.4f}%')
            plt.xlabel("Time (years)")
            plt.ylabel("Short Rate (%)")
            plt.title(f"G2++ Simulated Short Rate Paths ({n_paths} paths)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            plt.tight_layout()
            plt.show()

        return r_paths, t_grid

    def plot_fit(self):
        a, b, sigma, eta, rho = self.theta
        dt = 1 / 252

        maturities = sorted(self.P.keys())
        cmt_cols = np.array(self.df.columns.astype(float))
        cmt_matrix = self.df.values.astype(float) / 100
        aligned = np.array([PchipInterpolator(cmt_cols, row)(maturities) for row in cmt_matrix])
        delta_y = np.diff(aligned, axis=0)

        Sigma_emp = np.cov(delta_y.T)
        Bx = np.array([(1 - np.exp(-a * T)) / a for T in maturities])
        By = np.array([(1 - np.exp(-b * T)) / b for T in maturities])
        Var_x = sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * dt))
        Var_y = eta ** 2 / (2 * b) * (1 - np.exp(-2 * b * dt))
        Cov_xy = rho * sigma * eta / (a + b) * (1 - np.exp(-(a + b) * dt))
        Sigma_mod = (np.outer(Bx, Bx) * Var_x + np.outer(By, By) * Var_y + (np.outer(Bx, By) + np.outer(By, Bx)) * Cov_xy)
        emp_vols = np.sqrt(np.diag(Sigma_emp) / dt) * 100
        mod_vols = np.sqrt(np.diag(Sigma_mod) / dt) * 100

        fwd_rates = [self.forward_rate(self.P, maturities[i], maturities[i + 1]) * 100 for i in range(len(maturities) - 1)]
        fwd_mats = [(maturities[i] + maturities[i + 1]) / 2 for i in range(len(maturities) - 1)]
        phi_values = self._compute_phi(np.array(maturities)) * 100

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].plot(maturities, emp_vols, 'o-', color='yellow', label='Empirical vol')
        axes[0].plot(maturities, mod_vols, 's--', color='cyan', label='Model vol')
        axes[0].set_title("Volatility Term Structure")
        axes[0].set_xlabel("Maturity (years)")
        axes[0].set_ylabel("Annualized Vol (bps)")
        axes[0].legend()
        axes[0].grid(True)
        ax = axes[1]

        ax2 = ax.twinx()
        ax.plot(fwd_mats, fwd_rates, 'o-', color='orange', label='Forward curve f(0,T)', lw=2)
        ax2.plot(maturities, phi_values, 's--', color='magenta', label='φ(t)', lw=2)
        ax.set_title("Forward Curve and φ(t)")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Forward Rate (%)", color='orange')
        ax2.set_ylabel("φ(t) x100", color='magenta')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        plt.show()