import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
plt.style.use('dark_background')


class YieldCurveVisualizer:

    def plot_yield_surface(self, df, resample_period="YE"):
        """
        Showcase Macro Regimes Visually on a 3D Surface.
        :params: Resample period for time axis ('D'=daily, 'M'=monthly, 'Y'=yearly).
        :returns: 3D plot.
        """
        df_sample = df.resample(resample_period).mean()
        df_sample = df_sample.astype(float)
        Z = df_sample.T.values

        # Meshgrid
        T = mdates.date2num(df_sample.index.to_pydatetime())
        Tau = df_sample.columns.astype(float).values
        T_grid, Tau_grid = np.meshgrid(T, Tau)

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T_grid, Tau_grid, Z, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.9)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel("Year")
        ax.set_ylabel("Maturity (Years)")
        ax.set_zlabel("Yield (%)")
        ax.set_title("US Treasury Yield Surface")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Yield (%)")
        plt.show();


    def plot_ns_fit(self, df, date, ns_model):
        maturities = df.columns.astype(float)
        row = df.loc[date]
        params = ns_model.fit(maturities, row.values)
        b0, b1, b2, lamb = params
        tau_dense = np.linspace(0.05, 30, 200)
        fitted_curve = ns_model.predict(tau_dense)
        plt.figure(figsize=(8, 5))
        plt.scatter(maturities, row, label="Market", s=60)
        plt.plot(tau_dense, fitted_curve, label="Nelson–Siegel Fit", linewidth=2)
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title(f"Nelson–Siegel Fit — {date}")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"""beta0 (Level): {b0:.3f}, beta1 (Slope): {b1:.3f}, beta2 (Curvature): {b2:.3f}, lambda: {lamb:.3f}""")


    def plot_ns_vs_spline(self, df, date, ns_model):
        maturities = df.columns.astype(float).values
        row = df.loc[date].values
        ns_model.fit(maturities, row)
        tau_dense = np.linspace(maturities.min(), maturities.max(), 300)
        ns_curve = ns_model.predict(tau_dense)
        idx = np.argsort(maturities)
        tau_sorted = maturities[idx]
        y_sorted = row[idx]

        cs = CubicSpline(tau_sorted, y_sorted, bc_type="natural")
        spline_curve = cs(tau_dense)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(maturities, row, s=60)
        axes[0].plot(tau_dense, ns_curve, linewidth=2)
        axes[0].set_title(f"Nelson–Siegel — {date}")
        axes[0].set_xlabel("Maturity (Years)")
        axes[0].set_ylabel("Yield (%)")
        axes[0].grid(True)
        axes[1].scatter(maturities, row, s=60)
        axes[1].plot(tau_dense, spline_curve, linewidth=2)
        axes[1].set_title(f"Cubic Spline — {date}")
        axes[1].set_xlabel("Maturity (Years)")
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()