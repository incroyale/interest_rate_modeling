import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
import matplotlib.pyplot as plt

class BootstrapZeroCurve:

    def __init__(self, compounding=2):
        self.compounding = compounding

    def bootstrap(self, df):
        """
        Proper bootstrap:
        1) Interpolate par yields onto full semiannual grid
        2) Bootstrap discount factors sequentially
        3) Convert to zero rates
        """
        market_maturities = np.array(df.columns.astype(float))
        market_yields = np.array(df.iloc[-1].values) / 100  # convert to decimal

        max_T = market_maturities.max()

        # Full semiannual grid
        grid = np.arange(0.5, max_T + 0.5, 0.5)

        # PCHIP
        interpolator = PchipInterpolator(market_maturities, market_yields)
        par_yields = interpolator(grid)
        discount_factors = {}

        for i, T in enumerate(grid):
            y = par_yields[i]

            if T <= 1:
                # Treat <=1y as zero coupon
                DF = 1 / ((1 + y / self.compounding) ** (self.compounding * T))
            else:
                coupon = y / self.compounding
                n = int(T * self.compounding)
                pv_coupons = 0.0

                for j in range(1, n):
                    t_j = j / self.compounding
                    pv_coupons += coupon * discount_factors[t_j]
                DF = (1 - pv_coupons) / (1 + coupon)
            discount_factors[T] = DF

        zero_rates = {}
        for T in grid:
            DF = discount_factors[T]
            z = self.compounding * (DF ** (-1 / (self.compounding * T)) - 1)
            zero_rates[T] = z * 100  # back to percent
        return zero_rates


    def plot_bootstrap(self, df):
        zero_curve = self.bootstrap(df)
        zero_df = pd.DataFrame(list(zero_curve.items()), columns=['Maturity', 'ZeroRate'])
        print("\nBootstrapped Zero Curve:\n")
        # print(zero_df)
        plt.figure(figsize=(8, 5))
        plt.plot(zero_df['Maturity'], zero_df['ZeroRate'], marker='o', linestyle='-')
        plt.xlabel("Maturity (years)")
        plt.ylabel("Zero Rate (%)")
        plt.title("Bootstrapped Zero-Coupon Curve")
        plt.grid(True)
        plt.show();
