import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from data import fetch_treasury_data, preprocess_data
from visualizations import YieldCurveVisualizer
from models.NelsonSiegel import NelsonSiegel
from bootstrap import BootstrapZeroCurve
plt.style.use('dark_background')
matplotlib.use('Qt5Agg')

# --- Data ---
df = fetch_treasury_data()
df = preprocess_data(df)

# --- Visualization ---
visualizer = YieldCurveVisualizer()
visualizer.plot_yield_surface(df)

# --- Nelson-Siegel ---
ns_model = NelsonSiegel()
visualizer.plot_ns_fit(df, df.index[-1], ns_model)

# --- Bootstrapping ---
bootstrap_engine = BootstrapZeroCurve()
zero_curve = bootstrap_engine.bootstrap(df)
zero_df = pd.DataFrame(list(zero_curve.items()), columns=['Maturity', 'ZeroRate'])
zero_df.set_index('Maturity', inplace=True)
print("\nBootstrapped Zero Curve:\n")
print(zero_df)
plt.figure(figsize=(8, 5))
plt.plot(zero_df.index, zero_df['ZeroRate'], marker='o', linestyle='-')
plt.xlabel("Maturity (years)")
plt.ylabel("Zero Rate (%)")
plt.title("Bootstrapped Zero-Coupon Curve")
plt.grid(True)
plt.show()