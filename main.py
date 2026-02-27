import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from utils.data_loader import fetch_treasury_data, preprocess_data
from utils.visualizations import YieldCurveVisualizer
from models.NelsonSiegel import NelsonSiegel
from models.CIR import CIR
from utils.bootstrap import BootstrapZeroCurve
from models.PCA import YieldPCA
from models.Vasicek import Vasicek
plt.style.use('dark_background')
matplotlib.use('Qt5Agg')

# # Use Fresh Data
# df = fetch_treasury_data(start_date=None, end_date=None)
# df = preprocess_data(df)
# df.to_csv("data/trreasury_daily.csv")

# Use Stored Data
df = pd.read_csv("data/treasury_daily.csv", index_col=0, parse_dates=True)
#
# # --- Visualization ---
# visualizer = YieldCurveVisualizer()
# visualizer.plot_yield_surface(df)
#
# # --- Nelson-Siegel ---
# ns = NelsonSiegel()
# visualizer.plot_ns_fit(df, df.index[-1], ns)
#
# # --- Bootstrapping ---
# bootstrap_engine = BootstrapZeroCurve()
# bootstrap_engine.plot_bootstrap(df)
#
# # --- PCA ---
# pca_engine = YieldPCA(df)
# pca_engine.plot_graphs()
#
# # --- Vasicek ---
# df = pd.read_csv("data/10_years_data.csv", index_col=0, parse_dates=True)
# vasicek = Vasicek(df)
# vasicek.simulate_paths(n_paths=100, T=252)

# --- CIR ---
# df = pd.read_csv("data/10_years_data.csv", index_col=0, parse_dates=True)
# CIR = CIR(df)
# CIR.calibrate(sampling_period="M")






