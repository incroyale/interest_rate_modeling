import matplotlib.pyplot as plt
import numpy as np


class YieldPCA:

    def __init__(self,df):
        self.df = df

    def get_pca(self):
        if self.df.isna().sum().sum() != 0:
            raise ValueError("Dataframe contains missing values")

        # PCA of Covariance Matrix of Daily Yield Changes
        changes = self.df.diff().dropna()
        cov = np.cov(changes.T)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        explained = eigen_values / eigen_values.sum()
        pcs = np.arange(1, len(explained) + 1)
        maturities = self.df.columns.astype(float)
        return eigen_values, eigen_vectors, explained, pcs, maturities


    def plot_graphs(self):
        eigen_values, eigen_vectors, explained, pcs, maturities = self.get_pca()
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))

        # --- Variance Explained ---
        axs[0, 0].bar(pcs, explained)
        axs[0, 0].set_title("Variance Explained")
        axs[0, 0].set_xlabel("Principal Component")
        axs[0, 0].set_ylabel("Variance Ratio")
        axs[0, 0].grid(True)

        # --- Cumulative Variance ---
        axs[0, 1].plot(pcs, np.cumsum(explained))
        axs[0, 1].set_title("Cumulative Variance Explained")
        axs[0, 1].set_xlabel("Principal Component")
        axs[0, 1].set_ylabel("Cumulative Variance")
        axs[0, 1].grid(True)

        # --- PC1 ---
        axs[1, 0].plot(maturities, eigen_vectors[:, 0])
        axs[1, 0].set_title(f"PC1 (Level) - {explained[0]*100:.1f}%")
        axs[1, 0].set_xlabel("Maturity (Years)")
        axs[1, 0].set_ylabel("Factor Loading")
        axs[1, 0].grid(True)

        # --- PC2 ---
        axs[1, 1].plot(maturities, eigen_vectors[:, 1])
        axs[1, 1].set_title(f"PC2 (Slope) - {explained[1]*100:.1f}%")
        axs[1, 1].set_xlabel("Maturity (Years)")
        axs[1, 1].set_ylabel("Factor Loading")
        axs[1, 1].grid(True)

        # --- PC3 ---
        axs[2, 0].plot(maturities, eigen_vectors[:, 2])
        axs[2, 0].set_title(f"PC3 (Curvature) - {explained[2]*100:.1f}%")
        axs[2, 0].set_xlabel("Maturity (Years)")
        axs[2, 0].set_ylabel("Factor Loading")
        axs[2, 0].grid(True)

        # Remove empty subplot (bottom right)
        fig.delaxes(axs[2, 1])
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()
        plt.show();