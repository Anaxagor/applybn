import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.decomposition import PCA

class ResultsDisplay:
    """
    A class to display results of anomaly detection using scatter plots.
    """

    def __init__(self, outlier_scores: pd.DataFrame | np.ndarray, y_true: pd.DataFrame | np.ndarray):
        """
        Initializes the ResultsDisplay object.

        Args:
            outlier_scores: The outlier scores for the data points.
            y_true: The ground truth labels for anomalies (optional).
        """
        self.outlier_scores = outlier_scores
        self.y_true = y_true

    def show(self):
        """
        Displays a scatter plot of the outlier scores. If ground truth labels are provided,
        the points are colored based on their anomaly status.

        The x-axis represents the index of the data points, and the y-axis represents the outlier scores.
        """
        outlier_scores = np.array(self.outlier_scores)

        if self.y_true is None:
            # Create a DataFrame with only the outlier scores
            final = pd.DataFrame(outlier_scores.reshape(-1, 1), columns=["score"])
        else:
            # Create a DataFrame with outlier scores and ground truth labels
            y_true = np.array(self.y_true)
            final = pd.DataFrame(
                np.hstack(
                    [outlier_scores.reshape(-1, 1), y_true.reshape(-1, 1).astype(int)]
                ),
                columns=["score", "anomaly"],
            )

        plt.figure(figsize=(20, 12))
        sns.scatterplot(
            data=final,
            x=range(final.shape[0]),
            s=20,
            y="score",
            hue="anomaly" if not self.y_true is None else None,
        )

        plt.show()

    @staticmethod
    def plot_lof(X: pd.DataFrame | np.ndarray,
                 negative_factors: np.ndarray):
        """
        Plots the Local Outlier Factor (LOF) results using a scatter plot.

        If the data has more than 2 dimensions, PCA is applied to reduce it to 3 dimensions, but 2 are displayed.

        Args:
            X: The input data points.
            negative_factors: The negative outlier factor scores for the data points.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        def update_legend_marker_size(handle, orig):
            """
            Updates the size of the legend markers.

            Args:
                handle: The legend handle.
                orig: The original legend object.
            """
            handle.update_from(orig)
            handle.set_sizes([20])

        if X.shape[1] > 2:
            # Apply PCA to reduce dimensions to 3
            pca = PCA(n_components=3)
            X = pca.fit_transform(X)

        # Scatter plot of data points
        plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")

        # Calculate radius for outlier scores
        radius = (negative_factors.max() - negative_factors) / (
            negative_factors.max() - negative_factors.min()
        )

        scatter = plt.scatter(
            X[:, 0],
            X[:, 1],
            s=1000 * radius,
            edgecolors="r",
            facecolors="none",
            label="Outlier scores",
        )

        plt.axis("tight")
        plt.legend(
            handler_map={
                scatter: HandlerPathCollection(update_func=update_legend_marker_size)
            }
        )
        plt.title("Local Outlier Factor (LOF)")
        plt.show()