import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerPathCollection
from sklearn.decomposition import PCA


class ResultsDisplay:
    def __init__(self, outlier_scores, y_true):
        self.outlier_scores = outlier_scores
        self.y_true = y_true

    def show(self):
        outlier_scores = np.array(self.outlier_scores)

        if self.y_true is None:
            final = pd.DataFrame(outlier_scores.reshape(-1, 1), columns=["score"])
        else:
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
    def plot_lof(X, negative_factors):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        def update_legend_marker_size(handle, orig):
            handle.update_from(orig)
            handle.set_sizes([20])

        if X.shape[1] > 2:
            pca = PCA(n_components=3)
            X = pca.fit_transform(X)

        plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
        # plot circles with radius proportional to the outlier scores
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
