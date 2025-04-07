import pandas as pd

from anomaly_detection.scores.proximity_based import IsolationForestScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

import warnings
from applybn.anomaly_detection.scores.score import Score
from applybn.core.schema import bamt_network

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerPathCollection
from sklearn.decomposition import PCA
from scipy.stats import norm
from typing import Literal
from applybn.anomaly_detection.scores.model_based import BNBasedScore

class ODBPScore(Score):
    _model_estimation_method = {
        "original_modified": BNBasedScore,
        "iqr": None,
        "cond_ratio": None
    }

    _proximity_estimation_method = {
        "LOF": LocalOutlierScore,
        "IF": IsolationForestScore
    }

    def __init__(self,
                 bn: bamt_network,
                 iqr_sensivity=1.5, agg_funcs=None, verbose=1,
                 model_estimation_method: Literal["original_modified", "iqr", "cond_ratio"] = "iqr",
                 proximity_estimation_method: Literal["LOF", "IF"] = "LOF",
                 model_scorer_args=None, additional_scorer_args=None):
        super().__init__()
        if agg_funcs is None:
            agg_funcs = dict(proximity=np.sum, model=np.sum)

        if additional_scorer_args is None:
            additional_scorer_args = dict(proximity_steps=5)

        if model_scorer_args is None:
            model_scorer_args = dict()

        self.model_scorer = self._model_estimation_method[model_estimation_method](bn=bn, **model_scorer_args)
        self.proximity_scorer = self._proximity_estimation_method[proximity_estimation_method](**additional_scorer_args)

        self.agg_funcs = agg_funcs

        self.proximity_impact = 0
        self.model_impact = 0


        self.iqr_sensivity = iqr_sensivity
        self.verbose = verbose

    def __repr__(self):
        return f"ODBP Score (proximity={self.proximity_scorer})"

    def local_model_score_proba(self, X: pd.DataFrame, node_name):
        node = self.bn[node_name]
        diff = []
        parents = node.cont_parents + node.disc_parents

        for _, row in X.iterrows():
            pvalues = row[parents].to_dict()
            cond_dist = self.bn.get_dist(node_name, pvals=pvalues)

            if isinstance(cond_dist, tuple):
                cond_mean, var = cond_dist
                dist = norm(loc=cond_mean, scale=var)
                diff.append(1 - dist.cdf(row[node_name]))
            else:
                diff.append(1 - cond_dist[self.encoding[node_name][row[node_name]]])

        return np.asarray(diff).reshape(-1, 1)

    @staticmethod
    def score_iqr(upper, lower, y, max_distance, min_distance):
        if lower < y <= upper:
            return 0

        closest_value = min([upper, lower], key=lambda x: abs(x - y))

        current_distance = abs(closest_value - y)

        if closest_value == upper:
            ref_distance = max_distance
        elif closest_value == lower:
            ref_distance = min_distance
        else:
            # todo
            raise Exception
        result = min(1, current_distance / abs(ref_distance))

        return result

    def score_proba_ratio(self, sample: pd.Series, X_value, cond_dist):
        marginal_prob = sample.value_counts(normalize=True)[X_value]
        index = self.encoding[sample.name][X_value]
        cond_prob = cond_dist[index]

        if not np.isfinite(marginal_prob / cond_prob):
            # it is impossible to estimate if cond dataframe doesn't contain X_value
            return np.nan
        # the greater, the more abnormal
        return min(1, marginal_prob / cond_prob)

    def local_model_score_iqr(self, X: pd.DataFrame, node_name: str):
        node = self.bn[node_name]
        diff = []
        parents = node.cont_parents + node.disc_parents
        sources = []
        for _, row in X.iterrows():
            pvalues = row[parents].to_dict()
            cond_dist = self.bn.get_dist(node_name, pvals=pvalues)
            X_value = row[node_name]
            if isinstance(cond_dist, tuple):
                cond_mean, var = cond_dist
                if var == 0:
                    warnings.warn("Zero variance detected!")
                    continue

                dist = norm(loc=cond_mean, scale=var)

                q25 = dist.ppf(.25)
                q75 = dist.ppf(.75)
                iqr = q75 - q25

                lower_bound = q25 - iqr * self.iqr_sensivity
                upper_bound = q75 + iqr * self.iqr_sensivity

                diff.append(self.score_iqr(upper_bound, lower_bound, X_value,
                                           max_distance=1 * X[node_name].max(),
                                           min_distance=1 * X[node_name].min()))
                sources.append(0)
            else:
                diff.append(self.score_proba_ratio(X[node_name], X_value, cond_dist))
                sources.append(1)
        sources = np.asarray(sources)

        if np.all(sources == 0):
            result = diff
        elif np.all(sources == 1):
            result = diff
        else:
            raise Exception()

        return {node_name: result}

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
        radius = (negative_factors.max() - negative_factors) / (negative_factors.max() - negative_factors.min())
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
            handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
        )
        plt.title("Local Outlier Factor (LOF)")
        plt.show()

    def score(self, X):
        model_factors = self.model_scorer.score(X)
        proximity_factors = self.proximity_scorer.score(X)

        # make zero impact from factors less than 0 since they correspond to inliners
        proximity_factors = np.where(proximity_factors <= 0, 0, proximity_factors)

        # higher the more normal, only
        proximity_outliers_factors = self.agg_funcs["proximity"](proximity_factors, axis=1)

        # any sign can be here, so we take absolute values since distortion from mean is treated as an anomaly
        # model_outliers_factors = np.abs(model_factors).sum(axis=1)

        model_outliers_factors = self.agg_funcs["model"](np.abs(model_factors), axis=1)
        outlier_factors = proximity_outliers_factors + model_outliers_factors

        model_impact = model_outliers_factors / outlier_factors
        proximity_impact = proximity_outliers_factors / outlier_factors

        self.model_impact = np.nanmean(model_impact)
        self.proximity_impact = np.nanmean(proximity_impact)

        return outlier_factors
