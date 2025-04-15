from anomaly_detection.scores.proximity_based import IsolationForestScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

from applybn.anomaly_detection.scores.score import Score
from applybn.core.schema import bamt_network

import numpy as np

from typing import Literal
from applybn.anomaly_detection.scores.model_based import (
    BNBasedScore,
    IQRBasedScore,
    CondRatioScore,
    CombinedIQRandProbRatioScore,
)


class ODBPScore(Score):
    _model_estimation_method = {
        "original_modified": BNBasedScore,
        "iqr": IQRBasedScore,
        "cond_ratio": CondRatioScore,
    }

    _proximity_estimation_method = {
        "LOF": LocalOutlierScore,
        "IF": IsolationForestScore,
    }

    def __init__(
        self,
        bn: bamt_network,
        model_estimation_method: dict[
            Literal["cont", "disc"], Literal["original_modified", "iqr", "cond_ratio"]
        ],
        proximity_estimation_method: Literal["LOF", "IF"],
        iqr_sensivity=1.5,
        agg_funcs=None,
        verbose=1,
        model_scorer_args=None,
        additional_scorer_args=None,
    ):
        super().__init__()
        if agg_funcs is None:
            agg_funcs = dict(proximity=np.sum, model=np.sum)

        if model_scorer_args is None:
            model_scorer_args = dict()

        if additional_scorer_args is None:
            additional_scorer_args = dict()

        self.descriptor = bn.descriptor
        if isinstance(model_estimation_method, dict):
            if set(model_estimation_method.values()) == {"iqr", "cond_ratio"}:
                model_scorers = {
                    k: self._model_estimation_method[v](bn=bn, **model_scorer_args)
                    for k, v in model_estimation_method.items()
                }
                self.model_scorer = CombinedIQRandProbRatioScore(
                    scores=model_scorers, bn=bn, **model_scorer_args
                )
        else:
            self.model_scorer = self._model_estimation_method[model_estimation_method](
                bn=bn, **model_scorer_args
            )

        if proximity_estimation_method:
            self.proximity_scorer = self._proximity_estimation_method[
                proximity_estimation_method
            ](**additional_scorer_args)
        else:
            self.proximity_scorer = None

        self.agg_funcs = agg_funcs

        self.proximity_impact = 0
        self.model_impact = 0

        self.iqr_sensivity = iqr_sensivity
        self.verbose = verbose

    def __repr__(self):
        return f"ODBP Score (proximity={self.proximity_scorer})"

    def separate_cont_disc(self, X):
        data_types = self.descriptor["types"]
        cont_vals = ["cont"]
        disc_vals = ["disc", "disc_num"]

        disc = list(filter(lambda key: data_types.get(key) in disc_vals, data_types))
        cont = list(filter(lambda key: data_types.get(key) in cont_vals, data_types))
        return X[disc], X[cont]

    def score(self, X):
        model_factors = self.model_scorer.score(X)
        if self.proximity_scorer:
            proximity_factors = self.proximity_scorer.score(X)
        else:
            proximity_factors = np.zeros_like(model_factors)

        # make zero impact from factors less than 0 since they correspond to inliners
        proximity_factors = np.where(proximity_factors <= 0, 0, proximity_factors)

        # higher the more normal, only
        proximity_outliers_factors = self.agg_funcs["proximity"](
            proximity_factors, axis=1
        )

        # any sign can be here, so we take absolute values since distortion from mean is treated as an anomaly
        # model_outliers_factors = np.abs(model_factors).sum(axis=1)

        model_outliers_factors = self.agg_funcs["model"](np.abs(model_factors), axis=1)
        outlier_factors = proximity_outliers_factors + model_outliers_factors

        model_impact = model_outliers_factors / outlier_factors
        proximity_impact = proximity_outliers_factors / outlier_factors

        self.model_impact = np.nanmean(model_impact)
        self.proximity_impact = np.nanmean(proximity_impact)

        return outlier_factors
