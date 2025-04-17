from applybn.anomaly_detection.scores.proximity_based import IsolationForestScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

from applybn.anomaly_detection.scores.score import Score
from applybn.core.schema import bamt_network

import numpy as np
import pandas as pd

from typing import Literal
from applybn.anomaly_detection.scores.model_based import (
    BNBasedScore,
    IQRBasedScore,
    CondRatioScore,
    CombinedIQRandProbRatioScore,
)


class ODBPScore(Score):
    """
    A class for Outlier Detection using Bayesian Networks and Proximity-based scoring.

    This class combines model-based scoring (using Bayesian networks) and proximity-based scoring
    (e.g., Local Outlier Factor or Isolation Forest) to detect anomalies in data.
    """

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
        iqr_sensivity: float = 1.5,
        agg_funcs: dict =None,
        verbose: int =1,
        model_scorer_args:dict=None,
        additional_scorer_args:dict=None,
    ):
        """
        Initializes the ODBPScore object.

        Args:
            bn: The Bayesian network used for scoring.
            model_estimation_method: The method for model-based scoring, specified separately for
                continuous ("cont") and discrete ("disc") variables.
            proximity_estimation_method: The method for proximity-based scoring ("LOF" or "IF").
            iqr_sensivity: Sensitivity factor for IQR-based scoring. Default is 1.5.
            agg_funcs: Aggregation functions for combining scores. Default is None.
            verbose: Verbosity level for logging. Default is 1.
            model_scorer_args: Additional arguments for the model-based scorer. Default is None.
            additional_scorer_args: Additional arguments for the proximity-based scorer. Default is None.
        """
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
                raise NotImplementedError("Only iqr and cond_ratio mixins are supported.")
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
        """
        Returns a string representation of the ODBPScore object.

        Returns:
            str: A string representation of the object.
        """
        return f"ODBP Score (proximity={self.proximity_scorer})"

    def separate_cont_disc(self, X: pd.DataFrame):
        """
        Separates the input data into continuous and discrete variables.

        Args:
            X: The input data.

        Returns:
            tuple: A tuple containing two DataFrames: one for discrete variables and one for continuous variables.
        """
        data_types = self.descriptor["types"]
        cont_vals = ["cont"]
        disc_vals = ["disc", "disc_num"]

        disc = list(filter(lambda key: data_types.get(key) in disc_vals, data_types))
        cont = list(filter(lambda key: data_types.get(key) in cont_vals, data_types))
        return X[disc], X[cont]

    def score(self, X: pd.DataFrame):
        """
        Computes the outlier scores for the input data.

        Combines model-based and proximity-based scores to compute the final outlier score.
        Deals with continuous and discrete variables separately.

        Args:
            X: The input data.

        Returns:
            np.ndarray: An array of outlier scores for the input data.
        """
        model_factors = self.model_scorer.score(X)
        if self.proximity_scorer:
            proximity_factors = self.proximity_scorer.score(X)
        else:
            proximity_factors = np.zeros_like(model_factors)

        # Make zero impact from factors less than 0 since they correspond to inliers
        proximity_factors = np.where(proximity_factors <= 0, 0, proximity_factors)

        # Higher values indicate more normal data
        proximity_outliers_factors = self.agg_funcs["proximity"](
            proximity_factors, axis=1
        )

        # Take absolute values since distortion from the mean is treated as an anomaly
        model_outliers_factors = self.agg_funcs["model"](np.abs(model_factors), axis=1)
        outlier_factors = proximity_outliers_factors + model_outliers_factors

        model_impact = model_outliers_factors / outlier_factors
        proximity_impact = proximity_outliers_factors / outlier_factors

        self.model_impact = np.nanmean(model_impact)
        self.proximity_impact = np.nanmean(proximity_impact)

        return outlier_factors
