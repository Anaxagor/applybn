import pandas as pd
from sklearn.utils.validation import check_is_fitted

from applybn.anomaly_detection.displays.results_display import ResultsDisplay
from applybn.anomaly_detection.estimators.tabular_estimator import TabularEstimator

from applybn.anomaly_detection.scores.mixed import ODBPScore
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

from sklearn.utils._param_validation import StrOptions, Options

from typing import Literal
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
from applybn.anomaly_detection.anomaly_detection_pipeline import (
    AnomalyDetectionPipeline,
)


class TabularDetector:
    """
    A tabular detector for anomaly detection.

    This class provides methods for fitting a pipeline, scoring data, and predicting anomalies
    in tabular datasets. It supports multiple scoring methods, including mixed, proximity-based,
    and model-based scoring.
    """

    _parameter_constraints = {
        "target_name": [str, None],
        "score": StrOptions({"mixed", "proximity", "model"}),
        "additional_score": Options(options={StrOptions({"LOF"}), None}, type=str),
        "thresholding_strategy": Options(
            options={StrOptions({"best_from_range"}), None}, type=str
        ),
        "model_estimation_method": [dict],
        "verbose": [int],
    }

    _scores = {
        "mixed": ODBPScore,
        "proximity": LocalOutlierScore,
        "model": ModelBasedScore,
    }

    def __init__(
        self,
        target_name: None | str = None,
        score: Literal["mixed", "proximity", "model"] = "mixed",
        additional_score: None | str = "LOF",
        thresholding_strategy: None | str = "best_from_range",
        model_estimation_method: (
            None
            | str
            | dict[
                Literal["cont", "disc"],
                Literal["original_modified", "iqr", "cond_ratio"],
            ]
        ) = None,
        verbose: int = 1,
    ):
        """
        Initializes the TabularDetector object.

        Args:
            target_name: The name of the target column in the dataset.
            score: The scoring method to use ("mixed", "proximity", or "model").
            additional_score: The additional proximity-based scoring method (e.g., "LOF").
            thresholding_strategy: The strategy for thresholding scores (e.g., "best_from_range").
            model_estimation_method: The method for model-based scoring, specified separately
                for continuous and discrete variables.
            verbose: The verbosity level for logging. Default is 1.
        """
        if model_estimation_method is None:
            model_estimation_method = {"cont": "iqr", "disc": "cond_ratio"}

        self.target_name = target_name
        self.score = score
        self.additional_score = additional_score
        self.thresholding = thresholding_strategy
        self.model_estimation_method = model_estimation_method
        self.y_ = None
        self.verbose = verbose

    @property
    def impacts(self):
        """
        Returns a dictionary of impact scores from the pipeline's scorer.

            This method provides access to the proximity and model impact values
            calculated by the underlying scoring mechanism within the pipeline.

            Parameters:
                None

            Returns:
                dict: A dictionary containing 'proximity' and 'model' keys, with their
                      corresponding impact scores as values.
        """
        return {
            "proximity": self.pipeline_.scorer.proximity_impact,
            "model": self.pipeline_.scorer.model_impact,
        }

    def _is_fitted(self):
        """
        Checks whether the detector is fitted or not.

        Returns:
            bool: True if the detector is fitted, False otherwise.
        """
        return True if "pipeline_" in self.__dict__ else False

    def __getattr__(self, attr: str):
        """
        Delegates attribute access to the pipeline if the attribute is not found.

        Args:
            attr: The name of the attribute.

        Returns:
            Any: The value of the attribute.

        Raises:
            NotFittedError: If the pipeline is not fitted.
        """
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            if self._is_fitted():
                return getattr(self.pipeline_, attr)
            else:
                raise NotFittedError("BN Estimator has not been fitted.")

    def construct_score(self, **scorer_args):
        """
        Constructs a scoring object based on the selected scoring method.

        Args:
            **scorer_args: Additional arguments for the scoring object.

        Returns:
            Score: The constructed scoring object.
        """
        score_class = self._scores[self.score]
        score_obj = score_class(**scorer_args)
        return score_obj

    def _validate_methods(self):
        """
        Validates that the model estimation method matches the data types.

        Raises:
            ValueError: If the estimation method is unknown.
            TypeError: If the estimation method is incompatible with the data types.
        """
        if isinstance(self.model_estimation_method, dict):
            return  # Custom methods are allowed

        method = self.model_estimation_method
        node_types = set(self.descriptor["types"].values())

        # Define method compatibility
        method_compatibility = {
            "iqr": {"cont"},  # IQR only works with continuous data
            "cond_ratio": {
                "disc",
                "disc_num",
            },  # Conditional ratio only works with discrete data
            "original_modified": {"disc", "disc_num", "cont"},
        }

        # Check if method is known
        if method not in method_compatibility:
            raise ValueError(f"Unknown estimation method: {method}")

        # Check for incompatible data types
        incompatible_types = node_types - method_compatibility[method]
        if incompatible_types:
            raise TypeError(
                f"Method '{method}' cannot work with {', '.join(incompatible_types)} data types. "
                f"Compatible types: {', '.join(method_compatibility[method])}"
            )

    def _validate_target_name(self, X):
        """
        Validates that the target name exists in the columns of the input DataFrame.

            Args:
                X: The input DataFrame.

            Returns:
                bool: True if the target name is valid and present in the DataFrame's
                      columns, False otherwise.  Raises a KeyError if target_name is set
                      but not found in X.
        """
        if self.target_name is not None:
            if self.target_name not in X.columns:
                raise KeyError(
                    f"Target name '{self.target_name}' is not present in {X.columns.tolist()}."
                )
            else:
                return True
        return False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the anomaly detection pipeline to the data.

        Args:
            X: The input data.
            y: The target values. Not used.

        Returns:
            TabularDetector: The fitted detector.

        Raises:
            KeyError: If the target column is not found in the input data.
        """
        X_ = X.copy()
        if self._validate_target_name(X):
            self.y_ = X_.pop(self.target_name)

        factory = EstimatorPipelineFactory(task_type="classification")
        factory.estimator_ = TabularEstimator()
        pipeline = factory()

        ad_pipeline = AnomalyDetectionPipeline.from_core_pipeline(pipeline)

        ad_pipeline.fit(X_)

        self.pipeline_ = ad_pipeline
        return self

    def decision_function(self, X: pd.DataFrame):
        """
        Computes the anomaly scores for the input data.

        Args:
            X: The input data.

        Returns:
            np.ndarray: The computed anomaly scores.
        """
        self._validate_methods()
        score_obj = self.construct_score(
            bn=self.pipeline_.bn_,
            model_estimation_method=self.model_estimation_method,
            proximity_estimation_method=self.additional_score,
            model_scorer_args=dict(
                encoding=self.pipeline_.encoding, verbose=self.verbose
            ),
            additional_scorer_args=dict(verbose=self.verbose),
        )
        self.pipeline_.set_params(bn_estimator__scorer=score_obj)
        scores = self.pipeline_.score(X)
        return scores

    @staticmethod
    def threshold_search_supervised(y: np.ndarray, y_pred: np.ndarray):
        """
        Searches for the best threshold to maximize the F1 score.

        Args:
            y : The true labels.
            y_pred: The predicted scores.

        Returns:
            float: The best threshold.
        """
        thresholds = np.linspace(1, y_pred.max(), 100)
        eval_scores = []

        for t in thresholds:
            outlier_scores_thresholded = np.where(y_pred < t, 0, 1)
            eval_scores.append(f1_score(y, outlier_scores_thresholded))

        return thresholds[np.argmax(eval_scores)]

    def predict_scores(self, X: pd.DataFrame):
        """
        Predicts the anomaly scores for the input data.

        Args:
            X: The input data.

        Returns:
            np.ndarray: The predicted anomaly scores.
        """
        check_is_fitted(self)
        return self.decision_function(X)

    def predict(self, X: pd.DataFrame):
        """
        Predicts whether each data point is an anomaly or not.

        Args:
            X: The input data.

        Returns:
            np.ndarray: An array of binary predictions (1 for anomaly, 0 for normal).

        Raises:
            NotImplementedError: If unsupervised thresholding is not implemented.
        """
        check_is_fitted(self)
        X_ = X.copy()
        if self._validate_target_name(X):
            X_.drop(columns=[self.target_name], inplace=True)

        D = self.decision_function(X_)
        if self.y_ is not None:
            best_threshold = self.threshold_search_supervised(self.y_, D)
        else:
            raise NotImplementedError(
                "Unsupervised thresholding is not implemented yet."
                "Please specify a target column to use supervised thresholding or use predict_scores."
            )

        return np.where(D > best_threshold, 1, 0)

    def plot_result(self, predicted: np.ndarray | pd.Series):
        """
        Plots the results of the anomaly detection.

        Args:
            predicted: The predicted labels.
        """
        result_display = ResultsDisplay(predicted, self.y_)
        result_display.show()
