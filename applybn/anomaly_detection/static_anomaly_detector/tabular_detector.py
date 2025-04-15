from sklearn.utils.validation import check_is_fitted

from anomaly_detection.displays.results_display import ResultsDisplay
from anomaly_detection.estimators.tabular_estimator import TabularEstimator

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
        target_name=None,
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
        verbose=1,
    ):
        if model_estimation_method is None:
            model_estimation_method = {"cont": "iqr", "disc": "cond_ratio"}

        self.target_name = target_name
        self.score = score
        self.additional_score = additional_score
        self.thresholding = thresholding_strategy
        self.model_estimation_method = model_estimation_method
        self.y_ = None
        self.verbose = verbose

    def _is_fitted(self):
        """
        Checks whether the detector is fitted or not by checking "pipeline_" key if __dict__.
        This has to be done because check_is_fitted(self) does not imply correct and goes into recursion because of
        delegating strategy in getattr method.
        """
        return True if "pipeline_" in self.__dict__ else False

    def __getattr__(self, attr: str):
        """If attribute is not found in the pipeline, look in the last step of the pipeline."""
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            if self._is_fitted():
                return getattr(self.pipeline_, attr)
            else:
                raise NotFittedError("BN Estimator has not been fitted.")

    def construct_score(self, **scorer_args):
        score_class = self._scores[self.score]
        score_obj = score_class(**scorer_args)
        return score_obj

    def _validate_methods(self):
        """Validate that the model estimation method matches the data types."""
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

    def fit(self, X, y=None):
        if self.target_name is not None:
            if self.target_name not in X.columns:
                raise KeyError(
                    f"Target name '{self.target_name}' is not present in {X.columns.tolist()}."
                )
            else:
                self.y_ = X.pop(self.target_name)

        factory = EstimatorPipelineFactory(task_type="classification")
        factory.estimator_ = TabularEstimator()
        pipeline = factory()

        ad_pipeline = AnomalyDetectionPipeline.from_core_pipeline(pipeline)

        ad_pipeline.fit(X)

        self.pipeline_ = ad_pipeline
        return self

    def decision_function(self, X):
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
    def threshold_search_supervised(y, y_pred):
        thresholds = np.linspace(1, y_pred.max(), 100)
        eval_scores = []

        for t in thresholds:
            outlier_scores_thresholded = np.where(y_pred < t, 0, 1)
            eval_scores.append(f1_score(y, outlier_scores_thresholded))

        return thresholds[np.argmax(eval_scores)]

    def predict_scores(self, X):
        check_is_fitted(self)
        return self.decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        D = self.decision_function(X)
        if self.y_ is not None:
            best_threshold = self.threshold_search_supervised(self.y_, D)
        else:
            raise NotImplementedError(
                "Unsupervised thresholding is not implemented yet."
                "Please specify a target column to use supervised thresholding."
            )

        return np.where(D > best_threshold, 1, 0)

    def plot_result(self, predicted):
        result_display = ResultsDisplay(predicted, self.y_)
        result_display.show()
