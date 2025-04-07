from sklearn.utils.validation import check_is_fitted

from anomaly_detection.displays.results_display import ResultsDisplay
from anomaly_detection.estimators.tabular_estimator import TabularEstimator

from applybn.anomaly_detection.scores.mixed import ODBPScore
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

from sklearn.utils._param_validation import StrOptions

from typing import Literal
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
from applybn.anomaly_detection.anomaly_detection_pipeline import AnomalyDetectionPipeline

class TabularDetector:
    """
    A tabular detector for anomaly detection.

    Examples:
        >>>    d = TabularDetector(target_name="anomaly")
        >>>    d.fit(X=X)
        >>>    d.predict(X=X)
        >>>    preds = d.predict_scores(X)
        >>>    d.plot_result(preds)
    """
    _parameter_constraints = {
        "target_name": [str],
        "score": StrOptions({"mixed", "proximity", "model"})
    }

    _scores = {
        "mixed": ODBPScore,
        "proximity": LocalOutlierScore,
        "model": ModelBasedScore
    }

    def __init__(self, target_name,
                 score: Literal["mixed", "proximity", "model"] = "mixed",
                 additional_score: None | str = "LOF",
                 thresholding_strategy: None | str = "best_from_range",):
        # todo: type hints
        self.target_name = target_name
        self.score = score
        self.additional_score = additional_score
        self.thresholding = thresholding_strategy
        self.y_ = None

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

    def fit(self, X, y=None):
        factory = EstimatorPipelineFactory(task_type="classification")
        factory.estimator_ = TabularEstimator()
        pipeline = factory()
        self.y_ = X.pop(self.target_name)

        pipeline.fit(X)

        self.pipeline_ = AnomalyDetectionPipeline.from_core_pipeline(pipeline)
        return self

    def decision_function(self, X):
        score_obj = self.construct_score(
            bn=self.pipeline_.bn_,
            model_estimation_method="original_modified",
            proximity_estimation_method=self.additional_score,

            model_scorer_args=dict(encoding=self.pipeline_.encoding)

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
            # todo:
            pass

        return np.where(D > best_threshold, 1, 0)

    def plot_result(self, predicted):
        result_display = ResultsDisplay(predicted, self.y_)
        result_display.show()