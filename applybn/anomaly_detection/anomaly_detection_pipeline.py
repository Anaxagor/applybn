import pandas as pd

from applybn.core.pipelines import CorePipeline
from functools import wraps
from copy import deepcopy


class AnomalyDetectionPipeline(CorePipeline):
    """
    A pipeline for anomaly detection that extends the CorePipeline.

    This class provides additional functionality for scoring and encoding,
    including the ability to temporarily disable preprocessing during scoring.
    """

    @staticmethod
    def _score_context(method: callable):
        """
        A decorator to temporarily disable the preprocessor step in the pipeline during scoring.

        Args:
            method: The method to be wrapped.

        Returns:
            callable: The wrapped method with the preprocessor step temporarily disabled.
        """
        @wraps(method)
        def wrapper(pipeline, *args, **kwargs):
            """
            Wrapper function to disable and restore the preprocessor step.

            Args:
                pipeline: The pipeline instance.
                *args: Positional arguments for the wrapped method.
                **kwargs: Keyword arguments for the wrapped method.

            Returns:
                Any: The result of the wrapped method.
            """
            original_first_step = deepcopy(pipeline.steps[0])
            try:
                # Temporarily disable the first step
                pipeline.steps[0] = ["preprocessor", "passthrough"]
                return method(pipeline, *args, **kwargs)
            finally:
                # Restore the original step
                pipeline.steps[0] = original_first_step

        return wrapper

    @_score_context
    def score(self, X:pd.DataFrame, y=None, sample_weight=None, **params):
        """
        Computes the score for the given data.

        Temporarily disables the preprocessor step during scoring.

        Args:
            X: The input data.
            y: The target values. Not used.
            sample_weight: Sample weights. Default is None.
            **params: Additional parameters for scoring.

        Returns:
            float: The computed score.
        """
        return super(AnomalyDetectionPipeline, self).score(
            X, y, sample_weight, **params
        )

    @classmethod
    def from_core_pipeline(cls, core_pipeline: CorePipeline):
        """
        Creates an AnomalyDetectionPipeline instance from a CorePipeline.

        Args:
            core_pipeline: The core pipeline to convert.

        Returns:
            AnomalyDetectionPipeline: The created anomaly detection pipeline.
        """
        return cls(core_pipeline.steps)

    @property
    def encoding(self):
        """
        Retrieves the encoding used by the preprocessor step.

        Returns:
            Any: The encoder used in the preprocessor step.
        """
        return self.steps[0][1].coder